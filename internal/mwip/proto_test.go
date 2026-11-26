package mwip

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"google.golang.org/protobuf/proto"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
)

// wire sends a message through its binary encoding.
func wire[M proto.Message](t *testing.T, m M, into M) M {
	t.Helper()
	b, err := proto.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := proto.Unmarshal(b, into); err != nil {
		t.Fatal(err)
	}
	return into
}

// same compares two values by their JSON form (tool definitions are raw
// JSON whose key order is not significant).
func same(t *testing.T, what string, got, want any) {
	t.Helper()
	norm := func(v any) any {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		var out any
		if err := json.Unmarshal(b, &out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	if g, w := norm(got), norm(want); !reflect.DeepEqual(g, w) {
		gb, _ := json.Marshal(g)
		wb, _ := json.Marshal(w)
		t.Fatalf("%s differs after the wire:\n got %s\nwant %s", what, gb, wb)
	}
}

// Requests built from real tokenizer renders (testdata: DeepSeek V4.1 with
// tools, an image and a video; Qwen3-VL with media and a JSON constraint;
// Qwen3 with a required tool call) keep every field through the proto, with
// the execution fields of every handshake.
func TestRenderedRequestsRoundTripThroughProto(t *testing.T) {
	files, _ := filepath.Glob("testdata/*.json")
	if len(files) < 3 {
		t.Fatalf("fixtures: %v", files)
	}
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		var rr RenderResult
		if err := json.Unmarshal(b, &rr); err != nil {
			t.Fatal(err)
		}
		media := rr.Multimodal.Clone()
		if media != nil {
			for i := range media.Items {
				for _, p := range media.Items[i].Placements {
					media.Items[i].Embeddings = append(media.Items[i].Embeddings, EmbeddingArtifact{
						Reference: p.Reference, URI: "mock://enc-0/" + p.Reference, Tokens: p.Length, Encoder: "enc-0",
					})
				}
			}
		}
		for _, d := range []Disagg{
			{Mode: ModeNormal},
			{Mode: ModeSplit, Role: "decode", Handshake: HandshakePush, Room: "mw-1", BootstrapRoom: 4821930213,
				Prefill: &Peer{WorkerID: "prefill-0", Target: "prefill-0:9001", BootstrapAddr: "http://prefill-0:9000", EngineID: "p0"},
				Decode:  &Peer{WorkerID: "decode-0", Target: "decode-0:9001"}},
			{Mode: ModeSplit, Role: "prefill", Handshake: HandshakePush, Room: "mw-1",
				Receiver: &KVReceiver{WorkerID: "decode-0", URL: "http://decode-0:9000", Room: "mw-1", CachedTokens: 128}},
			{Mode: ModeSplit, Role: "decode", Handshake: HandshakePull, Room: "mw-1",
				Handle: &KVHandle{WorkerID: "prefill-0", URL: "http://prefill-0:9000", Room: "mw-1", Tokens: 897}},
			{Mode: ModeSplit, Role: "decode", Handshake: HandshakeBootstrap, Room: "mw-1"},
		} {
			req := &GenerateRequest{
				RequestID: "mw-1", Model: "engine-model", TokenIDs: rr.TokenIDs, Multimodal: media,
				Conversation: rr.Conversation, InferenceOptions: rr.InferenceOptions, ParsingOptions: rr.ParsingOptions,
				Disagg: d, TimeoutMS: 29750, SessionKey: "user-42", Priority: "high",
			}
			p, err := RequestToProto(req)
			if err != nil {
				t.Fatal(err)
			}
			back, err := RequestFromProto(wire(t, p, &mwipv1.GenerateRequest{}))
			if err != nil {
				t.Fatal(err)
			}
			same(t, filepath.Base(f)+" "+d.Handshake+" "+d.Role, back, req)
		}
	}
}

func TestEventsRoundTripThroughProto(t *testing.T) {
	ts := 2.5
	events := []*GenerateEvent{
		{Type: EventAccepted, Receiver: &KVReceiver{WorkerID: "decode-0", URL: "http://decode-0:9000", Room: "mw-1", CachedTokens: 896}, CachedTokens: 896},
		{Type: EventPrefillDone, PromptTokens: 1183, CachedTokens: 64, Handle: &KVHandle{WorkerID: "prefill-0", URL: "http://p", Room: "mw-1", Tokens: 1183},
			Timing: &Timing{QueueMS: 1, PrefillMS: 40.5}},
		{Type: EventDelta, Reasoning: "Let me think", TokenIDs: []uint32{671, 672}},
		{Type: EventDelta, ToolCalls: []DeltaToolCall{
			{Index: 0, ID: "call_1", Type: "function", Function: &DeltaFunction{Name: "get_weather"}},
			{Index: 0, Function: &DeltaFunction{Arguments: `{"city":"Paris"}`}},
		}, TokenIDs: []uint32{3816}},
		{Type: EventDelta, Content: "Hello", TokenIDs: []uint32{19923}, Logprobs: []LogprobContent{
			{Token: "Hello", Logprob: -0.05, Bytes: []int{72, 101}, TopLogprobs: []TopLogprob{{Token: "Hi", Logprob: -3, Bytes: []int{72, 105}}}},
		}},
		{Type: EventFinish, Reason: "stop", StopReason: "END", PromptTokens: 1183, CachedTokens: 896,
			Usage: &Usage{PromptTokens: 1183, CompletionTokens: 77, TotalTokens: 1260,
				PromptTokensDetails: &PromptTokensDetails{CachedTokens: 896}, CompletionTokensDetails: &CompletionTokensDetails{ReasoningTokens: 17}},
			Timing: &Timing{TTFTMS: 12, DecodeMS: ts}},
		{Type: EventFinish, Reason: "tool_calls", Usage: &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}, PromptTokens: 10},
		{Type: EventError, Code: ErrCapacity, Message: "full", Retriable: true},
		{Type: EventError, Code: ErrNormalRejected, Message: "no normal requests"},
	}
	for _, ev := range events {
		p := EventToProto(ev)
		if p == nil {
			t.Fatalf("%s did not convert", ev.Type)
		}
		got := EventFromProto(wire(t, p, &mwipv1.GenerateResponse{}))
		same(t, ev.Type, got, ev)
	}
	if EventToProto(&GenerateEvent{Type: EventReady}) != nil || EventFromProto(&mwipv1.GenerateResponse{}) != nil {
		t.Fatal("events that are not on the wire must not convert")
	}
}

func TestEncodeRequestRoundTripThroughProto(t *testing.T) {
	b, _ := os.ReadFile("testdata/qwen3vl_media_json.json")
	var rr RenderResult
	if err := json.Unmarshal(b, &rr); err != nil {
		t.Fatal(err)
	}
	req := &EncodeRequest{RequestID: "mw-1", AttemptID: "mw-1-enc0-1", Model: "qwen3-vl", Items: rr.Multimodal.Items, TimeoutMS: 29000}
	same(t, "encode request", EncodeRequestFromProto(wire(t, EncodeRequestToProto(req), &mwipv1.EncodeRequest{})), req)
}

// Every value of every protocol enum has an in-memory name, so a value
// added to the proto cannot be dropped silently.
func TestEnumsCoverEveryProtoValue(t *testing.T) {
	check := func(name string, values map[int32]string, mapped func(int32) bool) {
		for v, n := range values {
			if v != 0 && !mapped(v) {
				t.Errorf("%s value %s has no in-memory name", name, n)
			}
		}
	}
	check("Role", mwipv1.Role_name, func(v int32) bool { return roles.from[mwipv1.Role(v)] != "" })
	check("Modality", mwipv1.Modality_name, func(v int32) bool { return modalities.from[mwipv1.Modality(v)] != "" })
	check("DisaggMode", mwipv1.DisaggMode_name, func(v int32) bool { return modes.from[mwipv1.DisaggMode(v)] != "" })
	check("Handshake", mwipv1.Handshake_name, func(v int32) bool { return handshakes.from[mwipv1.Handshake(v)] != "" })
	check("ThinkingStage", mwipv1.ThinkingStage_name, func(v int32) bool { return stages.from[mwipv1.ThinkingStage(v)] != "" })
	check("ConstraintKind", mwipv1.ConstraintKind_name, func(v int32) bool { return constraints.from[mwipv1.ConstraintKind(v)] != "" })
	check("FinishReason", mwipv1.FinishReason_name, func(v int32) bool { return finishReasons.from[mwipv1.FinishReason(v)] != "" })
	check("ErrorCode", mwipv1.ErrorCode_name, func(v int32) bool { return errorCodes.from[mwipv1.ErrorCode(v)] != "" })
	check("Priority", mwipv1.Priority_name, func(v int32) bool { return priorities.from[mwipv1.Priority(v)] != "" })
	if RoleFromProto(RoleToProto("decode")) != "decode" || ErrorCodeFromProto(ErrorCodeToProto("nope")) != ErrInternal {
		t.Fatal("role or error code conversion")
	}
}
