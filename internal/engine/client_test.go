package engine

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

func f64(v float64) *float64 { return &v }

// render is an intermediate state: "<bos> a b [image x4] c [image x3] <think>".
func render() *mwip.RenderResult {
	r := &mwip.RenderResult{
		RequestID: "r", PromptTokens: 11,
		TokenIDs: []uint32{0, 10, 11, 1 << 30, 1 << 30, 1 << 30, 1 << 30, 12, 1<<30 + 5, 1<<30 + 5, 1<<30 + 5},
		Multimodal: &mwip.Multimodal{PadShift: 1 << 30, Items: []mwip.MediaItem{
			{Index: 0, Modality: "image", Digest: strings.Repeat("a", 64), Source: mwip.MediaSource{Kind: "url", URL: "https://a/0.png"},
				Plan: mwip.MediaPlan{Processor: "qwen_vl.v1", Width: 64, Height: 64, Tokens: 4, NFrames: 1, PatchSize: 16, SpatialMergeSize: 2, TemporalPatchSize: 2}, Placements: []mwip.Placement{{Offset: 3, Length: 4, Modality: "image"}}},
			{Index: 1, Modality: "image", Digest: strings.Repeat("b", 64), Source: mwip.MediaSource{Kind: "url", URL: "https://a/1.png"},
				Plan: mwip.MediaPlan{Processor: "qwen_vl.v1", Width: 96, Height: 32, Tokens: 3, NFrames: 1, PatchSize: 16, SpatialMergeSize: 2, TemporalPatchSize: 2}, Placements: []mwip.Placement{{Offset: 8, Length: 3, Modality: "image"}}},
		}},
		Conversation: mwip.Conversation{ThinkingMode: true, ReasoningEffort: "high", ToolChoice: "auto", ResponseFormat: "text",
			Tools: []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"search"}}`)}},
		InferenceOptions: mwip.InferenceOptions{MaxTokens: 64, Temperature: f64(1), TopP: f64(0.95), StopTokenIDs: []uint32{1},
			OutputConstraint: &mwip.OutputConstraint{Kind: "json", Schema: `{"type":"object"}`}},
		ParsingOptions: mwip.ParsingOptions{ParseToolCalls: true, ThinkingInitialStage: "reasoning_start", StopSequences: []string{"END"}},
	}
	for i := range r.Multimodal.Items {
		it := &r.Multimodal.Items[i]
		it.PlanID = mwip.PlanID(it.Modality, it.Digest, it.Plan)
		p := &it.Placements[0]
		p.Reference = it.PlanID
		p.PadValue = mwip.PadValue(p.Reference)
		for k := p.Offset; k < p.Offset+p.Length; k++ {
			r.TokenIDs[k] = p.PadValue
		}
	}
	return r
}

func collect(t *testing.T, s *Stream) (string, *mwip.GenerateEvent, error) {
	t.Helper()
	var text strings.Builder
	for {
		ev, err := s.Next()
		if errors.Is(err, io.EOF) {
			return text.String(), nil, nil
		}
		if err != nil {
			return text.String(), ev, err
		}
		text.WriteString(ev.Content)
		if ev.Type == mwip.EventFinish || ev.Type == mwip.EventPrefillDone {
			return text.String(), ev, nil
		}
	}
}

// fakeEngine is an mwip Engine answering Generate with a scripted stream.
type fakeEngine struct {
	mwipv1.UnimplementedEngineServer
	script      []*mwipv1.GenerateResponse
	status      error
	block       bool // keep the stream open after the script until canceled
	seen        chan *mwipv1.GenerateRequest
	canceled    chan struct{}
	aborted     chan *mwipv1.AbortRequest
	encoded     chan *mwipv1.EncodeRequest
	encodeReply func(*mwipv1.EncodeRequest) *mwipv1.EncodeResponse
}

func newFake(script ...*mwipv1.GenerateResponse) *fakeEngine {
	return &fakeEngine{script: script, seen: make(chan *mwipv1.GenerateRequest, 4), canceled: make(chan struct{}),
		aborted: make(chan *mwipv1.AbortRequest, 4), encoded: make(chan *mwipv1.EncodeRequest, 4)}
}

func (f *fakeEngine) Generate(req *mwipv1.GenerateRequest, s mwipv1.Engine_GenerateServer) error {
	f.seen <- req
	for _, m := range f.script {
		if err := s.Send(m); err != nil {
			return err
		}
	}
	if f.block {
		<-s.Context().Done()
		close(f.canceled)
		return s.Context().Err()
	}
	return f.status
}

func (f *fakeEngine) Encode(_ context.Context, req *mwipv1.EncodeRequest) (*mwipv1.EncodeResponse, error) {
	f.encoded <- req
	if f.encodeReply != nil {
		return f.encodeReply(req), nil
	}
	it := mwip.InputFromProto(req.GetItem())
	out := &mwipv1.EncodeResponse{EncodeMs: 4, Embedding: &mwipv1.EmbeddingLocation{Uri: "mock://e0/" + it.PlanID, Tokens: int32(it.Plan.EmbeddingTokens()), PlanId: it.PlanID, Encoder: "e0"}}
	return out, nil
}

func TestRejectEncoderPlanAndLocationMismatch(t *testing.T) {
	for _, kind := range []string{"missing", "uri", "plan", "rows"} {
		t.Run(kind, func(t *testing.T) {
			f := newFake()
			f.encodeReply = func(req *mwipv1.EncodeRequest) *mwipv1.EncodeResponse {
				it := mwip.InputFromProto(req.GetItem())
				e := &mwipv1.EmbeddingLocation{Uri: "mock://e/item", Tokens: int32(it.Plan.EmbeddingTokens()), PlanId: it.PlanID}
				switch kind {
				case "missing":
					e = nil
				case "uri":
					e.Uri = ""
				case "plan":
					e.PlanId = "different-plan"
				case "rows":
					e.Tokens++
				}
				return &mwipv1.EncodeResponse{Embedding: e}
			}
			spec := startEngine(t, f)
			c := New()
			defer c.Close()
			in := render().Multimodal.Items[0].Input()
			if _, err := c.Encode(context.Background(), spec, &EncodeStage{Item: &in}); err == nil {
				t.Fatal("invalid encoder result accepted")
			} else {
				var e *Error
				if !errors.As(err, &e) || e.Code != mwip.ErrInternal || !e.Retriable {
					t.Fatalf("contract violation must retry another encoder: %v", err)
				}
			}
		})
	}
}

func (f *fakeEngine) Abort(_ context.Context, req *mwipv1.AbortRequest) (*mwipv1.AbortResponse, error) {
	f.aborted <- req
	return &mwipv1.AbortResponse{Aborted: int32(len(req.GetRequestIds()))}, nil
}

func (f *fakeEngine) GetState(context.Context, *mwipv1.GetStateRequest) (*mwipv1.GetStateResponse, error) {
	return &mwipv1.GetStateResponse{State: &mwipv1.EngineState{RunningRequests: 3, KvUsedTokens: 1024, KvCapacityTokens: 4096}}, nil
}

func startEngine(t *testing.T, f *fakeEngine) registry.Spec {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer(grpc.MaxRecvMsgSize(MaxMessageBytes))
	mwipv1.RegisterEngineServer(srv, f)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	return registry.Spec{ID: "d0", Role: registry.RoleDecode, Target: lis.Addr().String()}
}

func delta(content, reasoning string, ids ...uint32) *mwipv1.GenerateResponse {
	return &mwipv1.GenerateResponse{Event: &mwipv1.GenerateResponse_Delta{Delta: &mwipv1.Delta{Content: content, Reasoning: reasoning, TokenIds: ids}}}
}

// A normal leg over gRPC: the engine receives the intermediate state as is
// and streams parsed deltas and a finish.
func TestMWIPGenerateCarriesTheIntermediateState(t *testing.T) {
	f := newFake(delta("", "think", 5), delta("Hi", "", 6),
		&mwipv1.GenerateResponse{Event: &mwipv1.GenerateResponse_Finish{Finish: &mwipv1.Finish{
			Reason: mwipv1.FinishReason_FINISH_REASON_STOP, StopSequence: "END",
			Usage: &mwipv1.Usage{PromptTokens: 11, CompletionTokens: 2, CachedTokens: 3, ReasoningTokens: 1}}}})
	spec := startEngine(t, f)
	c := New()
	defer c.Close()
	s, err := c.Generate(context.Background(), spec, &Stage{Render: render(), AttemptID: "r-1", EngineModel: "m",
		Disagg: mwip.Disagg{Mode: mwip.ModeNormal}, TimeoutMS: 500, SessionKey: "user-1", Priority: "high"})
	if err != nil {
		t.Fatal(err)
	}
	text, fin, err := collect(t, s)
	s.Close()
	if err != nil || text != "Hi" || fin.Reason != "stop" || fin.StopReason != "END" || fin.Usage.PromptTokens != 11 || fin.Usage.Cached() != 3 || fin.Usage.Reasoning() != 1 {
		t.Fatalf("text %q finish %+v err %v", text, fin, err)
	}
	req := <-f.seen
	po, io, mm := req.GetParsingOptions(), req.GetInferenceOptions(), req.GetMultimodal()
	if req.GetRequestId() != "r-1" || req.GetModel() != "m" || len(req.GetTokenIds()) != 11 ||
		req.GetTimeoutMs() != 500 || req.GetSessionKey() != "user-1" || req.GetPriority() != mwipv1.Priority_PRIORITY_HIGH {
		t.Fatalf("request fields %v", req)
	}
	if po.GetThinkingInitialStage() != mwipv1.ThinkingStage_THINKING_STAGE_REASONING_START || !po.GetParseToolCalls() || po.GetStopSequences()[0] != "END" ||
		io.GetOutputConstraint().GetKind() != mwipv1.ConstraintKind_CONSTRAINT_KIND_JSON || io.GetTemperature() != 1 ||
		req.GetConversation().GetTools()[0].GetFunction().GetName() != "search" || req.GetDisagg().GetMode() != mwipv1.DisaggMode_DISAGG_MODE_NORMAL {
		t.Fatalf("options %v %v %v", po, io, req.GetConversation())
	}
	want := render().Multimodal.Items[1]
	if len(mm.GetItems()) != 2 || mm.GetItems()[1].GetPlacements()[0].GetPadValue() != want.Placements[0].PadValue || mm.GetItems()[1].GetPlanId() != want.PlanID || int(mm.GetItems()[1].GetPlan().GetWidth()) != want.Plan.Width {
		t.Fatalf("media %v", mm)
	}
}

func TestMWIPLegsAndErrors(t *testing.T) {
	split := func(role, handshake string) mwip.Disagg {
		return mwip.Disagg{Mode: mwip.ModeSplit, Role: role, Handshake: handshake, Room: "r-1"}
	}
	detail, _ := status.New(codes.FailedPrecondition, "decode only").WithDetails(&mwipv1.Error{
		Code: mwipv1.ErrorCode_ERROR_CODE_NORMAL_REJECTED, Message: "decode only", Retriable: true})
	for _, tc := range []struct {
		name   string
		fake   *fakeEngine
		disagg mwip.Disagg
		check  func(ev *mwip.GenerateEvent, err error) bool
	}{
		{"prefill pull leg ends with its kv handle; deltas of a prefill leg are dropped", newFake(delta("x", "", 1),
			&mwipv1.GenerateResponse{Event: &mwipv1.GenerateResponse_PrefillDone{PrefillDone: &mwipv1.PrefillDone{
				PromptTokens: 11, KvHandle: &mwipv1.KVHandle{InstanceId: "p0", Url: "http://p0", Room: "r-1", Tokens: 11}}}}),
			split("prefill", mwip.HandshakePull),
			func(ev *mwip.GenerateEvent, err error) bool {
				return err == nil && ev.Type == mwip.EventPrefillDone && ev.Handle.Room == "r-1" && ev.PromptTokens == 11
			}},
		{"push decode leg acknowledges with its receiver", newFake(&mwipv1.GenerateResponse{Event: &mwipv1.GenerateResponse_Accepted{Accepted: &mwipv1.Accepted{
			KvReceiver: &mwipv1.KVReceiver{Url: "http://d0", Room: "r-1", CachedTokens: 64}}}}),
			split("decode", mwip.HandshakePush),
			func(ev *mwip.GenerateEvent, err error) bool {
				return err == nil && ev.Type == mwip.EventAccepted && ev.Receiver.WorkerID == "d0" && ev.CachedTokens == 64
			}},
		{"error event", newFake(&mwipv1.GenerateResponse{Event: &mwipv1.GenerateResponse_Error{Error: &mwipv1.Error{
			Code: mwipv1.ErrorCode_ERROR_CODE_CAPACITY_EXHAUSTED, Message: "full", Retriable: true}}}),
			mwip.Disagg{Mode: mwip.ModeNormal},
			func(ev *mwip.GenerateEvent, err error) bool {
				var e *Error
				return errors.As(err, &e) && e.Code == mwip.ErrCapacity && e.Retriable && ev.Type == mwip.EventError
			}},
		{"status with an mwip error detail", &fakeEngine{status: detail.Err(), seen: make(chan *mwipv1.GenerateRequest, 1)},
			mwip.Disagg{Mode: mwip.ModeNormal},
			func(_ *mwip.GenerateEvent, err error) bool { return IsCode(err, mwip.ErrNormalRejected) }},
		{"status code only", &fakeEngine{status: status.Error(codes.ResourceExhausted, "busy"), seen: make(chan *mwipv1.GenerateRequest, 1)},
			mwip.Disagg{Mode: mwip.ModeNormal},
			func(_ *mwip.GenerateEvent, err error) bool {
				var e *Error
				return errors.As(err, &e) && e.Code == mwip.ErrCapacity && e.Retriable
			}},
		{"invalid argument is final", &fakeEngine{status: status.Error(codes.InvalidArgument, "no"), seen: make(chan *mwipv1.GenerateRequest, 1)},
			mwip.Disagg{Mode: mwip.ModeNormal},
			func(_ *mwip.GenerateEvent, err error) bool {
				var e *Error
				return errors.As(err, &e) && e.Code == mwip.ErrBadRequest && !e.Retriable
			}},
		{"stream ended without finish", newFake(), mwip.Disagg{Mode: mwip.ModeNormal},
			func(_ *mwip.GenerateEvent, err error) bool {
				var e *Error
				return errors.As(err, &e) && e.Transport && e.Retriable
			}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			spec := startEngine(t, tc.fake)
			c := New()
			defer c.Close()
			s, err := c.Generate(context.Background(), spec, &Stage{Render: render(), AttemptID: "r-1", Disagg: tc.disagg})
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			ev, err := s.Next()
			if !tc.check(ev, err) {
				t.Fatalf("event %+v err %v", ev, err)
			}
			if err == nil && (ev.Type == mwip.EventPrefillDone || ev.Type == mwip.EventFinish) {
				if _, err := s.Next(); !errors.Is(err, io.EOF) {
					t.Fatalf("after the final event: %v", err)
				}
			}
		})
	}
}

// An unreachable instance is a transport error, retried elsewhere and
// counted against its health.
func TestMWIPUnreachableInstance(t *testing.T) {
	lis, _ := net.Listen("tcp", "127.0.0.1:0")
	addr := lis.Addr().String()
	lis.Close()
	c := New()
	defer c.Close()
	s, err := c.Generate(context.Background(), registry.Spec{ID: "gone", Target: "http://" + addr}, &Stage{Render: render(), AttemptID: "r"})
	if err == nil {
		_, err = s.Next()
		s.Close()
	}
	var e *Error
	if !errors.As(err, &e) || !e.Transport || !e.Retriable {
		t.Fatalf("want a transport error, got %v", err)
	}
}

// Closing a stream cancels the call: the engine sees its context end and
// stops the request.
func TestMWIPCloseCancelsTheCall(t *testing.T) {
	f := newFake(delta("a", "", 1))
	f.block = true
	spec := startEngine(t, f)
	c := New()
	defer c.Close()
	s, err := c.Generate(context.Background(), spec, &Stage{Render: render(), AttemptID: "r"})
	if err != nil {
		t.Fatal(err)
	}
	if ev, err := s.Next(); err != nil || ev.Content != "a" {
		t.Fatalf("first delta %+v %v", ev, err)
	}
	s.Close()
	select {
	case <-f.canceled:
	case <-time.After(5 * time.Second):
		t.Fatal("the engine did not see the cancellation")
	}
}

func TestMWIPEncodeAbortAndState(t *testing.T) {
	f := newFake()
	spec := startEngine(t, f)
	c := New()
	defer c.Close()
	r := render()
	for i, it := range r.Multimodal.Items {
		in := it.Input()
		res, err := c.Encode(context.Background(), spec, &EncodeStage{RequestID: "r", AttemptID: "r-enc", Model: "m", EngineModel: "engine-weights", Item: &in})
		if err != nil || res.Embedding == nil || res.Embedding.PlanID != it.PlanID || res.Embedding.Tokens != it.Plan.EmbeddingTokens() || res.Embedding.URI != "mock://e0/"+it.PlanID || res.EncodeMS != 4 {
			t.Fatalf("encode item %d: %+v %v", i, res, err)
		}
		seen := <-f.encoded
		if seen.GetModel() != "engine-weights" || seen.GetItem().GetPlanId() != it.PlanID {
			t.Fatalf("single-item encoder input: %+v", seen)
		}
	}
	if err := c.Abort(context.Background(), spec, "r-1"); err != nil {
		t.Fatal(err)
	}
	if a := <-f.aborted; len(a.GetRequestIds()) != 1 || a.GetRequestIds()[0] != "r-1" {
		t.Fatalf("abort %v", a)
	}
	st, err := c.State(context.Background(), spec)
	if err != nil || st.RunningRequests != 3 || st.KVUsedTokens != 1024 || st.KVCapacityTokens != 4096 {
		t.Fatalf("state %+v %v", st, err)
	}
}
