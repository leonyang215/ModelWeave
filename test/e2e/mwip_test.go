//go:build e2e

package e2e

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
)

// registryTarget is the gateway's mwip Registry, reachable inside the
// compose network.
var registryTarget = os.Getenv("MW_E2E_REGISTRY")

// registryClient dials the gateway's mwip Registry; ok is false when the
// suite runs outside the compose network.
func registryClient(t *testing.T) (mwipv1.RegistryClient, bool) {
	t.Helper()
	if registryTarget == "" {
		return nil, false
	}
	cc, err := grpc.NewClient(registryTarget, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cc.Close() })
	return mwipv1.NewRegistryClient(cc), true
}

func postEmpty(t *testing.T, url string) map[string]any {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, url, nil)
	r, err := client.Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer r.Body.Close()
	var m map[string]any
	_ = json.NewDecoder(r.Body).Decode(&m)
	if r.StatusCode != 200 {
		t.Fatalf("POST %s: %d %v", url, r.StatusCode, m)
	}
	return m
}

// waitFor polls until cond holds or the deadline passes.
func waitFor(t *testing.T, what string, d time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(d)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(200 * time.Millisecond)
	}
}

// protoEngine is an inference engine written against the mwip proto alone
// (the generated Go code): one Engine server is one instance.
type protoEngine struct {
	mwipv1.UnimplementedEngineServer
	id       string
	model    string
	labels   map[string]string
	target   string
	mu       sync.Mutex
	requests []*mwipv1.GenerateRequest
	full     bool // answer capacity_exhausted
	beating  bool // send heartbeats
	draining bool // report draining in heartbeats
}

func (e *protoEngine) Generate(req *mwipv1.GenerateRequest, s mwipv1.Engine_GenerateServer) error {
	e.mu.Lock()
	e.requests = append(e.requests, req)
	full := e.full
	e.mu.Unlock()
	if full {
		return s.Send(&mwipv1.GenerateResponse{Event: &mwipv1.GenerateResponse_Error{Error: &mwipv1.Error{
			Code: mwipv1.ErrorCode_ERROR_CODE_CAPACITY_EXHAUSTED, Message: "instance is full", Retriable: true}}})
	}
	for _, d := range []*mwipv1.Delta{
		{Reasoning: "Thinking on " + e.id + ".", TokenIds: []uint32{1, 2, 3}},
		{Content: "Answer from " + e.id + ".", TokenIds: []uint32{4}},
		{ToolCalls: []*mwipv1.ToolCallDelta{{Index: 0, Id: "call_e2e", Name: "get_weather"}}, TokenIds: []uint32{5}},
		{ToolCalls: []*mwipv1.ToolCallDelta{{Index: 0, Arguments: `{"city":"Paris"}`}}, TokenIds: []uint32{6, 7}},
	} {
		if err := s.Send(&mwipv1.GenerateResponse{Event: &mwipv1.GenerateResponse_Delta{Delta: d}}); err != nil {
			return err
		}
	}
	return s.Send(&mwipv1.GenerateResponse{Event: &mwipv1.GenerateResponse_Finish{Finish: &mwipv1.Finish{
		Reason: mwipv1.FinishReason_FINISH_REASON_TOOL_CALLS,
		Usage:  &mwipv1.Usage{PromptTokens: int32(len(req.GetTokenIds())), CompletionTokens: 7, ReasoningTokens: 3},
	}}})
}

func (e *protoEngine) set(f func(e *protoEngine)) {
	e.mu.Lock()
	defer e.mu.Unlock()
	f(e)
}

func (e *protoEngine) since(n int) []*mwipv1.GenerateRequest {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]*mwipv1.GenerateRequest(nil), e.requests[n:]...)
}

func (e *protoEngine) count() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.requests)
}

// heartbeat is the instance's registration: its description and load.
func (e *protoEngine) heartbeat() *mwipv1.HeartbeatRequest {
	e.mu.Lock()
	defer e.mu.Unlock()
	return &mwipv1.HeartbeatRequest{
		Instance: &mwipv1.Instance{Id: e.id, Model: e.model, Role: mwipv1.Role_ROLE_UNIFIED, Target: e.target, MaxRunningRequests: 8, Labels: maps.Clone(e.labels)},
		State:    &mwipv1.EngineState{RunningRequests: 0, MaxRunningRequests: 8}, Draining: e.draining,
	}
}

// startProtoEngine serves an instance's Engine service on its own port.
func startProtoEngine(t *testing.T, id string) *protoEngine {
	t.Helper()
	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatal(err)
	}
	e := &protoEngine{id: id, model: "external-engine", target: net.JoinHostPort(localIP(t), fmt.Sprint(lis.Addr().(*net.TCPAddr).Port)), beating: true}
	srv := grpc.NewServer(grpc.MaxRecvMsgSize(64 << 20))
	mwipv1.RegisterEngineServer(srv, e)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	return e
}

// localIP is this container's address on the compose network.
func localIP(t *testing.T) string {
	t.Helper()
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range addrs {
		if n, ok := a.(*net.IPNet); ok && !n.IP.IsLoopback() && n.IP.To4() != nil {
			return n.IP.String()
		}
	}
	t.Fatal("no network address")
	return ""
}

// An engine implemented from the proto alone runs two instances of
// external-engine, each an Engine server registered by its own heartbeats
// (invalid registrations are rejected with the reason). It receives the
// intermediate state the tokenizer rendered, and its parsed output is
// formatted for every client. A full instance makes the gateway retry on
// the other one, a draining instance gets no new requests, and an instance
// that deregisters is gone at once and registers again with its next
// heartbeat.
func TestEngineImplementedFromTheProto(t *testing.T) {
	reg, ok := registryClient(t)
	if !ok {
		t.Skip("MW_E2E_REGISTRY not set: the gateway cannot reach this process")
	}
	u0, u1 := startProtoEngine(t, "e2e-u0"), startProtoEngine(t, "e2e-u1")
	engines := map[string]*protoEngine{"e2e-u0": u0, "e2e-u1": u1}
	ctx := metadata.AppendToOutgoingContext(context.Background(), "x-modelweave-token", ctlToken)

	// The gateway rejects what it cannot serve, with the reason.
	for _, c := range []struct {
		inst *mwipv1.Instance
		code codes.Code
		want string
	}{
		{&mwipv1.Instance{Id: "e2e-bad", Model: "no-such-model", Role: mwipv1.Role_ROLE_UNIFIED, Target: u0.target}, codes.NotFound, `serves no model "no-such-model"`},
		{&mwipv1.Instance{Id: "e2e-bad", Model: "external-engine", Role: mwipv1.Role_ROLE_UNIFIED}, codes.InvalidArgument, "target is required"},
		{&mwipv1.Instance{Id: "e2e-bad", Model: "external-engine", Target: u0.target}, codes.InvalidArgument, "role is required"},
	} {
		_, err := reg.Heartbeat(ctx, &mwipv1.HeartbeatRequest{Instance: c.inst})
		if status.Code(err) != c.code || !strings.Contains(status.Convert(err).Message(), c.want) {
			t.Fatalf("%v: %v, want %s %q", c.inst, err, c.code, c.want)
		}
	}
	for _, e := range []*protoEngine{u0, u1} {
		resp, err := reg.Heartbeat(ctx, e.heartbeat())
		if err != nil || !resp.GetRegistered() || resp.GetTtlMs() != 6000 {
			t.Fatalf("%s: %v %v", e.id, resp, err)
		}
	}
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		tick := time.NewTicker(time.Second)
		defer tick.Stop()
		for {
			select {
			case <-stop:
				return
			case <-tick.C:
				for _, e := range []*protoEngine{u0, u1} {
					e.mu.Lock()
					beating := e.beating
					e.mu.Unlock()
					if beating {
						_, _ = reg.Heartbeat(ctx, e.heartbeat())
					}
				}
			}
		}
	}()
	defer func() {
		for _, e := range []*protoEngine{u0, u1} {
			e.set(func(e *protoEngine) { e.beating = false })
			if _, err := reg.Deregister(ctx, &mwipv1.DeregisterRequest{InstanceId: e.id}); err != nil {
				t.Errorf("deregister %s: %v", e.id, err)
			}
		}
	}()
	ws := workers(t, gateway)
	if ws["e2e-u0"].Target != u0.target || ws["e2e-u1"].Target != u1.target || ws["e2e-u0"].Model != "external-engine" || ws["e2e-u1"].Role != "unified" {
		t.Fatalf("registered instances %+v %+v", ws["e2e-u0"], ws["e2e-u1"])
	}

	// The engine receives the intermediate state as the tokenizer rendered it.
	weather := []any{map[string]any{"type": "function", "function": map[string]any{"name": "get_weather", "description": "Weather of a city",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []any{"city"}}}}}
	body := chatBody("external-engine", "What is the weather in Paris?", map[string]any{"tools": weather, "temperature": 0.4, "top_p": 0.9, "max_tokens": 100})
	_, rr := renderModel(t, "qwen3", "chat_completions", body)
	before := map[string]int{"e2e-u0": u0.count(), "e2e-u1": u1.count()}
	r := post(t, gateway, "/v1/chat/completions", keyA, body, nil)
	if r.status != 200 {
		t.Fatalf("external-engine: %d %s", r.status, r.body)
	}
	ch := message(t, r.json(t))
	m := ch["message"].(map[string]any)
	instance := r.header.Get("X-ModelWeave-Decode-Worker")
	call := m["tool_calls"].([]any)[0].(map[string]any)
	if m["reasoning_content"] != "Thinking on "+instance+"." || m["content"] != "Answer from "+instance+"." || ch["finish_reason"] != "tool_calls" ||
		path(call, "function", "name") != "get_weather" || path(call, "function", "arguments") != `{"city":"Paris"}` || call["id"] != "call_e2e" {
		t.Fatalf("formatted answer from %s: %v", instance, ch)
	}
	usage := r.json(t)["usage"].(map[string]any)
	if usage["prompt_tokens"] != rr["prompt_tokens"] || usage["completion_tokens"] != float64(7) || path(usage, "completion_tokens_details", "reasoning_tokens") != float64(3) {
		t.Fatalf("usage %v (tokenizer prompt_tokens %v)", usage, rr["prompt_tokens"])
	}
	got := engines[instance].since(before[instance])
	if len(got) != 1 {
		t.Fatalf("engine calls on %s: %d", instance, len(got))
	}
	req := got[0]
	io, po, conv := req.GetInferenceOptions(), req.GetParsingOptions(), req.GetConversation()
	if fmt.Sprint(req.GetTokenIds()) != fmt.Sprint(ints(rr["token_ids"])) ||
		req.GetDisagg().GetMode() != mwipv1.DisaggMode_DISAGG_MODE_NORMAL || req.GetPriority() != mwipv1.Priority_PRIORITY_HIGH || req.GetTimeoutMs() <= 0 {
		t.Fatalf("request to %s: %d tokens (tokenizer %v), disagg %v, priority %v", instance, len(req.GetTokenIds()), rr["prompt_tokens"], req.GetDisagg(), req.GetPriority())
	}
	if io.GetTemperature() != 0.4 || io.GetTopP() != 0.9 || io.GetMaxTokens() != 100 || fmt.Sprint(io.GetStopTokenIds()) != fmt.Sprint([]uint32{qwenIMEnd, qwenEOT}) ||
		po.GetReasoningParser() != "qwen3" || po.GetToolCallParser() != "hermes" || !po.GetParseToolCalls() ||
		po.GetThinkingInitialStage() != mwipv1.ThinkingStage_THINKING_STAGE_REASONING_START {
		t.Fatalf("options %v %v", io, po)
	}
	if len(conv.GetTools()) != 1 || conv.GetTools()[0].GetFunction().GetName() != "get_weather" ||
		conv.GetTools()[0].GetFunction().GetParameters().GetFields()["required"] == nil || !conv.GetChatTemplateKwargs().GetFields()["enable_thinking"].GetBoolValue() {
		t.Fatalf("conversation %v", conv)
	}

	// Streaming clients get the same output as SSE; round robin reaches the
	// other instance.
	h, events := stream(t, gateway, "/v1/chat/completions", keyA, map[string]any{"model": "external-engine", "stream": true,
		"messages": []any{map[string]any{"role": "user", "content": "Weather?"}}, "tools": weather}, nil)
	joined := ""
	for _, ev := range events {
		joined += ev.data
	}
	other := h.Get("X-ModelWeave-Decode-Worker")
	if other == instance || engines[other] == nil || !strings.Contains(joined, `"reasoning_content":"Thinking on `+other+`."`) ||
		!strings.Contains(joined, `"finish_reason":"tool_calls"`) || !strings.HasSuffix(joined, "[DONE]") {
		t.Fatalf("stream from %s (first %s): %s", other, instance, joined)
	}

	// served sends n requests and counts the instances that answered them.
	served := func(what string, n int) map[string]int {
		out := map[string]int{}
		for i := 0; i < n; i++ {
			r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("external-engine", fmt.Sprintf("%s %d %s", what, i, nonce()), nil), nil)
			if r.status != 200 {
				t.Fatalf("%s %d: %d %s", what, i, r.status, r.body)
			}
			out[r.header.Get("X-ModelWeave-Decode-Worker")]++
		}
		return out
	}

	// A full instance answers capacity_exhausted: the gateway retries on the
	// other instance, and every client still gets its answer.
	u1.set(func(e *protoEngine) { e.full = true })
	n1 := u1.count()
	if got := served("Retry", 4); got["e2e-u0"] != 4 || u1.count() == n1 {
		t.Fatalf("answers with e2e-u1 full: %v (e2e-u1 tried %d times)", got, u1.count()-n1)
	}
	u1.set(func(e *protoEngine) { e.full = false })

	// A draining instance takes no new requests.
	u1.set(func(e *protoEngine) { e.draining = true })
	if _, err := reg.Heartbeat(ctx, u1.heartbeat()); err != nil {
		t.Fatal(err)
	}
	n1 = u1.count()
	if got := served("Drain", 3); got["e2e-u0"] != 3 || u1.count() != n1 {
		t.Fatalf("answers while e2e-u1 drains: %v", got)
	}
	u1.set(func(e *protoEngine) { e.draining = false })
	if _, err := reg.Heartbeat(ctx, u1.heartbeat()); err != nil {
		t.Fatal(err)
	}

	// e2e-u1 shuts down: deregistered at once, e2e-u0 serves alone. Its
	// next heartbeat registers it again.
	u1.set(func(e *protoEngine) { e.beating = false })
	removed0 := metric(t, gateway, "modelweave_instance_removals_total", `model="external-engine"`, `reason="deregistered"`)
	dr, err := reg.Deregister(ctx, &mwipv1.DeregisterRequest{InstanceId: "e2e-u1"})
	if err != nil || !dr.GetRemoved() {
		t.Fatalf("deregister: %v %v", dr, err)
	}
	if _, ok := workers(t, gateway)["e2e-u1"]; ok {
		t.Fatal("e2e-u1 still registered after deregistering")
	}
	if dr, err := reg.Deregister(ctx, &mwipv1.DeregisterRequest{InstanceId: "e2e-u1"}); err != nil || dr.GetRemoved() {
		t.Fatalf("deregister twice: %v %v", dr, err)
	}
	if d := metric(t, gateway, "modelweave_instance_removals_total", `model="external-engine"`, `reason="deregistered"`) - removed0; d != 1 {
		t.Fatalf("deregistrations counted: %v", d)
	}
	if got := served("Alone", 3); got["e2e-u0"] != 3 {
		t.Fatalf("answers without e2e-u1: %v", got)
	}
	resp, err := reg.Heartbeat(ctx, u1.heartbeat())
	if err != nil || !resp.GetRegistered() {
		t.Fatalf("registering again: %v %v", resp, err)
	}
	u1.set(func(e *protoEngine) { e.beating = true })
	if got := served("Back", 4); got["e2e-u1"] == 0 {
		t.Fatalf("answers after e2e-u1 came back: %v", got)
	}
}
