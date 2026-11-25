package gateway

import (
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leonyang215/ModelWeave/internal/config"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/qos"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
)

func newTestServer(t *testing.T, extra string) *Server {
	t.Helper()
	cfg, err := config.Parse([]byte("tokenizer: {endpoints: [\"http://127.0.0.1:1\"]}\nmodels: [{name: m, aliases: [m-alias]}]\n" + extra))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), "test")
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestErrorShapes(t *testing.T) {
	for _, tc := range []struct {
		err    error
		status int
		kind   string
	}{
		{&qos.Rejection{Reason: "rpm"}, 429, "rate_limit_exceeded"},
		{qos.ErrQueueFull, 503, "overloaded"},
		{scheduler.ErrNoWorkers, 503, "no_worker_available"},
		{errors.New("boom"), 500, "internal_error"},
	} {
		e := classify(tc.err)
		if e.Status != tc.status || e.Kind != tc.kind {
			t.Fatalf("%v: %+v", tc.err, e)
		}
		rec := httptest.NewRecorder()
		writeError(rec, mwip.ProtocolMessages, e)
		var body map[string]any
		_ = json.Unmarshal(rec.Body.Bytes(), &body)
		if rec.Code != tc.status || body["type"] != "error" {
			t.Fatalf("anthropic shape: %d %s", rec.Code, rec.Body)
		}
		if (tc.status == 429 || tc.status == 503) && rec.Header().Get("Retry-After") == "" {
			t.Fatalf("missing Retry-After for %d", tc.status)
		}
		rec = httptest.NewRecorder()
		writeError(rec, mwip.ProtocolChatCompletions, e)
		_ = json.Unmarshal(rec.Body.Bytes(), &body)
		if body["error"].(map[string]any)["type"] != tc.kind {
			t.Fatalf("openai shape: %s", rec.Body)
		}
	}
}

func TestHeartbeatRegistersWorkerAndControlToken(t *testing.T) {
	s := newTestServer(t, "server: {control_token: secret}\n")
	h := s.Handler()
	hb := `{"worker":{"id":"w1","model":"m-alias","role":"decode","url":"http://w1:9000","normal":true},
	        "state":{"running_requests":2,"kv_capacity_tokens":1000,"kv_used_tokens":100}}`
	req := httptest.NewRequest(http.MethodPost, "/mwip/v1/heartbeat", strings.NewReader(hb))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("heartbeat without token: %d", rec.Code)
	}
	req = httptest.NewRequest(http.MethodPost, "/mwip/v1/heartbeat", strings.NewReader(hb))
	req.Header.Set("X-ModelWeave-Token", "secret")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("heartbeat: %d %s", rec.Code, rec.Body)
	}
	w := s.reg.Get("w1")
	if w == nil || w.Spec().Model != "m" || !w.Spec().ServesNormal() {
		t.Fatalf("worker not registered under the canonical model: %+v", w)
	}
	req = httptest.NewRequest(http.MethodPost, "/mwip/v1/heartbeat", strings.NewReader(`{"worker":{"id":"x","model":"nope","role":"decode","url":"u"}}`))
	req.Header.Set("X-ModelWeave-Token", "secret")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("unknown model accepted: %d", rec.Code)
	}
	// The API itself does not need the control token.
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "m-alias") {
		t.Fatalf("models: %d %s", rec.Code, rec.Body)
	}
}
