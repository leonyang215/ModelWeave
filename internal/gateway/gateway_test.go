package gateway

import (
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http/httptest"
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
