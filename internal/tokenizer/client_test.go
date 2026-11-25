package tokenizer

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// h2c starts a test server speaking HTTP/1.1 and h2c, like the tokenizer.
func h2c(h http.Handler) *httptest.Server {
	srv := httptest.NewUnstartedServer(h)
	srv.Config.Protocols = new(http.Protocols)
	srv.Config.Protocols.SetHTTP1(true)
	srv.Config.Protocols.SetUnencryptedHTTP2(true)
	srv.Start()
	return srv
}

// A replica that cannot be reached is skipped: the render succeeds on the
// next one, and the dead replica is not tried again during its cooldown.
// Rejections are answers, not failures.
func TestRenderFailsOverToAnotherReplica(t *testing.T) {
	var hits atomic.Int64
	ok := h2c(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		var req mwip.RenderRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model == "bad" {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"no"}}`))
			return
		}
		_ = json.NewEncoder(w).Encode(mwip.RenderResult{RequestID: req.RequestID, PromptTokens: 3, TokenIDs: []uint32{1, 2, 3}})
	}))
	defer ok.Close()
	dead := httptest.NewServer(http.NotFoundHandler())
	deadURL := dead.URL
	dead.Close()

	c := New([]string{deadURL, ok.URL}, 5*time.Second)
	now := time.Unix(1000, 0)
	c.now = func() time.Time { return now }
	var unreachable []string
	c.OnUnreachable = func(endpoint string, _ error) { unreachable = append(unreachable, endpoint) }
	for i := 0; i < 4; i++ {
		rr, err := c.Render(context.Background(), &mwip.RenderRequest{RequestID: "r", Protocol: "chat_completions", Body: json.RawMessage(`{}`)})
		if err != nil || rr.PromptTokens != 3 {
			t.Fatalf("render %d: %v %+v", i, err, rr)
		}
	}
	if hits.Load() != 4 {
		t.Fatalf("healthy replica served %d renders", hits.Load())
	}
	if len(unreachable) != 1 || unreachable[0] != deadURL {
		t.Fatalf("unreachable replicas reported: %v", unreachable)
	}
	if order := c.order(); order[0] != ok.URL {
		t.Fatalf("dead replica not in cooldown: %v", order)
	}
	now = now.Add(10 * time.Second)
	if order := c.order(); len(order) != 2 {
		t.Fatalf("order %v", order)
	}
	_, err := c.Render(context.Background(), &mwip.RenderRequest{RequestID: "r", Model: "bad", Body: json.RawMessage(`{}`)})
	if te, isErr := err.(*Error); !isErr || te.Status != 400 {
		t.Fatalf("rejection: %v", err)
	}
	// Every replica down: the last transport error.
	c2 := New([]string{deadURL}, time.Second)
	if _, err := c2.Render(context.Background(), &mwip.RenderRequest{Body: json.RawMessage(`{}`)}); err == nil {
		t.Fatal("want an error")
	}
}
