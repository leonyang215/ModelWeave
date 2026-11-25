package tokenizer

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// Integration test against a running tokenizer:
//
//	MW_TEST_TOKENIZER_URL=http://127.0.0.1:38090 go test ./internal/tokenizer
func tokenizerURL(t *testing.T) string {
	u := os.Getenv("MW_TEST_TOKENIZER_URL")
	if u == "" {
		t.Skip("MW_TEST_TOKENIZER_URL not set")
	}
	return u
}

func TestRenderProducesIntermediateState(t *testing.T) {
	c := New([]string{tokenizerURL(t)}, 10*time.Second)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	body := `{"model":"deepseek-v4.1","stream":true,"thinking":{"type":"disabled"},"stop":["END"],
	          "messages":[{"role":"user","content":"Say hi"}]}`
	rr, err := c.Render(ctx, &mwip.RenderRequest{
		RequestID: "it-1", Protocol: mwip.ProtocolChatCompletions, Model: "deepseek-v4.1", Body: json.RawMessage(body),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(rr.TokenIDs) == 0 || rr.TokenIDs[0] != 0 || rr.PromptTokens != len(rr.TokenIDs) || !rr.Stream || rr.Multimodal != nil {
		t.Fatalf("unexpected token list: %+v", rr)
	}
	if rr.InferenceOptions.StopTokenIDs[0] != 1 || rr.ParsingOptions.StopSequences[0] != "END" || rr.Conversation.ThinkingMode {
		t.Fatalf("unexpected options: %+v %+v %+v", rr.InferenceOptions, rr.ParsingOptions, rr.Conversation)
	}
	if rr.Protocol != mwip.ProtocolChatCompletions || rr.RenderArtifact.Version != 2 {
		t.Fatalf("unexpected protocol/artifact: %s %+v", rr.Protocol, rr.RenderArtifact)
	}
}

func TestRenderErrorIsProtocolShaped(t *testing.T) {
	c := New([]string{tokenizerURL(t)}, 10*time.Second)
	_, err := c.Render(context.Background(), &mwip.RenderRequest{
		RequestID: "it-2", Protocol: mwip.ProtocolMessages, Model: "deepseek-v4.1",
		Body: json.RawMessage(`{"model":"x","messages":"bad"}`),
	})
	te, ok := err.(*Error)
	if !ok || te.Status != 400 || !strings.Contains(string(te.Body), "invalid_request_error") {
		t.Fatalf("want protocol-shaped 400, got %v", err)
	}
}
