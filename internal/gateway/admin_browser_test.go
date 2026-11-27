package gateway

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http/httptest"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/leonyang215/ModelWeave/internal/config"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

// Optional real-browser contract test: it exercises the embedded assets and
// real HTTP handlers, not a separately mocked frontend API. No GPU/services.
func TestAdminBrowser(t *testing.T) {
	if os.Getenv("MW_ADMIN_BROWSER") != "1" {
		t.Skip("set MW_ADMIN_BROWSER=1 after installing test/admin-ui Playwright")
	}
	const token = "browser-test-operator-token"
	cfg, err := config.Parse([]byte(`
server: {admin_token: browser-test-operator-token, control_token: browser-test-engine-token}
tokenizer: {endpoints: ["http://127.0.0.1:1"]}
engine_state: {stale_after: 10m}
tenants:
  - {name: inference-team, api_keys: [test-not-a-real-api-key], priority: high, weight: 2, rpm: 600, tpm: 2000000, max_concurrency: 128}
  - {name: batch-team, priority: low, rpm: 120, tpm: 500000, max_concurrency: 32}
models:
  - name: deepseek-v4.1
    aliases: [DeepSeek-V4.1-Flash, deepseek-flash]
    context_length: 131072
    normal: {enabled: true}
    routing:
      default:
        scorers: [{name: prefix-cache, weight: 3}, {name: load, weight: 1}]
      encode: {policy: load}
  - name: qwen3
    aliases: [Qwen3-8B]
    mode: normal
    context_length: 32768
    routing:
      default:
        scorers: [{name: load, weight: 4}, {name: affinity, weight: 0.5}]
  - {name: qwen3-vl, aliases: [Qwen3-VL-8B-Instruct], mode: pd}
  - {name: future-model, mode: normal}
`))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), "v0.3.0-browser-test")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	for i, item := range []struct {
		id, model string
		role      registry.Role
	}{
		{"encode-0", "deepseek-v4.1", registry.RoleEncode},
		{"prefill-0", "deepseek-v4.1", registry.RolePrefill},
		{"decode-0", "deepseek-v4.1", registry.RoleDecode},
		{"decode-1", "deepseek-v4.1", registry.RoleDecode},
		{"unified-0", "qwen3", registry.RoleUnified},
		{"unified-1", "qwen3", registry.RoleUnified},
		{"qvl-encode-0", "qwen3-vl", registry.RoleEncode},
		{"qvl-prefill-0", "qwen3-vl", registry.RolePrefill},
		{"qvl-decode-0", "qwen3-vl", registry.RoleDecode},
	} {
		labels := map[string]string{"zone": "test-west", "runtime": "mwip"}
		if item.id == "decode-0" {
			labels["untrusted"] = `<img src=x onerror="window.__mwXSS=true">`
		}
		w, _, err := s.reg.Upsert(registry.Spec{
			ID: item.id, Model: item.model, Role: item.role, Target: fmt.Sprintf("10.0.1.%d:9001", i+10),
			Normal: true, Labels: labels, KVCapacityTokens: 1048576, MaxRunningRequests: 128,
		}, registry.SourceHeartbeat)
		if err != nil {
			t.Fatal(err)
		}
		if err := w.UpdateState(registry.EngineState{
			RunningRequests: i * 2, KVUsedTokens: int64(i+1) * 65536, KVCapacityTokens: 1048576,
			PrefillQueueTokens: int64(i) * 128,
		}, time.Now()); err != nil {
			t.Fatal(err)
		}
	}
	for _, id := range []string{"decode-1", "offline-maintenance"} {
		if _, err := s.reg.Store().Block(context.Background(), id, "节点维护，等待恢复确认", time.Now()); err != nil {
			t.Fatal(err)
		}
	}
	s.reg.Get("unified-1").SetDraining(true)
	s.reg.Get("qvl-encode-0").ReportFailure(time.Now(), 1, time.Hour)
	s.updateOverload()
	server := httptest.NewServer(s.Handler())
	t.Cleanup(server.Close)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "node", "--test", "../../test/admin-ui/browser.test.mjs")
	cmd.Env = append(os.Environ(), "MW_ADMIN_URL="+server.URL, "MW_ADMIN_TOKEN="+token)
	output, err := cmd.CombinedOutput()
	t.Logf("browser tests:\n%s", output)
	if err != nil {
		t.Fatalf("browser tests: %v", err)
	}
}
