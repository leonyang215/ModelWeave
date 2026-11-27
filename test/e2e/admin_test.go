//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
	"google.golang.org/grpc/metadata"
)

func adminCall(t *testing.T, method, path string, body any) []byte {
	t.Helper()
	var content []byte
	if body != nil {
		content, _ = json.Marshal(body)
	}
	req, _ := http.NewRequest(method, gateway+path, bytes.NewReader(content))
	req.Header.Set("X-ModelWeave-Token", ctlToken)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	response, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	data, _ := io.ReadAll(response.Body)
	if response.StatusCode != 200 {
		t.Fatalf("admin %s %s: %d %s", method, path, response.StatusCode, data)
	}
	return data
}

func TestAdminConsoleInventoryAndPluginDiscovery(t *testing.T) {
	if body := string(get(t, gateway+"/admin/")); !strings.Contains(body, "/admin/assets/app.mjs") {
		t.Fatal("embedded admin console missing")
	}
	var catalog struct {
		Scorers  []struct{ Name, Version, Source string }
		Policies []string
	}
	if err := json.Unmarshal(adminCall(t, "GET", "/admin/scheduling/plugins", nil), &catalog); err != nil {
		t.Fatal(err)
	}
	names := map[string]bool{}
	for _, p := range catalog.Scorers {
		if p.Version == "" || p.Source == "" {
			t.Fatalf("incomplete plugin metadata: %+v", p)
		}
		names[p.Name] = true
	}
	for _, name := range []string{"prefix-cache", "load", "affinity", "label"} {
		if !names[name] {
			t.Fatalf("missing plugin %s", name)
		}
	}
	var model struct {
		Name     string
		Policies map[string]any
	}
	if err := json.Unmarshal(adminCall(t, "GET", "/admin/models/deepseek-flash", nil), &model); err != nil {
		t.Fatal(err)
	}
	if model.Name != "deepseek-v4.1" || len(model.Policies) != 4 {
		t.Fatalf("admin model alias/config: %+v", model)
	}
	request, _ := http.NewRequest("GET", gateway+"/admin/scheduling/plugins", nil)
	response, err := client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != 401 {
		t.Fatalf("admin catalog without token: %d", response.StatusCode)
	}
}

func TestWeightedPluginsDrainAndPersistentBlacklist(t *testing.T) {
	response, err := client.Get(gateway + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(response.Body)
	response.Body.Close()
	if !bytes.Contains(data, []byte(`"e2e-weighted"`)) {
		t.Skip("weighted fixture enabled by make test-e2e / test-e2e-redis")
	}
	reg, ok := registryClient(t)
	if !ok {
		t.Skip("MW_E2E_REGISTRY required")
	}
	target := gateway
	if replica := os.Getenv("MW_E2E_GATEWAY_REPLICA"); replica != "" {
		target = replica
		t.Log("requests use a second gateway; engines heartbeat only to the first")
	}
	preferred := startProtoEngine(t, "weighted-preferred-"+nonce())
	fallback := startProtoEngine(t, "weighted-fallback-"+nonce())
	preferred.model, fallback.model = "e2e-weighted", "e2e-weighted"
	preferred.labels, fallback.labels = map[string]string{"pool": "preferred"}, map[string]string{"pool": "fallback"}
	engines := []*protoEngine{preferred, fallback}
	ctx := metadata.AppendToOutgoingContext(context.Background(), "x-modelweave-token", ctlToken)
	var beatMu sync.Mutex
	beat := func() {
		beatMu.Lock()
		defer beatMu.Unlock()
		for _, engine := range engines {
			engine.mu.Lock()
			beating := engine.beating
			engine.mu.Unlock()
			if beating {
				_, _ = reg.Heartbeat(ctx, engine.heartbeat())
			}
		}
	}
	for _, engine := range engines {
		if _, err := reg.Heartbeat(ctx, engine.heartbeat()); err != nil {
			t.Fatal(err)
		}
	}
	stop, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				beat()
			}
		}
	}()
	defer func() {
		close(stop)
		<-done
		adminCall(t, "DELETE", "/admin/blacklist/"+preferred.id, nil)
		for _, engine := range engines {
			_, _ = reg.Deregister(ctx, &mwipv1.DeregisterRequest{InstanceId: engine.id})
		}
	}()
	infer := func(want string) {
		t.Helper()
		r := post(t, target, "/v1/chat/completions", keyA, chatBody("e2e-weighted", "weighted routing "+nonce(), map[string]any{"max_tokens": 32}), nil)
		if r.status != 200 || r.header.Get("X-ModelWeave-Decode-Worker") != want {
			t.Fatalf("weighted inference: %d worker=%q want=%q: %s", r.status, r.header.Get("X-ModelWeave-Decode-Worker"), want, r.body)
		}
	}
	infer(preferred.id)
	adminCall(t, "POST", "/admin/workers/"+preferred.id+"/drain", nil)
	infer(fallback.id)
	adminCall(t, "POST", "/admin/workers/"+preferred.id+"/resume", nil)
	infer(preferred.id)
	adminCall(t, "PUT", "/admin/blacklist/"+preferred.id, map[string]string{"reason": "e2e isolation"})
	infer(fallback.id)
	beatMu.Lock()
	preferred.set(func(e *protoEngine) { e.beating = false })
	_, err = reg.Deregister(ctx, &mwipv1.DeregisterRequest{InstanceId: preferred.id})
	beatMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(adminCall(t, "GET", "/admin/blacklist", nil), []byte(preferred.id)) {
		t.Fatal("offline block disappeared")
	}
	if _, err := reg.Heartbeat(ctx, preferred.heartbeat()); err != nil {
		t.Fatal(err)
	}
	preferred.set(func(e *protoEngine) { e.beating = true })
	infer(fallback.id)
	adminCall(t, "DELETE", "/admin/blacklist/"+preferred.id, nil)
	infer(preferred.id)
	preferred.set(func(e *protoEngine) { e.full = true })
	infer(fallback.id)
	preferred.set(func(e *protoEngine) { e.full = false })
	infer(preferred.id)
}
