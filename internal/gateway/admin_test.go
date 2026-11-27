package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
	"github.com/leonyang215/ModelWeave/internal/config"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

func adminRequest(s *Server, method, path, body, token string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	r.Header.Set("X-ModelWeave-Token", token)
	if body != "" {
		r.Header.Set("Content-Type", "application/json")
	}
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, r)
	return w
}

func decodeAdmin(t *testing.T, w *httptest.ResponseRecorder, out any) {
	t.Helper()
	if w.Code != http.StatusOK {
		t.Fatalf("admin response: %d %s", w.Code, w.Body)
	}
	if err := json.Unmarshal(w.Body.Bytes(), out); err != nil {
		t.Fatal(err)
	}
}

func TestAdminTokenIsolationAndEmbeddedConsole(t *testing.T) {
	s := newTestServer(t, `
server: {admin_token: operator-secret-913, control_token: engine-secret-857}
tenants: [{name: tenant, api_keys: [sk-tenant-secret-246]}]
`)
	t.Cleanup(s.Close)
	for _, op := range []struct{ method, path, body string }{
		{"GET", "/admin/models", ""}, {"GET", "/admin/models/m-alias", ""},
		{"GET", "/admin/workers", ""}, {"GET", "/admin/workers/missing", ""},
		{"GET", "/admin/stats", ""}, {"GET", "/admin/blacklist", ""},
		{"PUT", "/admin/blacklist/offline", `{"reason":"quarantine"}`},
		{"DELETE", "/admin/blacklist/offline", ""},
		{"POST", "/admin/workers/missing/drain", ""}, {"POST", "/admin/workers/missing/resume", ""},
	} {
		for _, token := range []string{"", "wrong", "engine-secret-857", "sk-tenant-secret-246"} {
			w := adminRequest(s, op.method, op.path, op.body, token)
			if w.Code != http.StatusUnauthorized {
				t.Fatalf("admin auth %s %s: %d %s", op.method, op.path, w.Code, w.Body)
			}
			if w.Header().Get("Cache-Control") != "no-store" {
				t.Fatal("admin error may be cached")
			}
		}
	}
	for _, path := range []string{"/admin/models", "/admin/stats", "/admin/workers", "/admin/blacklist"} {
		w := adminRequest(s, "GET", path, "", "operator-secret-913")
		if w.Code != 200 {
			t.Fatalf("operator denied: %d %s", w.Code, w.Body)
		}
		for _, secret := range []string{"operator-secret-913", "engine-secret-857", "sk-tenant-secret-246"} {
			if strings.Contains(w.Body.String(), secret) {
				t.Fatalf("%s leaked a credential", path)
			}
		}
	}
	for path, contentType := range map[string]string{
		"/admin": "text/html", "/admin/": "text/html",
		"/admin/assets/app.mjs": "text/javascript", "/admin/assets/utils.mjs": "text/javascript",
		"/admin/assets/styles.css": "text/css", "/admin/assets/favicon.svg": "image/svg+xml",
	} {
		w := request(s, "GET", path, "")
		if w.Code != 200 || !strings.HasPrefix(w.Header().Get("Content-Type"), contentType) {
			t.Fatalf("embedded asset %s: %d %s", path, w.Code, w.Header())
		}
		if !strings.Contains(w.Header().Get("Content-Security-Policy"), "frame-ancestors 'none'") || w.Header().Get("X-Content-Type-Options") != "nosniff" {
			t.Fatalf("missing browser protection: %s", w.Header())
		}
		if strings.Contains(w.Body.String(), "operator-secret-913") {
			t.Fatal("public shell contains a secret")
		}
	}
	for _, path := range []string{"/admin/unknown", "/admin/assets/control.go", "/admin/assets/%2e%2e", "/admin/assets/web%2findex.html"} {
		if w := request(s, "GET", path, ""); w.Code != 404 {
			t.Fatalf("unknown asset %s: %d", path, w.Code)
		}
	}
	rc := mwipv1.NewRegistryClient(registryClient(t, s))
	hb := instance("w0", "m", mwipv1.Role_ROLE_UNIFIED, "host:9001")
	adminCtx := metadata.AppendToOutgoingContext(context.Background(), TokenMetadata, "operator-secret-913")
	if _, err := rc.Heartbeat(adminCtx, hb); status.Code(err) != codes.Unauthenticated {
		t.Fatalf("admin token accepted by Registry: %v", err)
	}
	engineCtx := metadata.AppendToOutgoingContext(context.Background(), TokenMetadata, "engine-secret-857")
	if _, err := rc.Heartbeat(engineCtx, hb); err != nil {
		t.Fatalf("engine token rejected: %v", err)
	}
}

func TestAdminTokenFallbackAndUIDisable(t *testing.T) {
	s := newTestServer(t, "server: {control_token: legacy-token, disable_admin_ui: true}\n")
	t.Cleanup(s.Close)
	if w := adminRequest(s, "GET", "/admin/models", "", "legacy-token"); w.Code != 200 {
		t.Fatalf("legacy control token no longer works: %d %s", w.Code, w.Body)
	}
	for _, path := range []string{"/admin", "/admin/", "/admin/assets/app.mjs"} {
		if w := request(s, "GET", path, ""); w.Code != 404 {
			t.Fatalf("disabled UI still served: %s %d", path, w.Code)
		}
	}
}

func TestAdminRejectsCrossOriginMutations(t *testing.T) {
	s := newTestServer(t, "") // Protect even tokenless local installations.
	t.Cleanup(s.Close)
	for _, headers := range []map[string]string{
		{"Origin": "https://evil.example"}, {"Sec-Fetch-Site": "cross-site"},
		{"Sec-Fetch-Site": "same-site", "Origin": "https://sibling.example"},
	} {
		r := httptest.NewRequest("PUT", "http://gateway.example/admin/blacklist/w0", strings.NewReader(`{"reason":"csrf"}`))
		r.Header.Set("Content-Type", "application/json")
		for k, v := range headers {
			r.Header.Set(k, v)
		}
		w := httptest.NewRecorder()
		s.Handler().ServeHTTP(w, r)
		if w.Code != http.StatusForbidden {
			t.Fatalf("cross-origin mutation accepted: %d %s", w.Code, w.Body)
		}
	}
	if b, err := s.reg.Store().Blocks(context.Background()); err != nil || len(b) != 0 {
		t.Fatalf("CSRF changed the blacklist: %+v %v", b, err)
	}
	r := httptest.NewRequest("PUT", "http://gateway.example/admin/blacklist/w0", strings.NewReader(`{"reason":"same origin"}`))
	r.Header.Set("Origin", "http://gateway.example")
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, r)
	if w.Code != 200 {
		t.Fatalf("same-origin request rejected: %d %s", w.Code, w.Body)
	}
}

func TestAdminModelAndWorkerViews(t *testing.T) {
	s := newTestServer(t, `
discovery:
  workers:
    - {id: active, model: m, role: unified, target: 'a:9001', labels: {zone: test}, kv_capacity_tokens: 2048}
    - {id: blocked, model: m, role: unified, target: 'b:9001'}
    - {id: draining, model: m, role: prefill, target: 'p:9001'}
    - {id: unhealthy, model: m, role: decode, target: 'd:9001'}
`)
	t.Cleanup(s.Close)
	s.reg.Get("active").UpdateState(registry.EngineState{RunningRequests: 7}, time.Now().Add(-time.Hour))
	s.reg.Store().Block(context.Background(), "blocked", "bad output", time.Now())
	s.reg.Get("draining").SetDraining(true)
	s.reg.Get("unhealthy").ReportFailure(time.Now(), 1, time.Minute)
	var list struct {
		Models []adminModelView `json:"models"`
	}
	decodeAdmin(t, request(s, "GET", "/admin/models", ""), &list)
	if len(list.Models) != 1 {
		t.Fatalf("models: %+v", list)
	}
	m := list.Models[0]
	if m.Name != "m" || len(m.Aliases) != 1 || m.Instances != (instanceCounts{Total: 4, Available: 1, Blocked: 1, Draining: 1, Unhealthy: 1}) || m.Roles[registry.RoleUnified].Total != 2 {
		t.Fatalf("model aggregates: %+v", m)
	}
	if m.Policies[registry.RoleDecode].Name != "weighted" || len(m.Policies[registry.RoleDecode].Scorers) != 2 || m.QueueCapacity != 512 || m.MaxWaitMS != 30000 {
		t.Fatalf("model config: %+v", m)
	}
	decodeAdmin(t, request(s, "GET", "/admin/models/m-alias", ""), &m)
	if m.Name != "m" {
		t.Fatal("admin model alias did not resolve")
	}
	var v workerView
	decodeAdmin(t, request(s, "GET", "/admin/workers/active", ""), &v)
	if v.State != nil || v.StateStatus != "stale" || v.StateAgeMS < 3600000 || v.KVCapacity != 2048 || v.Labels["zone"] != "test" {
		t.Fatalf("stale report or static metadata: %+v", v)
	}
	for _, tc := range []struct{ path, id string }{
		{"?model=m-alias&role=unified&status=active", "active"},
		{"?status=blocked", "blocked"}, {"?status=draining", "draining"}, {"?status=unhealthy", "unhealthy"},
	} {
		var out struct{ Workers []workerView }
		decodeAdmin(t, request(s, "GET", "/admin/workers"+tc.path, ""), &out)
		if len(out.Workers) != 1 || out.Workers[0].ID != tc.id {
			t.Fatalf("filter %s: %+v", tc.path, out.Workers)
		}
	}
	for _, tc := range []struct {
		path string
		code int
	}{
		{"/admin/models/unknown", 404}, {"/admin/workers/unknown", 404},
		{"/admin/workers?model=unknown", 404}, {"/admin/workers?role=bogus", 400},
		{"/admin/workers?status=bogus", 400},
	} {
		if w := request(s, "GET", tc.path, ""); w.Code != tc.code {
			t.Fatalf("%s: %d %s", tc.path, w.Code, w.Body)
		}
	}
	metrics := request(s, "GET", "/metrics", "").Body.String()
	for _, line := range []string{
		`modelweave_worker_blocked{model="m",role="unified",worker="blocked"} 1`,
		`modelweave_worker_up{model="m",role="unified",worker="blocked"} 0`,
	} {
		if !strings.Contains(metrics, line) {
			t.Fatalf("metrics missing %s", line)
		}
	}
}

func TestAdminBlacklistInputValidationAndEmptyLists(t *testing.T) {
	s := newTestServer(t, "")
	t.Cleanup(s.Close)
	for _, path := range []string{"/admin/workers", "/admin/blacklist"} {
		w := request(s, "GET", path, "")
		if w.Code != 200 || strings.Contains(w.Body.String(), "null") || !strings.Contains(w.Body.String(), "[]") {
			t.Fatalf("empty collection: %d %s", w.Code, w.Body)
		}
	}
	for _, body := range []string{
		`{}`, `null`, `[]`, `{"reason":null}`, `{"reason":"  "}`, `{"reason":17}`,
		`{"reason":"ok","other":true}`, `{"reason":"ok"} {}`, `{"reason":"ok"} garbage`,
		`{"reason":"` + strings.Repeat("字", 513) + `"}`,
	} {
		if w := adminRequest(s, "PUT", "/admin/blacklist/offline", body, ""); w.Code != 400 {
			t.Fatalf("accepted invalid body: %d %s", w.Code, w.Body)
		}
	}
	for _, body := range []string{`{"reason":"` + strings.Repeat("x", 9000) + `"}`, `{"reason":"ok"}` + strings.Repeat(" ", 9000)} {
		if w := adminRequest(s, "PUT", "/admin/blacklist/offline", body, ""); w.Code != 413 {
			t.Fatalf("oversized body: %d %s", w.Code, w.Body)
		}
	}
	if w := request(s, "PUT", "/admin/blacklist/offline", `{"reason":"ok"}`); w.Code != 415 {
		t.Fatalf("non-JSON content type: %d", w.Code)
	}
	for _, id := range []string{"%20", strings.Repeat("x", 257)} {
		if w := adminRequest(s, "PUT", "/admin/blacklist/"+id, `{"reason":"ok"}`, ""); w.Code != 400 {
			t.Fatalf("invalid ID: %d %s", w.Code, w.Body)
		}
	}
	id := "worker/space ?#中文"
	path := "/admin/blacklist/" + url.PathEscape(id)
	var b registry.Block
	decodeAdmin(t, adminRequest(s, "PUT", path, `{"reason":"  运维隔离  "}`, ""), &b)
	if b.ID != id || b.Reason != "运维隔离" {
		t.Fatalf("escaped ID/reason: %+v", b)
	}
	decodeAdmin(t, adminRequest(s, "DELETE", path, "", ""), &map[string]any{})
	decodeAdmin(t, adminRequest(s, "DELETE", path, "", ""), &map[string]any{})
}

func TestAdminOperationsChangeRealInferenceRouting(t *testing.T) {
	_, target := mockWorker(t)
	cfg, err := config.Parse([]byte(fmt.Sprintf(`
tokenizer: {endpoints: [%q]}
discovery: {workers: [{id: w0, model: m, role: unified, target: %q}]}
models: [{name: m, mode: normal}]
`, fakeTokenizer(t), target)))
	if err != nil {
		t.Fatal(err)
	}
	var audit bytes.Buffer
	s, err := New(cfg, slog.New(slog.NewJSONHandler(&audit, nil)), "admin-test")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	infer := func(want int) {
		t.Helper()
		w := request(s, "POST", "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
		if w.Code != want {
			t.Fatalf("inference: %d %s, want %d", w.Code, w.Body, want)
		}
	}
	infer(200)
	held := s.reg.Get("w0").Reserve(100, 0, nil)
	decodeAdmin(t, adminRequest(s, "PUT", "/admin/blacklist/w0", `{"reason":"quarantine"}`, ""), &registry.Block{})
	if snap := s.reg.Get("w0").Snapshot(time.Now(), 0); snap.Inflight != 1 || !snap.Blocked {
		t.Fatalf("block aborted in-flight work: %+v", snap)
	}
	infer(503)
	if err := held.Release(true); err != nil {
		t.Fatal(err)
	}
	decodeAdmin(t, request(s, "POST", "/admin/workers/w0/drain", ""), &map[string]any{})
	decodeAdmin(t, request(s, "DELETE", "/admin/blacklist/w0", ""), &map[string]any{})
	infer(503) // Unblocking must not undo the independent drain.
	decodeAdmin(t, request(s, "POST", "/admin/workers/w0/resume", ""), &map[string]any{})
	infer(200)
	for _, action := range []string{`"action":"block"`, `"action":"unblock"`, `"action":"set_drain"`, `"remote_addr"`} {
		if !strings.Contains(audit.String(), action) {
			t.Fatalf("missing audit field %s: %s", action, &audit)
		}
	}
}

func TestRedisAdminBlacklistPersistsAcrossGatewayRestart(t *testing.T) {
	a, b, mini := gatewayPair(t)
	registerGatewayWorker(t, a)
	decodeAdmin(t, adminRequest(a, "PUT", "/admin/blacklist/w0", `{"reason":"shared isolation"}`, ""), &registry.Block{})
	if ws := workerViews(t, b); len(ws) != 1 || !ws[0].Blocked || ws[0].Block.Reason != "shared isolation" {
		t.Fatalf("block not shared: %+v", ws)
	}
	spec := a.reg.Get("w0").Spec()
	a.reg.Remove("w0")
	if ws := workerViews(t, b); len(ws) != 0 {
		t.Fatal("removed worker still visible")
	}
	var blocks struct{ Blacklist []registry.Block }
	decodeAdmin(t, request(b, "GET", "/admin/blacklist", ""), &blocks)
	if len(blocks.Blacklist) != 1 {
		t.Fatalf("offline rule lost: %+v", blocks)
	}
	a.Close()
	b.Close()
	restarted, err := New(a.cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), "restarted")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(restarted.Close)
	if _, _, err := restarted.reg.Upsert(spec, registry.SourceHeartbeat); err != nil {
		t.Fatal(err)
	}
	if ws := workerViews(t, restarted); len(ws) != 1 || !ws[0].Blocked {
		t.Fatalf("restart/re-registration evaded rule: %+v", ws)
	}
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
	if w := request(restarted, "POST", "/v1/chat/completions", body); w.Code != 503 {
		t.Fatalf("restarted gateway routed to blocked instance: %d %s", w.Code, w.Body)
	}
	if mini != nil {
		mini.SetError("ERR injected outage")
		for _, op := range []struct{ method, path, body string }{
			{"GET", "/admin/models", ""}, {"GET", "/admin/blacklist", ""},
			{"PUT", "/admin/blacklist/new", `{"reason":"failure"}`}, {"DELETE", "/admin/blacklist/w0", ""},
		} {
			if w := adminRequest(restarted, op.method, op.path, op.body, ""); w.Code != 503 {
				t.Fatalf("outage %s: %d %s", op.path, w.Code, w.Body)
			}
		}
		mini.SetError("")
		decodeAdmin(t, request(restarted, "GET", "/admin/blacklist", ""), &blocks)
		if len(blocks.Blacklist) != 1 {
			t.Fatal("failed mutations changed rules")
		}
	}
	decodeAdmin(t, request(restarted, "DELETE", "/admin/blacklist/w0", ""), &map[string]any{})
	if w := request(restarted, "POST", "/v1/chat/completions", body); w.Code != 200 {
		t.Fatalf("unblocked gateway did not recover routing: %d %s", w.Code, w.Body)
	}
}
