package gateway

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"

	"github.com/leonyang215/ModelWeave/internal/config"
	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
	scoring "github.com/leonyang215/ModelWeave/pkg/scheduling"
)

func TestSchedulingPluginCatalogIsAuthenticatedAndComplete(t *testing.T) {
	s := newTestServer(t, "server: {admin_token: operator}\n")
	t.Cleanup(s.Close)
	if w := request(s, "GET", "/admin/scheduling/plugins", ""); w.Code != 401 {
		t.Fatalf("unprotected catalog: %d %s", w.Code, w.Body)
	}
	var out struct {
		Scorers       []scoring.Descriptor `json:"scorers"`
		Policies      []string             `json:"policies"`
		Combination   string               `json:"combination"`
		Loading       string               `json:"loading"`
		Configuration string               `json:"configuration"`
	}
	w := adminRequest(s, "GET", "/admin/scheduling/plugins", "", "operator")
	decodeAdmin(t, w, &out)
	plugins := make(map[string]scoring.Descriptor)
	for _, d := range out.Scorers {
		plugins[d.Name] = d
	}
	for _, name := range []string{"prefix-cache", "load", "affinity", "label"} {
		if p := plugins[name]; p.Version == "" || p.Source == "" || p.Description == "" || p.Parameters == nil {
			t.Fatalf("missing plugin metadata: %s %+v", name, p)
		}
	}
	if out.Combination != "weighted" || out.Loading != "compiled-in" || !strings.Contains(out.Configuration, "restart") || len(out.Policies) == 0 || len(plugins["label"].Parameters) != 2 {
		t.Fatalf("catalog contract: %+v", out)
	}
	if w.Header().Get("Cache-Control") != "no-store" {
		t.Fatal("catalog may be cached without authorization")
	}
}

// A real HTTP inference request traverses model resolution, tokenization,
// scheduling and the gRPC mock engine. Three models use different compositions
// and relative weights; the Redis case dispatches through a second gateway.
func TestModelCompositionsChangeRealInferenceRouting(t *testing.T) {
	for _, backend := range []string{"memory", "redis"} {
		t.Run(backend, func(t *testing.T) {
			t.Setenv("MW_ADMIN_TOKEN", "operator")
			cfg, err := config.Load("../../configs/gateway.scheduling.example.yaml")
			if err != nil {
				t.Fatal(err)
			}
			cfg.Tokenizer.Endpoints = []string{fakeTokenizer(t)}
			cfg.Server.ExposeRoutingHeader = true
			cfg.KVIndex.BlockSize = 2
			cfg.Models[0].Aliases = []string{"cache-alias"}
			if backend == "redis" {
				mini := miniredis.RunT(t)
				cfg.State.Backend = "redis"
				cfg.State.Redis.URL = "redis://" + mini.Addr()
				cfg.State.Redis.Namespace = t.Name()
			}
			newServer := func() *Server {
				s, err := New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), "scheduling-test")
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(s.Close)
				return s
			}
			a := newServer()
			_, target := mockWorker(t)
			for _, m := range cfg.Models {
				for _, suffix := range []string{"hot", "idle"} {
					used, running, zone := int64(0), 0, "east"
					if suffix == "hot" {
						used, running, zone = 8000, 80, "west"
					}
					_, _, err := a.reg.UpsertContext(context.Background(), registry.Spec{
						ID: m.Name + "-" + suffix, Model: m.Name, Role: registry.RoleUnified, Target: target,
						KVCapacityTokens: 10000, MaxRunningRequests: 100, Labels: map[string]string{"zone": zone},
					}, registry.SourceStatic, &registry.Beat{State: &registry.EngineState{
						KVUsedTokens: used, KVCapacityTokens: 10000, RunningRequests: running, MaxRunningRequests: 100,
					}})
					if err != nil {
						t.Fatal(err)
					}
				}
			}
			b := a
			if backend == "redis" {
				b = newServer()
			}
			seed := func(model string) {
				for _, suffix := range []string{"hot", "idle"} {
					a.index.ForgetWorker(a.reg.Get(model + "-" + suffix).CacheKey())
				}
				hashes := kvindex.BlockHashes([]uint32{0, 11, 12, 128821}, 2, kvindex.ModelSeed(model))
				a.index.Observe(a.reg.Get(model+"-hot").CacheKey(), hashes)
			}
			infer := func(model, expected string) {
				t.Helper()
				body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hello"}]}`, model)
				w := request(b, "POST", "/v1/chat/completions", body)
				if w.Code != http.StatusOK || w.Header().Get("X-ModelWeave-Decode-Worker") != expected {
					t.Fatalf("model %s: HTTP %d worker=%q want=%q: %s", model, w.Code, w.Header().Get("X-ModelWeave-Decode-Worker"), expected, w.Body)
				}
				t.Logf("%s model=%s selected=%s", backend, model, expected)
			}
			for _, tc := range []struct{ model, want string }{
				{"cache-first", "cache-first-hot"}, {"latency-first", "latency-first-idle"}, {"zone-preferred", "zone-preferred-hot"},
			} {
				seed(tc.model)
				infer(tc.model, tc.want)
			}
			infer("cache-alias", "cache-first-hot")
			// A high-scoring cached/preferred instance cannot bypass operator
			// isolation or hard capacity, irrespective of the model's weights.
			decodeAdmin(t, adminRequest(a, "PUT", "/admin/blacklist/cache-first-hot", `{"reason":"isolation overrides score"}`, "operator"), &registry.Block{})
			infer("cache-first", "cache-first-idle")
			decodeAdmin(t, adminRequest(a, "DELETE", "/admin/blacklist/cache-first-hot", "", "operator"), &map[string]any{})
			seed("cache-first")
			infer("cache-first", "cache-first-hot")
			if err := a.reg.Get("cache-first-hot").UpdateState(registry.EngineState{RunningRequests: 100, MaxRunningRequests: 100}, time.Now()); err != nil {
				t.Fatal(err)
			}
			infer("cache-first", "cache-first-idle")
			if got := b.models["latency-first"].rt.Prefill.Scorers(); len(got) != 1 || got[0].Name != "load" {
				t.Fatalf("role override not wired into gateway: %+v", got)
			}
			var m adminModelView
			decodeAdmin(t, adminRequest(b, "GET", "/admin/models/zone-preferred", "", "operator"), &m)
			if params := m.Policies[registry.RoleUnified].Scorers[0].Params; params["key"] != "zone" || params["value"] != "west" {
				t.Fatalf("active plugin params missing from admin view: %v", params)
			}
		})
	}
}

// This implementation depends only on the public SDK. Registration through
// this API must be sufficient to configure a new weighted algorithm.
type externalStyleScorer struct{ name string }

func (s externalStyleScorer) Name() string  { return s.name }
func (externalStyleScorer) UsesCache() bool { return false }
func (externalStyleScorer) Score(req *scoring.Request, c *scoring.Candidate, _ []*scoring.Candidate) float64 {
	if req.Model == "m" && c.Labels["accelerator"] == "preferred" {
		return 1
	}
	return 0
}

func TestPublicSDKPluginCanExtendTheGatewayWithoutCoreChanges(t *testing.T) {
	name := fmt.Sprintf("external-%d", time.Now().UnixNano())
	if err := scoring.Register(scoring.Descriptor{Name: name, Version: "test", Description: "Public SDK extension."}, func(scoring.Config) (scoring.Scorer, error) {
		return externalStyleScorer{name: name}, nil
	}); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Parse([]byte(fmt.Sprintf(`
tokenizer: {endpoints: ["http://127.0.0.1:1"]}
models:
  - name: m
    routing:
      default:
        policy: weighted
        scorers:
          - {name: %q, weight: 3}
          - {name: load, weight: 1}
`, name)))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), "extension-test")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	for _, id := range []string{"preferred", "other"} {
		if _, _, err := s.reg.Upsert(registry.Spec{ID: id, Model: "m", Role: registry.RoleUnified, Target: "host:9001", Labels: map[string]string{"accelerator": id}}, registry.SourceStatic); err != nil {
			t.Fatal(err)
		}
	}
	d, err := s.models["m"].rt.Unified.Schedule(context.Background(), &scheduler.Request{Model: "m", Role: registry.RoleUnified})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Reservation.Release(true)
	if d.Worker.ID() != "preferred" || d.Scores[name] != 1 {
		t.Fatalf("external-style extension ignored: %+v", d)
	}
}
