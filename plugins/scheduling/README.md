# Scheduling plugin directory

This directory contains **compiled-in, composable scoring algorithms**. The
gateway imports `plugins.go`; each child package registers itself through the
public [`pkg/scheduling`](../../pkg/scheduling/plugin.go) SDK. Adding an algorithm
does not require a new switch case in the scheduler or any engine-protocol
change. A plugin can also live in another Go module: it imports only the public
SDK, not ModelWeave's `internal` packages.

```text
plugins/scheduling/
  plugins.go              # explicit import list / compiled plugin allowlist
  prefixcache/plugin.go   # prefix-cache
  load/plugin.go          # load
  affinity/plugin.go      # affinity
  label/plugin.go         # label + validated per-model parameters
pkg/scheduling/plugin.go  # scorer contract, registry, metadata, parameter helpers
```

The gateway never scans a writable directory for executable code, loads `.so`
files, or runs uploaded scripts. New code must be imported, reviewed, tested and
compiled into a new gateway binary/image. Merely copying a Go file to a running
gateway does not install a plugin. Registration errors fail startup instead of
silently replacing an existing algorithm.

## Built-ins

| Plugin | Score (higher is better) | Parameters |
|---|---|---|
| `prefix-cache` | fraction of prompt tokens already cached, `[0,1]` | none |
| `load` | `1 - normalized_load`; the scheduler incorporates running/waiting requests, prefill backlog and KV occupancy according to role | none |
| `affinity` | `1` for the rendezvous-hashed session instance, `0` otherwise; `0` without a session key | none |
| `label` | `1` when an instance has the exact configured label value, `0` otherwise | required string `key` (non-empty) and string `value` |

`label` is a **soft preference**, not a whitelist, authorization check, or hard
locality constraint. A non-matching instance remains eligible if the matching
ones are full, drained, blocked or unhealthy. An explicit empty value matches
only an existing label with an empty value, not a missing label.

Weighted scoring happens **after** health/drain/blacklist/capacity filtering:

```text
total(candidate) = sum(weight_i * clamp(score_i(candidate), 0, 1))
```

The maximum total wins; ties are randomized. Weights are relative coefficients,
**not probabilities**, and do not have to sum to one. They must be finite and
non-negative with a finite, positive total. An explicit zero disables execution
and cache lookup for that term, while its name/parameters are still validated.
Unknown/duplicate names, invalid parameters and all-zero/NaN/Inf/negative weights
fail config validation. An omitted/empty scorer list preserves the historical
`prefix-cache=1, load=1` defaults. Invalid plugin scores (NaN/Inf) become zero;
other out-of-range scores are clamped. Score breakdowns appear on scheduler
decisions. No score can bypass the final atomic capacity reservation.

`weighted` is the composition policy. Legacy standalone selection policies
(`round_robin`, `random`, `power_of_two`, `least_load`, `cache_aware`,
`session_affinity`) remain available, but **are not scorer terms**: do not put
them in `scorers`. A `session_affinity` policy can still use a weighted fallback
for requests without a session key.

## Configure each model independently

[`configs/gateway.scheduling.example.yaml`](../../configs/gateway.scheduling.example.yaml)
is a complete, validated three-model example. Model-wide defaults reduce
duplication; a role can replace its whole policy/composition:

```yaml
models:
  - name: cache-first
    tokenizer_model: Qwen3-8B
    mode: normal
    routing:
      default:
        policy: weighted
        scorers:
          - {name: prefix-cache, weight: 3}
          - {name: load, weight: 1}
      encode: {policy: load}

  - name: latency-first
    tokenizer_model: Qwen3-8B
    mode: normal
    routing:
      default:
        policy: weighted
        scorers:
          - {name: prefix-cache, weight: 0.25}
          - {name: load, weight: 4}
      prefill:
        scorers:
          - {name: load, weight: 1}
      encode: {policy: load}

  - name: zone-preferred
    tokenizer_model: Qwen3-8B
    mode: normal
    routing:
      default:
        scorers:
          - {name: label, weight: 2, params: {key: zone, value: west}}
          - {name: load, weight: 1}
          - {name: affinity, weight: 0.5}
      encode: {policy: load}
```

Resolution rules:

1. A non-empty `routing.<role>` completely overrides `routing.default`; there
   is **no implicit merging** of policy fields, plugin lists, weights or params.
2. An omitted/empty role inherits `routing.default` if it is set, including the
   encode role. Override encode explicitly if it should remain load-only.
3. With no model default, historical behaviour is unchanged: weighted
   P/D/unified (`prefix-cache=1, load=1`) and load-only encode.
4. Defaults are validated even when all roles override them. Model aliases use
   the canonical model's same runtime composition, not separate configuration.

Models and roles get separate factory calls/instances and detached parameter
maps. Active configuration is read-only after construction. YAML changes take
effect after rebuilding only if plugin code changed, and **restarting/redeploying
the gateway** for any configuration change. Apply consistent config to all
replicas; this is not a distributed live-configuration controller. No Redis
storage schema change is introduced by routing composition configuration.

Validate without connecting to a tokenizer, engine or Redis:

```sh
make build-go
MW_ADMIN_TOKEN=example-operator-token bin/modelweave-gateway \
  -check-config -config configs/gateway.scheduling.example.yaml
# configuration valid: 3 models, 4 scoring plugins
```

The catalog is available in the admin console's **调度插件** page or through
authenticated **`GET /admin/scheduling/plugins`**. It lists metadata, source
directories, parameter descriptions and available policies without constructing
parameterized plugins. The console shows effective weights for every model/role,
including inherited defaults and overrides, and generates editable YAML
templates. It does not claim to mutate live scheduling when showing a template.
`GET /admin/models/{name}` and `/admin/stats` expose active weights and detached
plugin parameters. See [admin security](../../docs/admin.md).

## Add a plugin

1. Add `plugins/scheduling/<your-plugin>/plugin.go` (or use another Go module).
2. Implement `scheduling.Scorer` using only the SDK data views.
3. Register a descriptor and factory from `init`. Include version, description,
   directory and parameter documentation; strictly decode/validate parameters.
4. Add a blank import in `plugins/scheduling/plugins.go`. For an out-of-tree
   module, add its blank import to a file in `cmd/modelweave-gateway` and declare
   the module dependency instead. Do not edit `BuildScored` or `Schedule`.
5. Add tests, build the binary/image, configure the plugin by name and deploy.

For example, a custom `kv-headroom` package can contain:

```go
package kvheadroom

import "github.com/leonyang215/ModelWeave/pkg/scheduling"

type Scorer struct{}

func (Scorer) Name() string    { return "kv-headroom" }
func (Scorer) UsesCache() bool { return false }
func (Scorer) Score(_ *scheduling.Request, c *scheduling.Candidate, _ []*scheduling.Candidate) float64 {
    if c.KVCapacity <= 0 { return 0.5 }
    return 1 - float64(c.KVUsed)/float64(c.KVCapacity)
}

func init() {
    err := scheduling.Register(scheduling.Descriptor{
        Name: "kv-headroom", Version: "1.0.0",
        Description: "Prefer spare KV capacity.",
        Source: "plugins/scheduling/kvheadroom",
    }, func(c scheduling.Config) (scheduling.Scorer, error) {
        if err := scheduling.DecodeParams(c.Params, &struct{}{}); err != nil {
            return nil, err
        }
        return Scorer{}, nil
    })
    if err != nil { panic(err) }
}
```

Then compose `{name: kv-headroom, weight: 2}` with other scorers. The `label`
package is a runnable example of a plugin with per-model parameters.

### Plugin contract

- Return raw preference in `[0,1]`; **do not apply the configured weight inside
  the plugin**. `UsesCache` declares whether gateway prefix-index lookup is needed.
- Requests/candidates are read-only views containing canonical model, role,
  session key, cached fraction, normalized/raw load, labels and capacity. There
  are no registry/store/reservation handles. Labels do not alias live registry
  maps; factories and introspection do not share mutable parameter maps.
- Use `DecodeParams` with typed structs instead of asserting YAML-specific
  numeric types. It rejects unknown fields and preserves integer precision.
  Validate required fields, finite values and ranges yourself; metadata is
  documentation, not a substitute for factory validation.
- Factories should be cheap and side-effect-free: config validation calls them
  too. Do not start background jobs or open resources requiring shutdown.
- Scorers must be concurrency-safe and not mutate the provided views. Do not
  perform blocking/network I/O in the hot scoring loop. Stateful extensions
  must keep state inside the per-model/role instance, not global variables.
- Plugins are trusted native code, **not a security/fault sandbox**. Score
  clamping does not isolate arbitrary panics or side effects.
- Existing `internal/scheduler.RegisterScorer` in-module extensions remain
  compatible; new code should use the public SDK. Duplicate SDK/legacy names
  are rejected when registering/building rather than silently shadowed.

## Verification

```sh
make test-go          # includes SDK/directory/core tests and real HTTP inference routing
make test-admin-ui    # catalog filtering and YAML-template safety
make test-admin-browser  # optional Playwright setup: docs/admin.md
```

The gateway integration test runs three independently configured models against
a gRPC mock engine in memory and Redis modes (the latter through another gateway),
verifies weight-dependent selections, aliases, label preferences and role
overrides, and confirms blacklist/capacity protections still win. SDK tests cover
parameter isolation, nil/error factories, concurrent discovery/registration,
score clamping, disabled terms and external-style plugins without core changes.
