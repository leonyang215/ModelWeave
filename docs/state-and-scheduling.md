# Shared state and scheduling plugins

## Memory or Redis

`state.backend: memory` is the default and needs no Redis. All mutable state
is kept inside one gateway. Existing configurations continue to work.

For gateway replicas, select `state.backend: redis`. Engines heartbeat to
**any one** replica (or a load-balanced Registry endpoint); they do not have
to register with every gateway. Use the same Redis URL, namespace,
`discovery.heartbeat_ttl` and `kv_index.block_size` for all replicas. A
namespace isolates fleets; incompatible heartbeat TTLs/block sizes/storage
schemas in one namespace are rejected at startup. Keep model names, static
instance specs, KV-index settings and control tokens consistent too. Tenant
policies and each model's admission settings must match across replicas;
incompatible settings are rejected rather than splitting quota or fairness.

```yaml
state:
  backend: redis
  sync_interval: 250ms
  redis:
    url: "${MW_REDIS_URL}"       # redis://... or rediss://...; credentials via environment
    namespace: production-a
    timeout: 1s
    lease_ttl: 15s
```

| State | Memory | Redis |
|---|---|---|
| Instance membership/spec, model/role pools | local | shared, versioned membership |
| Heartbeat liveness, engine reports, operator/engine drain | local | shared; atomic heartbeat update |
| Operator instance-ID blacklist (including offline instances) | process-local; survives re-registration, not gateway restart | shared; survives re-registration and gateway restarts, subject to Redis persistence |
| Passive-health failure count and cooldown | local | shared |
| In-flight requests, KV/prefill reservations | local atomic check+reserve | global atomic conditional check+reserve |
| Approximate prefix observations and engine KV events | local | shared; TTL, bounded approximate index |
| Session affinity, embedding-artifact metadata | local | shared per model; TTL and bounded artifact LRU |
| Tenant RPM/TPM/concurrency limits | per gateway | global per tenant across models and gateways |
| Admission queue bound, priority, weighted fairness, backfill, shedding | per model in this process | one shared queue per model across gateways |
| Overload hysteresis / probabilistic shedding | derived locally | derived locally from shared instance/queue utilization; not a quota counter |
| Request/latency/completion/error metrics and access logs | per gateway | per gateway; worker capacity/load gauges show shared totals |

Redis stores **metadata**, not KV tensors or embedding bytes. Artifact URIs
must be reachable from every engine that reuses them.

### Guarantees and failure behaviour

- Registration, spec/source changes and removals advance a durable membership
  version. Request-path refreshes and periodic syncs discover changes even
  after a missed notification/disconnect; Redis Pub/Sub is not required.
- Heartbeat expiry and load/health ages use **Redis TIME**, not gateway clock
  synchronization. Only heartbeats refresh instance liveness; `GetState`
  polls do not. One replica commits an expiration/removal, while all replicas
  observe it. Static registrations never expire; in Redis they persist until
  explicitly deregistered, even if removed from a gateway's YAML.
- The operator drain is independent of engine heartbeats and remains set
  across them. A new registration epoch is allocated after removal/expiry or
  a model/role/target change. Late polls and releases cannot affect a newer
  epoch; prefix entries and sessions are epoch-scoped as well.
- [Admin blacklist rules](admin.md) are independent of the registration epoch:
  removal, heartbeat expiry, a changed model/role/target and re-registration do
  not clear a blocked ID. Both scheduling snapshots and the atomic reservation
  reject it. Unblocking does not override drains or passive-health cooldowns.
- Selection uses shared load snapshots. The final reservation verifies the
  registration, state/drain/health revision and capacity bounds **atomically
  in Redis**, retrying on contention. There is no process-local lock acting
  as a substitute for distributed admission. Engine usage remains the truth:
  effective KV usage is `max(all live reservations, reported usage + KV
  reserved since that report)`. Engines still enforce their hard limits.
- Each gateway renews a reservation-owner lease. Requests can outlive
  `lease_ttl` while their gateway keeps renewing. A crashed gateway's claims
  stop counting when its lease expires; remaining gateways reclaim them on
  state reads. Releases and early prefill releases are idempotent; failed or
  ambiguous releases are retried by maintenance. An expired owner is fenced
  and must **restart** before admitting again (it cannot revive old claims).
  Size `lease_ttl` above expected pauses/outages; it must be at least three
  operation timeouts. `sync_interval` must be at most `lease_ttl/3`.
- Tenant concurrency leases and queued tickets use that **same gateway-owner
  lease**. A crashed gateway cannot strand global tenant slots or queue places.
  Limits do not reset when a replica starts/stops. Failed token reconciliation
  and ticket cleanup are idempotently retried after a transient outage.
- Shared poll leases avoid redundant `GetState` calls across replicas. A
  slow poll cannot overwrite a newer heartbeat/state mutation.
- Redis unavailability fails startup, readiness, registry mutations and
  admission closed (`503` / gRPC `UNAVAILABLE`), **never silently falls back
  to independent memory**. Advisory-cache read failures are misses, and KV
  event writes are not acknowledged until stored. Recovery within the owner
  lease requires no restart; after lease expiry, restart that gateway.
- Use a dedicated Redis primary endpoint (Redis 7+ recommended), TLS/ACLs in
  production, persistence/HA as appropriate and `maxmemory-policy noeviction`.
  Arbitrary key eviction, database flushes, or accepted writes lost during an
  asynchronous Redis failover are outside the consistency guarantee. A
  multi-primary Redis Cluster client is not configured by this implementation.
  Closing one gateway releases **its own** lease/client, not other gateways'
  registrations or caches. Do not flush a live namespace to reconfigure it;
  use a fresh namespace for incompatible settings.

### Two replicas locally

Start a private Redis and the tokenizer, then build the Go binaries
(`make build-go`). With [gateway.redis.example.yaml](../configs/gateway.redis.example.yaml):

```sh
export MW_REDIS_URL=redis://127.0.0.1:6379/0 MW_CONTROL_TOKEN=local-control-token
# Terminal A
MW_API_LISTEN=127.0.0.1:8080 MW_REGISTRY_LISTEN=127.0.0.1:8081 \
  bin/modelweave-gateway -config configs/gateway.redis.example.yaml
# Terminal B (same exports)
MW_API_LISTEN=127.0.0.1:8180 MW_REGISTRY_LISTEN=127.0.0.1:8181 \
  bin/modelweave-gateway -config configs/gateway.redis.example.yaml
# Terminal C: one registration target is enough for both gateways
bin/modelweave-mockengine -id u0 -role unified -registry 127.0.0.1:8081 \
  -control-token "$MW_CONTROL_TOKEN" -kv-events
curl -H "X-ModelWeave-Token: $MW_CONTROL_TOKEN" localhost:8180/admin/workers
# Draining via A also stops B from assigning new requests to u0.
curl -X POST -H "X-ModelWeave-Token: $MW_CONTROL_TOKEN" localhost:8080/admin/workers/u0/drain
```

## Tenant limits and admission: two implementations

`state.backend` selects the **whole** QoS backend through `qos.Backend`,
`TenantLimiter` and `AdmissionQueue`. There is no independent QoS switch and
no Redis mode with an unnoticed local limiter/queue.

```yaml
state:
  backend: redis                  # memory: the same semantics in one process
  redis: {url: "${MW_REDIS_URL}", namespace: fleet-v3}
tenants:
  - name: team-a
    api_keys: [sk-team-a]
    priority: high
    weight: 2
    rpm: 600                      # 600 total, NOT 600 per gateway
    tpm: 2000000
    max_concurrency: 128           # includes queued/encoding/running requests
models:
  - name: deepseek-v4.1
    admission:
      queue_size: 512              # one bound for this model across the fleet
      max_wait: 30s
      backfill_depth: 4
```

### Tenant quotas

- RPM and TPM are continuously refilled token buckets (one minute of burst
  capacity), keyed by **tenant name**, not API key, model or gateway ID. The
  concurrency check and both bucket charges are atomic; a TPM rejection
  does not consume RPM or a concurrency slot. A rejected limit returns
  `429` with `Retry-After` in both modes.
- TPM is reserved using prompt plus capped output, then reconciled once with
  actual prompt/output usage at completion. Underestimates create debt;
  overestimates refund tokens. An oversized request can borrow from a full
  bucket, like the memory implementation. Repeated completion/cleanup does
  not double-refund or free another request's slot.
- When a gateway dies, its concurrency slots stop counting after its owner
  lease expires. Unknown actual usage is **not** invented or automatically
  refunded: the conservative token estimate remains charged and refills
  normally. Starting/stopping a gateway never creates fresh global buckets.

### Shared admission

- Each canonical model has one queue, one size bound, strict priority classes,
  per-tenant virtual-token counters (VTC), FIFO tie-breaking, lower-priority
  shedding and the configured backfill depth. A model alias uses the same
  queue. New requests cannot bypass its head via a replica-local fast path.
- Redis stores only ticket metadata (owner, tenant, priority, weight, cost,
  sequence, deadline and turn token). Bodies, callbacks, engine streams and
  client connections stay on the gateway that accepted the request. It runs
  its own scheduling callback when the global queue gives it a turn.
- A turn's fencing token is checked **inside the worker reservation script**.
  Cancellation/timeout can revoke the turn while a slow callback is still
  running. A late callback cannot reserve under a stale token; a ready plan
  is rolled back if its queue commit fails or has an ambiguous result. An
  in-turn lower-priority ticket being shed may still complete its existing
  attempt, but cannot requeue after a failure.
- Capacity notifications advance a durable wake version. A failed scheduling
  result cannot overwrite a wake received during its attempt. Pub/Sub gives
  low-latency cross-gateway wakeups; periodic durable polling recovers missed
  messages. Correctness does not depend on reliable notification delivery.
- Client disconnects/timeouts clean up their tickets. Closing a gateway drops
  only its own tickets; another gateway prunes dead-owner tickets after lease
  expiry. A process crash does not migrate/replay its HTTP request body.
- This is initial request admission. Engine retries/fallback of an already
  admitted request retain its tenant lease and existing retry semantics;
  they do not acquire a second tenant concurrency slot.

Storage failures return `503`, not a local quota/queue fallback. Readiness
checks maintenance as well. `/admin/stats` reports `qos_backend`, global
tenant balances/concurrency and global per-model queue lengths; shared
queue-depth gauges must not be summed across replicas. Request outcomes,
rejections and latency counters remain local so they **can** be summed.

The storage schema is now **3** (schema 2 introduced shared QoS; schema 3 adds
instance-ID blacklisting). A mixed rollout with older gateways is rejected:
they cannot enforce the blacklist. Stop the old fleet and use a fresh namespace
for the coordinated rollout; never send traffic to old and new fleets as if
they shared isolation rules. Engines must heartbeat again to populate the new
namespace; existing rules/state are not copied automatically. All gateways in a live namespace must agree on tenant policies and
per-model queue settings. API keys can rotate without changing tenant names
or resetting their quotas. Redis ACLs must permit the namespace's data/script
operations and the QoS wake channel's `PUBLISH`/`SUBSCRIBE` operations.

## Scheduling plugins and weights

Scoring plugins live in [`plugins/scheduling/`](../plugins/scheduling/README.md)
and register metadata/factories through the public `pkg/scheduling` SDK.
`internal/scheduler` composes them and retains the legacy policy/scorer
registries. They are compiled-in Go plugins, not runtime `.so` loading or an
executable-directory scan. Adding a scoring plugin requires an import and a new
binary, not a new switch case in the scheduling loop.

`models[].routing.default` defines a model-wide composition. Omitted/empty roles
inherit it; any non-empty role replaces the whole composition without merging.
Without a model default, historical per-role defaults are unchanged. See the
[complete multi-model example](../configs/gateway.scheduling.example.yaml) and
[configuration resolution rules](../plugins/scheduling/README.md#configure-each-model-independently).

The default P/D/unified policy is now **`weighted`**, with two pre-registered
scorers, each weight **1**. Encode defaults to the `load` policy. Each role of
each model can configure its own composition:

```yaml
routing:
  prefill:
    policy: weighted
    scorers:
      - {name: prefix-cache, weight: 2}
      - {name: load, weight: 1}
  decode:
    policy: weighted
    scorers:
      - {name: prefix-cache, weight: 1}
      - {name: load, weight: 3}
  encode: {policy: load}
```

| Scorer | Score (higher is better) |
|---|---|
| `prefix-cache` | fraction of prompt tokens cached on the candidate (0..1) |
| `load` | `1 - normalized_load`; P uses queued prefill work, E uses running requests, D/unified use running/waiting requests and KV utilization |
| `affinity` (optional) | 1 for the session's rendezvous-hashed worker, 0 otherwise |
| `label` (optional) | 1 for an exact instance-label match; `params: {key: zone, value: west}`; a soft preference, not a hard filter |

`score = Σ(weight × scorer_score)`; the highest score wins, with random tie
breaking. Plugin scores are bounded to [0,1] (NaN/Inf becomes 0). Weights
must be finite/non-negative, with a positive finite sum; **0 disables a
term**, including its cache lookup requirement. Unknown/duplicate scorers
and invalid weights fail configuration validation. An omitted/empty list
uses the default two terms. Use `policy: prefix-cache` or `policy: load` for
a single built-in scorer. Filtering health/drain/normal capability/capacity
always precedes plugins; no score bypasses admission.

Explicit legacy policies (`cache_aware`, `least_load`, `power_of_two`,
`round_robin`, `random`, `session_affinity`) remain registered. In particular,
set `policy: cache_aware` to retain the old default threshold/rebalancing
behaviour. The old `weighted` fields `cache_weight`, `load_weight` and
`affinity_weight` are translated into scorer terms, but cannot be mixed with
`scorers`. A session-affinity policy can use `fallback: weighted` and a scorer
list for requests without a session key.

### Add a plugin

Implement `pkg/scheduling.Scorer`, register a descriptor and factory, and add a
blank import in `plugins/scheduling/plugins.go` (or a custom gateway binary).
Then configure `{name: my-scorer, weight: 0.5}` in any model/role. See the
[SDK guide and complete extension example](../plugins/scheduling/README.md#add-a-plugin).
Scorers receive detached scheduling views without registry/reservation handles;
parameter decoding is strict and mutable config maps are not shared across
factories or introspection. Scorers must be concurrency-safe and not mutate views.

Whole selection policies can still use the in-module `scheduler.Policy` and
`RegisterPolicy` interface. Existing `RegisterScorer` integrations remain
supported, while the public SDK is preferred for new code. Nil factories and
duplicate registrations are rejected; plugin discovery does not invoke factories.

`bin/modelweave-gateway -check-config -config <file>` validates model policies,
weights and plugin parameters without connecting to backends. Configuration is
static for the life of a gateway: deploy/restart all replicas with consistent
configuration to apply changes; there is no hidden live-config store.

The admin **调度插件** page and authenticated `GET /admin/scheduling/plugins`
expose the plugin catalog, versions, directories and parameter descriptions.
The UI also lists effective per-model/role combinations and YAML templates.
`Decision.Score` and `Decision.Scores` expose the chosen worker's aggregate
and per-plugin scores. `/admin/stats` exposes active policy/scorer weights
and `state_backend`; `/admin/workers` and Prometheus worker gauges read the
shared instance state (do not sum these global gauges across replicas).

## Verification

```sh
make test-go                       # includes in-process Redis contract/integration tests, -race
# Optional: the same gateway/index/cache/store tests against a PRIVATE real Redis.
MW_TEST_REDIS_URL=redis://127.0.0.1:6379/0 go test -race -count=1 \
  ./internal/redisstate ./internal/kvindex ./internal/orchestrator ./internal/qos ./internal/gateway
```

Virtual-clock expiry and injected-outage tests use miniredis (no external
service) and skip those injections for the real-Redis run. Every test uses
an isolated namespace; no test flushes database contents.
