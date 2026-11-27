# Code organization and scheduling framework

ModelWeave borrows the **contract / runtime / plugin / profile assembly** split
from [kube-scheduler's scheduling framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/),
without importing Kubernetes libraries or turning inference routing into Pod
scheduling. A scheduling unit is an inference stage; its candidates are registered
mwip engine instances. Kubernetes still schedules ModelWeave's own Pods normally.

## Boundaries

```text
cmd/modelweave-gateway/       CLI, config validation, process lifecycle
internal/gateway/            HTTP APIs/admin, composition root, observability
internal/qos/                tenant admission and per-model fair queues
internal/orchestrator/       E/P/D planning, execution, retry and abort
internal/scheduler/
  scheduler.go               one scheduling-cycle coordinator
  cycle.go                   prefix lookup, snapshots, filtering, load preparation
  reserve.go                 atomic capacity claims and prefill cost
  compat.go                  stable aliases/wrappers for existing integrations
  framework/
    types.go, status.go      request/candidate/policy/filter contracts
    runtime/                 registry, Filter/Score execution, weighted SDK bridge
    plugins/                 mandatory Eligibility and Capacity filters
  policy/                    standalone selection implementations
  profile/                   in-tree policy factories and SDK-plugin assembly
internal/scheduling/         private, data-only scorer contract and registry
internal/scheduler/plugins/  independently registered in-repository algorithms
internal/registry/           membership snapshots and per-instance handles
internal/redisstate/         shared instance state and fenced reservations
internal/kvindex/            advisory prefix metadata
deploy/compose/              local/full-functional e2e stack
deploy/kubernetes/           Kustomize base and GPU-free demo overlay
```

The framework does not import profile assembly, concrete policies, gateway code
or configuration. `architecture_test.go` enforces that dependency direction.
Only profile assembly installs built-ins; the runtime executes already selected
algorithms. The data-only plugin contract has no registry/store/reservation
handles. All implementation packages, including plugin contracts and algorithms,
live under `internal`; Go enforces the repository import boundary. There is no
separate `pkg` tree or externally supported Go SDK. Add plugins within this
repository and rebuild. The generated `api/mwip/v1` engine protocol remains
public so independent engines can implement mwip without importing internals.

## One cycle

```text
registry refresh + advisory prefix lookup (outside the policy lock)
  -> fresh batched snapshot
  -> Filter: Eligibility -> Capacity
  -> normalized load preparation
  -> Score / policy selection, checked against the filtered candidate list
  -> Reserve: atomic store capacity + revision check
  -> Decision; on a conflicting snapshot, repeat snapshot/filter/score/reserve
```

This is deliberately a subset of kube-scheduler's extension model, not its full
PreFilter/PreScore/Permit/Bind lifecycle. In particular, advisory cache lookup
remains outside the policy lock; actual request dispatch remains in the
orchestrator. Reserving capacity is a required store operation, not an optional
scoring plugin or a substitute for Kubernetes binding.

- **Eligibility** rejects wrong model/role, gone, unhealthy, drained, blocked,
  excluded, non-pinned and unsupported-normal candidates.
- **Capacity** checks running slots, KV space and prefill backlog. It reports
  `Unschedulable`, while ineligible instances report
  `UnschedulableAndUnresolvable`. The coordinator maps these to the existing
  `ErrNoCapacity` (can queue) versus `ErrNoWorkers` behaviour.
- The mandatory filters cannot be disabled by YAML scorer configuration. A
  policy returning a pointer outside the filtered list is rejected.
- Final reserve still uses the registration epoch, state/drain/blacklist revision
  and capacity bounds atomically in memory/Redis. A concurrent block cannot slip
  between scoring and dispatch by creating a new reservation.
- Policies/scorers are constructed separately for each model/role. Existing
  `routing.default`, role overrides, weights, parameter validation, standalone
  policies and scorer registration interfaces retain their behaviour. No wire
  protocol, Redis schema or YAML migration is introduced by this reorganization.

See [plugin interfaces and examples](../internal/scheduler/plugins/README.md) and
[state/admission consistency](state-and-scheduling.md) for implementation contracts.

## Validation

```sh
make test-go           # vet + race tests, including dependency/filter/reserve tests
make test-rust         # tokenizer tests
make test-admin-ui     # dependency-free view tests
make test-admin-browser  # optional Chromium; setup in docs/admin.md
make test-e2e          # isolated full stack, memory backend
make test-e2e-redis    # same full suite, real Redis + another gateway replica
```

Both e2e targets build the current images, use a fresh compose project/network,
publish no host ports, use test-only credentials and clean up only their own
containers. The primary and event-index gateways retain separate state namespaces;
the Redis-only replica shares the primary's namespace/config. Artifacts are saved
under `tmp/modelweave-e2e-*/`. The extra model verifies actual weighted routing,
drain/resume and blacklist survival across re-registration; in Redis mode engines
heartbeat to one gateway while inference goes through the other. Existing tests
exercise all client APIs, streaming, tools, logprobs, JSON/stop/length, retry,
admission/rate limits, Normal/PD/EPD, push/pull handshakes, lifecycle and multimodal
DeepSeek/Qwen behaviour with the real Rust tokenizer, without GPUs.

Instance-removal counters remain **replica-local**: only the replica committing
expiry increments its counter. The multi-replica lifecycle assertion aggregates
those counters over the shared primary fleet and still requires exactly one
increment. Shared gauges must not be summed.

Kubernetes support is validated offline only; no test creates a cluster or
changes a kubeconfig. See [deployment instructions](../deploy/kubernetes/README.md).
