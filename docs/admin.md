# Admin console and API

The gateway embeds its management console at **`/admin/`** on the existing HTTP
listener (`:8080` by default, `:38080` in the local Compose stack). `make build-go`
and the gateway Docker target include all HTML/CSS/JavaScript. There is no CDN,
frontend build, Node runtime, or separate admin service in production.

## Console

- **Overview**: configured models, eligible instances, in-flight reservations,
  queue lengths, pool distribution, overload levels and backend/version/uptime.
- **Models**: canonical names and aliases, tokenizer/engine names, context cap,
  execution mode, handshake, per-role inventory and active policies/scorer weights.
- **Instances**: search by ID/target/labels and filter by model/role/status; inspect
  KV capacity/use, reservations, heartbeat/report ages and complete snapshots.
- **Blacklist**: isolate or restore an ID, including offline/not-yet-registered IDs;
  inspect the reason and last rule-update time.
- **Tenant quotas**: read-only configured limits, balances and concurrency. API
  keys are never returned. YAML remains the source of model/tenant/routing config.
- **Scheduling plugins**: compiled plugin catalog, source directories, versions,
  parameter schemas, effective per-model/role weights and editable YAML templates.
  Templates are not live mutations; validate with `modelweave-gateway -check-config`
  and redeploy. See the [plugin SDK and configuration guide](../plugins/scheduling/README.md).

The console refreshes every 5 seconds while visible (can be paused), and supports
manual refresh. Mutations require confirmation; blocking requires a reason. A
failed refresh marks the last snapshot stale and disables mutations. A `401`
clears the current data/credential and asks the operator to reconnect.

“Available”/“可调度” means passing the health, drain and blacklist gates, **not**
guaranteed capacity for a particular request or complete P/D/E availability for
every modality. Unknown/stale engine reports are marked separately, not presented
as fresh zero load. Instance completion/error counters remain replica-local.

## Authentication and exposure

```yaml
server:
  control_token: "${MW_CONTROL_TOKEN}"  # engine Registry credential
  admin_token: "${MW_ADMIN_TOKEN}"      # distinct operator credential
  disable_admin_ui: false              # true: remove UI/assets, retain admin API
```

All admin API calls use **`X-ModelWeave-Token`**. When `admin_token` is set,
`control_token` does **not** authorize HTTP admin operations, and the admin token
does not authorize engine Registry calls. Tenant API keys do not grant admin
access. For backwards compatibility, an empty `admin_token` falls back to
`control_token`. If both are empty the admin API remains unauthenticated, the
gateway logs a warning, and the console displays a warning. This mode is only
appropriate in a trusted local environment.

The public UI shell/assets contain no credentials or gateway inventory. The token
is kept only in page memory and sent in a header; it is not stored in cookies,
local/session storage, or URLs. Reloading the page requires re-entry. Responses
use `Cache-Control: no-store`, a restrictive same-origin CSP, anti-framing and
`nosniff` headers. Cross-origin browser mutations are rejected, even in tokenless
mode; CLI calls without browser-origin headers remain supported. CORS is not
enabled for admin APIs. Mount the console at the origin root (`/admin/`), not
under a rewritten URL prefix.

**Production**: set a strong, separate admin token, terminate HTTPS at a trusted
reverse proxy, restrict `/admin` and `/admin/*` to the management network (or an
SSO/auth proxy), and keep all replicas' tokens consistent. Disabling the UI is
not disabling the API. `/v1/models` retains its existing public OpenAI-compatible
listing; this change does not change inference API authentication. Admin tokens
are shared operator secrets, not per-user RBAC or an identity provider.

## Drain versus blacklist

| Operation | Scope | Heartbeat refresh | Deregistration/expiry/re-registration |
|---|---|---|---|
| Drain / resume | current registration's operator flag | cannot undo operator drain | registration-local flag disappears on removal/expiry |
| Block / unblock | instance ID, independent of registration | cannot undo the rule | blocked ID remains blocked, including changed model/role/target |

Neither operation aborts in-flight work. Blacklisting filters scheduling
snapshots **and** atomically rejects new reservations at the store. A request
with a reservation committed before the block may still dispatch/finish.
Unblocking does not clear operator drain, engine drain or health cooldown;
resuming does not clear a blacklist rule. A blocked engine may continue
heartbeating/reporting state, so operators can still inspect it.

- **Memory**: rules are local to the gateway process. They survive instance
  re-registration but not a gateway restart; other independent memory gateways
  are unaffected.
- **Redis**: rules are shared by all upgraded gateways using that namespace, and
  survive gateway restarts. Durability across Redis restarts/failover depends on
  its persistence/HA setup, just like other shared gateway state. Offline rules
  stay queryable and removable. Mutations fail closed with `503` during an outage,
  never silently applying only to local memory. A timeout can be ambiguous: read
  back the rule before retrying; PUT/DELETE/drain/resume are idempotent.

This requires Redis storage **schema 3**. Schema-1/2 gateways do not understand
blacklists and must not share serving traffic with this fleet. Perform a
coordinated cutover with a fresh namespace (for example `modelweave-v3`), stop the
old fleet, and let engines re-register. No automatic state/rule migration or
database flushing is performed. See [state and scheduling](state-and-scheduling.md).

Successful operations log structured `admin action` events with `action`
(`block`, `unblock`, `set_drain`), `worker`, `remote_addr`, and the reason/drain
value where applicable. Credentials are not logged. `remote_addr` is the socket
peer, not a verified individual operator identity; collect logs externally if
you need durable audit history. The console does not invent an audit database.

## HTTP API

All inventory endpoints are authenticated, read-only GETs. Lists are stable-sorted
by canonical model name or instance ID and return `[]`, not `null`, when empty.

| Method | Path | Result / operation |
|---|---|---|
| GET | `/admin/models` | `{"models":[...]}`: configuration, role counts, policy weights, queue/overload |
| GET | `/admin/models/{name}` | one model; accepts canonical name or alias |
| GET | `/admin/workers` | `{"workers":[...]}`: existing fields plus labels, capacity, block metadata, report status |
| GET | `/admin/workers/{id}` | one registered instance; offline/unknown IDs return `404` |
| GET | `/admin/stats` | existing statistics plus `ready`, `uptime_seconds`, `admin_auth_required` |
| GET | `/admin/scheduling/plugins` | scoring plugin metadata, available policy names, composition formula and loading/configuration mode |
| POST | `/admin/workers/{id}/drain` | set operator drain; legacy `?undo=1` still resumes |
| POST | `/admin/workers/{id}/resume` | clear operator drain only |
| GET | `/admin/blacklist` | `{"blacklist":[{"id":"...","reason":"...","blocked_at_ms":...}]}` |
| PUT | `/admin/blacklist/{id}` | create/update a rule, including offline IDs; JSON `{"reason":"..."}` |
| DELETE | `/admin/blacklist/{id}` | remove only that ID's rule; repeated calls succeed |

`GET /admin/workers` accepts optional query parameters:

- `model`: canonical name or alias (`404` if unknown).
- `role`: `encode`, `prefill`, `decode`, `unified`.
- `status`: `active`, `blocked`, `draining`, `unhealthy`. These exclusive display
  categories use precedence `blocked → draining → unhealthy → active`; the raw
  flags remain independent in responses.

Worker `state_status` is `fresh`, `stale` or `unknown`; expired reports are not
returned as trusted `state`. Model `context_length: 0` means use the tokenizer's
context window, not a zero-length window. Encode each path identifier as **one
URL component**, even if the ID contains `/`, spaces, `?` or `#`.

Block bodies must use `Content-Type: application/json`, contain only `reason`
(1–512 Unicode characters after trimming), and fit in 8 KiB. IDs accept 1–256
UTF-8 bytes and cannot be all whitespace. Repeating a PUT with the same reason
preserves `blocked_at_ms`; updating the reason updates the time. Redis uses its
own clock, memory uses the gateway clock.

Errors use the existing admin shape `{"type":"error","code":"...","message":"..."}`:
`400` invalid arguments/JSON, `401` unauthorized, `403` cross-origin mutation,
`404` unknown resource, `409` registration changed during mutation,
`413` oversized body, `415` wrong content type, `503` unavailable state backend.

```sh
# MW_ADMIN_TOKEN contains the configured operator secret; do not put it in URLs.
curl -H "X-ModelWeave-Token: $MW_ADMIN_TOKEN" http://localhost:8080/admin/models
curl -G -H "X-ModelWeave-Token: $MW_ADMIN_TOKEN" \
  --data-urlencode model=deepseek-v4.1 --data-urlencode role=decode \
  http://localhost:8080/admin/workers

curl -X PUT -H "X-ModelWeave-Token: $MW_ADMIN_TOKEN" \
  -H 'Content-Type: application/json' -d '{"reason":"repeated transport errors"}' \
  http://localhost:8080/admin/blacklist/decode-0
curl -X DELETE -H "X-ModelWeave-Token: $MW_ADMIN_TOKEN" \
  http://localhost:8080/admin/blacklist/decode-0
curl -X POST -H "X-ModelWeave-Token: $MW_ADMIN_TOKEN" \
  http://localhost:8080/admin/workers/decode-0/drain
curl -X POST -H "X-ModelWeave-Token: $MW_ADMIN_TOKEN" \
  http://localhost:8080/admin/workers/decode-0/resume
```

Prometheus adds `modelweave_worker_blocked`; `modelweave_worker_up` is zero for a
blocked instance. These are shared-state gauges in Redis mode, not values to sum
across gateway replicas. Offline rules have no per-worker time series.

## Development and verification

Assets live in `internal/gateway/web/`; rebuild the gateway after editing. The
console is native ES modules and CSS; no JavaScript dependency is shipped.

```sh
make test-go          # Go race tests: API/auth/CSRF, inference routing, memory/Redis contracts
make test-admin-ui    # Node 20+; pure view, validation, escaping and URL encoding tests

# Optional Chromium integration tests against an isolated real Go gateway:
npm ci --prefix test/admin-ui --ignore-scripts --no-audit --no-fund
export PLAYWRIGHT_BROWSERS_PATH="$PWD/tmp/playwright-browsers"
node test/admin-ui/node_modules/playwright/cli.js install chromium
make test-admin-browser
```

Browser tests cover login/expiry, alias search, composed instance filters,
confirmation/cancellation, drain/block/resume/unblock, offline rules, outage
recovery, XSS-safe rendering, scheduling plugin discovery/templates, per-model
weights and mobile overflow. They save desktop/mobile
screenshots under `tmp/admin-ui/` (override with `MW_ADMIN_ARTIFACTS`). They do not
use production services or GPUs. Normal Go tests skip this optional browser test.
