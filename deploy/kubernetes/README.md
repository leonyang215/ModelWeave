# Kubernetes deployment

Kustomize manifests for the gateway/admin console and the real Rust tokenizer.
ModelWeave is an **application-level inference scheduler**, not a replacement for
`kube-scheduler`: do not set a Pod's `schedulerName` to ModelWeave. No Kubernetes
API access, watches, leader election, CRDs or RBAC bindings are required.

## Layout

- `namespace.yaml`: production namespace and restricted Pod Security admission.
- `base/`: two gateway replicas, two tokenizer replicas, internal Services,
  ConfigMaps, PDBs and an unprivileged ServiceAccount; external Redis and secrets
  are intentionally required, not silently replaced with per-Pod memory.
- `overlays/demo/`: separate `modelweave-demo` namespace, known demo credentials,
  ephemeral Redis and mock encode/prefill/decode Deployments. No GPUs or real
  inference engines. Role overlays reuse `components/mockengine/`.

Container images default to the local build names `modelweave/{gateway,tokenizer,mockengine}:dev`.
**These are not claimed to be published images.** Build/push them to your registry
and override image names/tags (prefer immutable digests) before applying to a real
cluster. Configure imagePullSecrets in your site overlay if required.

## Production-oriented base

Requirements: Kubernetes with the standard `apps/v1`/`policy/v1` APIs, a container
registry accessible to the nodes, and a managed Redis 7+ primary endpoint. Redis
needs the script/data/PubSub permissions described in
[state and scheduling](../../docs/state-and-scheduling.md), HA/persistence for
durable state, and `noeviction`. All gateway replicas must use identical model,
tenant, heartbeat/block-size and queue settings. There is no Redis schema change
in this deployment addition; existing schema-3 rollout rules still apply.

1. Build and publish images under **your** registry. For example, with explicit
   `REGISTRY` and a release `TAG`, build each Docker target and push the result:

   ```sh
   docker build --target gateway -t "$REGISTRY/gateway:$TAG" .
   docker build --target tokenizer -t "$REGISTRY/tokenizer:$TAG" .
   docker push "$REGISTRY/gateway:$TAG"
   docker push "$REGISTRY/tokenizer:$TAG"
   ```

2. Create a site overlay referring to `../../base`, overriding the two images:

   ```yaml
   apiVersion: kustomize.config.k8s.io/v1beta1
   kind: Kustomization
   resources: [../../base]
   namespace: modelweave
   images:
     - name: modelweave/gateway
       newName: registry.example.com/your-team/gateway
       newTag: your-release-tag
     - name: modelweave/tokenizer
       newName: registry.example.com/your-team/tokenizer
       newTag: your-release-tag
   ```

   Save it as `deploy/kubernetes/overlays/site/kustomization.yaml` and replace
   example registry/tag values. Customize resources, model/tokenizer configs and
   tenant quotas for your workload; the shipped limits are starting points, not
   throughput guarantees. Model/tokenizer files beyond the bundled families need
   to be included in your tokenizer image or mounted at configured paths.

3. Copy `secret.env.example` **outside the repository**, replace every placeholder
   and `chmod 600` the result. Use distinct strong operator/engine/client secrets
   (hex/base64url values are convenient for YAML expansion), URL-encode Redis
   credentials and use TLS (`rediss://`) where appropriate. Never commit the real
   file or put secret values in command-line URLs/logs.

4. Select an explicit cluster context and create namespace/Secret before workloads:

   ```sh
   CTX=your-reviewed-context
   kubectl --context "$CTX" apply -f deploy/kubernetes/namespace.yaml
   kubectl --context "$CTX" -n modelweave create secret generic modelweave-secrets \
     --from-env-file=/secure/path/modelweave.env
   kubectl --context "$CTX" apply -k deploy/kubernetes/overlays/site
   kubectl --context "$CTX" -n modelweave rollout status deployment/tokenizer
   kubectl --context "$CTX" -n modelweave rollout status deployment/gateway
   ```

   If your platform manages secrets through an external-secret controller, create
   the same Secret/key contract there instead. Secret env updates require a
   gateway rollout/restart; they are not hot-reloaded. ConfigMap generators use
   hash suffixes so applied config changes update Pod templates automatically.

5. Integrate real engine Pods through [mwip](../../docs/mwip.md). Each instance
   serves one model/role at an individually reachable Engine endpoint and
   heartbeats to **`gateway:8081`** in the namespace (or
   `gateway.modelweave.svc:8081` across namespaces), with `MW_CONTROL_TOKEN` as
   `x-modelweave-token` metadata. Advertise a **unique instance ID and Pod-reachable
   target**, not a load-balanced Service representing several engine instances.
   E/P/D peer/artifact addresses must also be reachable. Redis shares registrations
   across gateway replicas, so heartbeat to one Registry Service is enough.

The gateway's state namespace includes the Pod namespace (`modelweave-<namespace>-v3`)
to avoid accidentally sharing one Redis namespace between installations. Preserve
or explicitly migrate this setting when moving an existing fleet.

## Security, networking and operations

- Services are **ClusterIP only**; no Ingress, LoadBalancer or host port is
  installed. HTTP `:8080` includes `/admin/` and `/admin/*`; do not expose that
  listener unfiltered to the Internet. Use your authenticated TLS proxy/Ingress
  with deliberate `/v1/*` versus admin routing. Configure SSE/body/timeouts there.
- Distinct `MW_ADMIN_TOKEN` and `MW_CONTROL_TOKEN` prevent engine credentials from
  authorizing operator actions. See [admin security](../../docs/admin.md).
- ClusterIP is not a security boundary. Apply your platform's NetworkPolicies,
  transport encryption/service mesh, secret rotation and network access controls.
  Internal HTTP/gRPC is not magically TLS-encrypted by these manifests.
- All workloads run non-root with dropped capabilities, no privilege escalation
  and no mounted ServiceAccount token. Config is mounted read-only. There is no
  need to grant permission to list Pods or modify cluster scheduling.
- Readiness checks shared-state availability; liveness checks process health.
  Gateway shutdown drains with 30 seconds grace inside a 60-second Pod termination
  budget. PDBs retain one replica during voluntary disruptions; they do not make a
  single-node cluster highly available. Gateway topology spreading is a preference.
- Tokenizer replicas are stateless. Its Service balances connections, not each
  HTTP/2 stream independently; use an appropriate L7/discovery arrangement if
  request-level balancing across many tokenizer replicas is required.
- `/metrics` exposes Prometheus metrics; scrape gateway replicas individually.
  Sum replica-local counters, not shared state/QoS gauges. No Prometheus Operator
  dependency is introduced.

For private access to the admin console:

```sh
kubectl --context "$CTX" -n modelweave port-forward --address 127.0.0.1 svc/gateway 8080:8080
# Open http://127.0.0.1:8080/admin/ and enter the configured admin token.
```

## GPU-free demo overlay

Build/publish the additional `mockengine` Docker target and use a site overlay over
`../demo` to override **all three** ModelWeave images. In a local cluster you can
instead load the locally built image tags using that cluster's supported tooling.
Then apply the demo overlay to an explicit, disposable context:

```sh
kubectl --context "$CTX" apply -k deploy/kubernetes/overlays/demo
kubectl --context "$CTX" -n modelweave-demo rollout status deployment/gateway
kubectl --context "$CTX" -n modelweave-demo get deployments,pods,services
kubectl --context "$CTX" -n modelweave-demo port-forward --address 127.0.0.1 svc/gateway 8080:8080
```

The demo uses `demo-api-key` for inference, `demo-admin-token` for admin and
`demo-control-token` for engine registration. It simulates **deepseek-v4.1**
Normal/PD/EPD via registered E/P/D instances. The other configured models need
their own engines before they can serve inference. Example:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Authorization: Bearer demo-api-key' -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1","messages":[{"role":"user","content":"Hello"}]}'
```

The demo is **not production**: known credentials, one ephemeral unauthenticated
Redis, no durable state or real model execution. Redis replacement loses quotas,
blacklists and metadata. Mock engine instances advertise their IPv4 Pod IP using
Downward API fields; adapt address formatting for an IPv6-only environment.

## Offline validation (no cluster test)

```sh
kustomize build deploy/kubernetes/base > /path/to/reviewed-base.yaml
kustomize build deploy/kubernetes/overlays/demo > /path/to/reviewed-demo.yaml
# kubectl kustomize <directory> is an alternative offline renderer.
KUSTOMIZE=/path/to/kustomize make test-deploy
```

Offline checks cover config parsing, selectors, generated references, namespace,
Secret separation, private Services and container safeguards. **No Kubernetes
cluster was created or tested.** Functional regression uses ModelWeave itself in
isolated Docker Compose stacks:

```sh
make test-e2e
make test-e2e-redis
```

These exercise the real tokenizer plus mock engines, including the full existing
protocol/orchestration suite and multi-gateway Redis behaviour. They do not imply
validation of your cluster's CNI, registry permissions, Ingress, Redis HA or actual
GPU engine images.
