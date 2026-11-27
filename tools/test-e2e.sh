#!/usr/bin/env bash
# GPU-free, full-stack tests. Never operates on the default compose project.
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
backend="${MW_E2E_BACKEND:-memory}"
case "$backend" in memory|redis) ;; *) echo "MW_E2E_BACKEND must be memory or redis" >&2; exit 2 ;; esac
project="${MW_E2E_PROJECT:-modelweave-e2e-$(id -un)-$backend-$(date +%s)}"
if [[ ! "$project" =~ ^modelweave-e2e-[a-z0-9_-]+$ ]]; then
  echo "refusing non-test or unsafe compose project: $project" >&2
  exit 2
fi
if [[ -n "$(docker ps -aq --filter "label=com.docker.compose.project=$project")" ]]; then
  echo "project already has containers; choose a fresh MW_E2E_PROJECT: $project" >&2
  exit 2
fi
tag="${MW_E2E_IMAGE_TAG:-e2e-$(git rev-parse --short HEAD)-$(date +%s)}"
export MW_GATEWAY_IMAGE="modelweave/gateway:$tag"
export MW_MOCKENGINE_IMAGE="modelweave/mockengine:$tag"
export MW_TOKENIZER_IMAGE="modelweave/tokenizer:$tag"
export MW_E2E_IMAGE="modelweave/e2e:$tag"
export MW_CONTROL_TOKEN="${MW_E2E_CONTROL_TOKEN:-local-control-token}"
export MW_E2E_CONFIG_DIR="$root/tmp/$project"
export MW_E2E_GATEWAY_REPLICA=""
mkdir -p "$MW_E2E_CONFIG_DIR"
"${GO:-go}" run ./tools/e2e -backend "$backend" -output "$MW_E2E_CONFIG_DIR"
if [[ "${MW_E2E_BUILD:-1}" == 1 ]]; then
  for target in gateway tokenizer mockengine e2e; do
    docker build --builder "${MW_DOCKER_BUILDER:-default}" --target "$target" \
      --build-arg "http_proxy=${http_proxy:-${HTTP_PROXY:-}}" \
      --build-arg "https_proxy=${https_proxy:-${HTTPS_PROXY:-}}" \
      --build-arg "no_proxy=${no_proxy:-${NO_PROXY:-}}" \
      --build-arg VERSION="e2e-$(git rev-parse --short HEAD)" \
      -t "modelweave/$target:$tag" .
  done
fi
compose=(docker compose --env-file "$MW_E2E_CONFIG_DIR/compose.env" -p "$project" -f deploy/compose/docker-compose.yaml -f deploy/compose/e2e.override.yaml)
if [[ "$backend" == redis ]]; then
  compose+=(--profile redis)
  export MW_E2E_GATEWAY_REPLICA=http://gateway-replica:8080
fi
# Explicit interpolation input also works with Docker wrappers that sanitize
# the client environment. Only isolated test variables are written, never the
# caller's unrelated environment or production credentials.
(
  umask 077
  for key in MW_GATEWAY_IMAGE MW_MOCKENGINE_IMAGE MW_TOKENIZER_IMAGE MW_E2E_IMAGE MW_CONTROL_TOKEN MW_E2E_CONFIG_DIR MW_E2E_GATEWAY_REPLICA; do
    printf '%s=%s\n' "$key" "${!key}"
  done >"$MW_E2E_CONFIG_DIR/compose.env"
  chmod 600 "$MW_E2E_CONFIG_DIR/compose.env"
)
cleanup() {
  status=$?
  trap - EXIT
  "${compose[@]}" ps -a >"$MW_E2E_CONFIG_DIR/containers.txt" 2>&1 || true
  "${compose[@]}" logs --no-color >"$MW_E2E_CONFIG_DIR/services.log" 2>&1 || true
  if [[ "${MW_E2E_KEEP:-0}" != 1 ]]; then
    "${compose[@]}" --profile test down --volumes --remove-orphans || true
  fi
  echo "e2e artifacts: $MW_E2E_CONFIG_DIR"
  exit "$status"
}
trap cleanup EXIT
"${compose[@]}" up -d --no-build --wait --wait-timeout 180
"${compose[@]}" --profile test run --rm --no-deps e2e 2>&1 | tee "$MW_E2E_CONFIG_DIR/e2e.log"
