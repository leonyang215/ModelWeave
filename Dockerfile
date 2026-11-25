# Multi-target build:
#   docker build --target gateway    -t modelweave/gateway .
#   docker build --target tokenizer  -t modelweave/tokenizer .
#   docker build --target mockengine -t modelweave/mockengine .
ARG GO_VERSION=1.27
ARG RUST_VERSION=1
ARG DEBIAN_VERSION=bookworm

FROM golang:${GO_VERSION}-${DEBIAN_VERSION} AS go-build
WORKDIR /src
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY cmd ./cmd
COPY internal ./internal
ARG VERSION=dev
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /out/ ./cmd/...

FROM rust:${RUST_VERSION}-${DEBIAN_VERSION} AS rust-build
WORKDIR /src/tokenizer
COPY tokenizer/Cargo.toml tokenizer/Cargo.lock ./
COPY tokenizer/src ./src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/src/tokenizer/target \
    cargo build --release --locked && install -D target/release/modelweave-tokenizer /out/modelweave-tokenizer

FROM debian:${DEBIAN_VERSION}-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /nonexistent modelweave
USER 10001

FROM runtime AS gateway
COPY --from=go-build /out/modelweave-gateway /usr/local/bin/modelweave-gateway
COPY configs/gateway.example.yaml /etc/modelweave/gateway.yaml
EXPOSE 8080
ENTRYPOINT ["modelweave-gateway"]
CMD ["-config", "/etc/modelweave/gateway.yaml"]

FROM runtime AS mockengine
COPY --from=go-build /out/modelweave-mockengine /usr/local/bin/modelweave-mockengine
EXPOSE 9000
ENTRYPOINT ["modelweave-mockengine"]

# Serves every bundled model (configs/tokenizer.example.toml): DeepSeek V4.1,
# Qwen3 and Qwen3-VL. Mount another TOML file at
# /etc/modelweave/tokenizer.toml (and its model files) to register others.
FROM runtime AS tokenizer
COPY --from=rust-build /out/modelweave-tokenizer /usr/local/bin/modelweave-tokenizer
WORKDIR /opt/modelweave
COPY assets/tokenizers ./assets/tokenizers
COPY configs/tokenizer.example.toml /etc/modelweave/tokenizer.toml
ENV MW_TOKENIZER_CONFIG=/etc/modelweave/tokenizer.toml \
    MW_TOKENIZER_LISTEN=0.0.0.0:8090
EXPOSE 8090
ENTRYPOINT ["modelweave-tokenizer"]

# End-to-end test runner (docker compose --profile test run --rm e2e).
FROM go-build AS e2e-build
COPY test ./test
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go test -c -tags e2e -o /out/e2e.test ./test/e2e

FROM runtime AS e2e
COPY --from=e2e-build /out/e2e.test /usr/local/bin/e2e.test
ENTRYPOINT ["e2e.test", "-test.v", "-test.count=1", "-test.timeout=10m"]
