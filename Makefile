# ModelWeave build and test entry points.
GO      ?= go
CARGO   ?= cargo
COMPOSE ?= docker compose -f deploy/compose/docker-compose.yaml
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)

.PHONY: all build build-go build-tokenizer test test-go test-rust lint images up down e2e e2e-host clean

all: build

build: build-go build-tokenizer

build-go:
	$(GO) build -trimpath -ldflags "-X main.version=$(VERSION)" -o bin/ ./cmd/...

build-tokenizer:
	cd tokenizer && $(CARGO) build --release --locked
	install -D tokenizer/target/release/modelweave-tokenizer bin/modelweave-tokenizer

test: test-go test-rust

test-go:
	$(GO) vet ./...
	$(GO) test -race -count=1 ./internal/...

test-rust:
	cd tokenizer && $(CARGO) test --locked

lint:
	gofmt -l cmd internal test | (! grep .)
	cd tokenizer && $(CARGO) fmt --check && $(CARGO) clippy --locked --all-targets -- -D warnings

# Container images (BuildKit; pass proxies through when needed).
images:
	for t in tokenizer gateway mockengine e2e; do \
	  docker build --build-arg VERSION=$(VERSION) \
	    --build-arg http_proxy=$(http_proxy) --build-arg https_proxy=$(https_proxy) --build-arg no_proxy=$(no_proxy) \
	    --target $$t -t modelweave/$$t:dev . || exit 1; \
	done

up:
	$(COMPOSE) up -d --no-build

down:
	$(COMPOSE) --profile monitoring --profile test down --remove-orphans

# End-to-end tests inside the compose network.
e2e:
	$(COMPOSE) --profile test run --rm --no-deps e2e

# End-to-end tests from the host against the published ports.
e2e-host:
	$(GO) test -tags e2e -count=1 -v ./test/e2e

clean:
	rm -rf bin
