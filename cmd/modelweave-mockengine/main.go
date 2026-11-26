// Command modelweave-mockengine runs a simulated inference engine for local
// and CI testing: one instance (one model in one role) that implements the
// mwip Engine service (gRPC) and registers with the gateways' mwip Registry.
// Every flag can also be set with an MOCK_* variable (for example
// --decode-tps / MOCK_DECODE_TPS).
package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/leonyang215/ModelWeave/internal/mockengine"
	"github.com/leonyang215/ModelWeave/internal/obs"
)

func env(name, def string) string {
	if v, ok := os.LookupEnv("MOCK_" + strings.ToUpper(strings.ReplaceAll(name, "-", "_"))); ok {
		return v
	}
	return def
}

func envInt(name string, def int) int {
	v, err := strconv.Atoi(env(name, strconv.Itoa(def)))
	if err != nil {
		return def
	}
	return v
}

func envFloat(name string, def float64) float64 {
	v, err := strconv.ParseFloat(env(name, strconv.FormatFloat(def, 'f', -1, 64)), 64)
	if err != nil {
		return def
	}
	return v
}

func envBool(name string, def bool) bool {
	v, err := strconv.ParseBool(env(name, strconv.FormatBool(def)))
	if err != nil {
		return def
	}
	return v
}

func envDur(name string, def time.Duration) time.Duration {
	v, err := time.ParseDuration(env(name, def.String()))
	if err != nil {
		return def
	}
	return v
}

func main() {
	var c mockengine.Config
	var sc mockengine.ServeConfig
	flag.StringVar(&c.ID, "id", env("id", "mock-0"), "instance ID")
	flag.StringVar(&c.Model, "model", env("model", "deepseek-v4.1"), "served model name")
	flag.StringVar(&c.Role, "role", env("role", "unified"), "encode, prefill, decode or unified")
	flag.StringVar(&sc.Listen, "listen", env("listen", ":9000"), "HTTP listen address (KV hand-off, artifacts, metrics, registration controls)")
	flag.StringVar(&sc.GRPCListen, "grpc-listen", env("grpc-listen", ":9001"), "gRPC listen address (the mwip Engine service)")
	flag.StringVar(&c.AdvertiseURL, "advertise-url", env("advertise-url", "http://127.0.0.1:9000"), "HTTP URL other mock instances use to reach this one")
	flag.StringVar(&sc.Target, "target", env("target", ""), "mwip target registered for the instance (default: the advertise URL's host on the gRPC port)")
	registry := flag.String("registry", env("registry", ""), "gateways' mwip Registry targets, comma separated (empty: configured statically)")
	flag.StringVar(&sc.ControlToken, "control-token", env("control-token", ""), "x-modelweave-token of Registry calls")
	flag.DurationVar(&sc.HeartbeatInterval, "heartbeat-interval", envDur("heartbeat-interval", time.Second), "heartbeat period")
	flag.StringVar(&c.TokenizerURL, "tokenizer-url", env("tokenizer-url", ""), "tokenizer URL used to produce output tokens")
	flag.StringVar(&c.TokenizerModel, "tokenizer-model", env("tokenizer-model", ""), "model name known to the tokenizer (default: its only model)")
	flag.BoolVar(&c.Normal, "normal", envBool("normal", false), "decode instance also runs normal (non-PD) requests")
	flag.Int64Var(&c.KVCapacityTokens, "kv-capacity-tokens", int64(envInt("kv-capacity-tokens", 262144)), "KV cache capacity in tokens")
	flag.IntVar(&c.MaxRunning, "max-running", envInt("max-running", 64), "maximum concurrent requests")
	flag.Int64Var(&c.MaxPrefillTokens, "max-prefill-tokens", int64(envInt("max-prefill-tokens", 0)), "advertised prefill queue limit (0: unlimited)")
	flag.Float64Var(&c.PrefillTPS, "prefill-tps", envFloat("prefill-tps", 50000), "simulated prefill throughput (tokens/s)")
	flag.Float64Var(&c.DecodeTPS, "decode-tps", envFloat("decode-tps", 200), "simulated decode speed per request (tokens/s)")
	flag.IntVar(&c.MaxOutputTokens, "max-output-tokens", envInt("max-output-tokens", 256), "cap on generated tokens")
	flag.IntVar(&c.AnswerRepeat, "answer-repeat", envInt("answer-repeat", 0), "filler sentences appended to answers")
	flag.IntVar(&c.BlockSize, "block-size", envInt("block-size", 64), "KV block size in tokens (must match the gateway)")
	flag.BoolVar(&c.KVEvents, "kv-events", envBool("kv-events", false), "publish KV cache events with heartbeats")
	flag.DurationVar(&c.EncodeLatency, "encode-latency", envDur("encode-latency", 30*time.Millisecond), "simulated encode time per media item")
	flag.BoolVar(&c.RejectNormal, "reject-normal", envBool("reject-normal", false), "reject normal requests (fault injection)")
	flag.Float64Var(&c.FailRate, "fail-rate", envFloat("fail-rate", 0), "probability of failing a request before output")
	flag.IntVar(&c.FailAfterTokens, "fail-after-tokens", envInt("fail-after-tokens", 0), "break streams after N tokens")
	flag.BoolVar(&c.VerifyArtifacts, "verify-artifacts", envBool("verify-artifacts", true), "check embedding artifacts with their encoder")
	logLevel := flag.String("log-level", env("log-level", "info"), "log level")
	flag.Parse()
	switch c.Role {
	case "encode", "prefill", "decode", "unified":
	default:
		fail("invalid role %q: want encode, prefill, decode or unified", c.Role)
	}
	c.Log = obs.NewLogger(*logLevel, "json")
	sc.Log = c.Log
	for _, t := range strings.Split(*registry, ",") {
		if t = strings.TrimSpace(t); t != "" {
			sc.Registry = append(sc.Registry, t)
		}
	}
	if sc.Target == "" {
		sc.Target = grpcTarget(c.AdvertiseURL, sc.GRPCListen)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := mockengine.NewServer(sc, mockengine.New(c)).Run(ctx); err != nil {
		c.Log.Error("mock engine stopped", "error", err)
		os.Exit(1)
	}
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(2)
}

// grpcTarget is the advertise URL's host on the gRPC listen port.
func grpcTarget(advertise, listen string) string {
	host := "127.0.0.1"
	if u, err := url.Parse(advertise); err == nil && u.Hostname() != "" {
		host = u.Hostname()
	}
	_, port, err := net.SplitHostPort(listen)
	if err != nil {
		port = "9001"
	}
	return net.JoinHostPort(host, port)
}
