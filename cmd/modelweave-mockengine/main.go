// Command modelweave-mockengine runs a simulated inference worker for local
// and CI testing. It serves every engine protocol the gateway can be
// started with: the native intermediate state (/mwip/v1/generate), vLLM's
// token-in generate (/inference/v1/generate) and SGLang's /generate. Every
// flag can also be set with an MOCK_* variable (for example --decode-tps /
// MOCK_DECODE_TPS).
package main

import (
	"context"
	"flag"
	"fmt"
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
	flag.StringVar(&c.ID, "id", env("id", "mock-0"), "worker ID")
	flag.StringVar(&c.Model, "model", env("model", "deepseek-v4.1"), "served model name")
	flag.StringVar(&c.Role, "role", env("role", "unified"), "encode, prefill, decode or unified")
	flag.StringVar(&c.Listen, "listen", env("listen", ":9000"), "listen address")
	flag.StringVar(&c.AdvertiseURL, "advertise-url", env("advertise-url", "http://127.0.0.1:9000"), "URL other components use to reach this worker")
	flag.StringVar(&c.GatewayURL, "gateway-url", env("gateway-url", ""), "gateway URL for heartbeats (empty: static registration)")
	flag.StringVar(&c.ControlToken, "control-token", env("control-token", ""), "X-ModelWeave-Token for the gateway control plane")
	flag.StringVar(&c.TokenizerURL, "tokenizer-url", env("tokenizer-url", ""), "tokenizer URL used to produce output tokens")
	flag.StringVar(&c.TokenizerModel, "tokenizer-model", env("tokenizer-model", ""), "model name known to the tokenizer (default: its only model)")
	flag.DurationVar(&c.HeartbeatInterval, "heartbeat-interval", envDur("heartbeat-interval", time.Second), "heartbeat period")
	flag.BoolVar(&c.Normal, "normal", envBool("normal", false), "decode worker also runs normal (non-PD) requests")
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
	c.Log = obs.NewLogger(*logLevel, "json")
	switch c.Role {
	case "encode", "prefill", "decode", "unified":
	default:
		fmt.Fprintf(os.Stderr, "invalid role %q\n", c.Role)
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := mockengine.New(c).Run(ctx); err != nil {
		c.Log.Error("mock engine stopped", "error", err)
		os.Exit(1)
	}
}
