// Command modelweave-gateway runs the ModelWeave gateway and scheduler.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/leonyang215/ModelWeave/internal/config"
	"github.com/leonyang215/ModelWeave/internal/gateway"
	"github.com/leonyang215/ModelWeave/internal/obs"
)

var version = "dev"

func main() {
	cfgPath := flag.String("config", envOr("MW_GATEWAY_CONFIG", "configs/gateway.yaml"), "path of the gateway YAML config")
	engineProtocol := flag.String("engine-protocol", envOr("MW_ENGINE_PROTOCOL", ""),
		"engine protocol of workers: mwip (send the intermediate state), sglang or vllm; overrides engine.protocol")
	showVersion := flag.Bool("version", false, "print the version and exit")
	flag.Parse()
	if *showVersion {
		fmt.Println(version)
		return
	}
	cfg, err := config.LoadWith(*cfgPath, config.Overrides{EngineProtocol: *engineProtocol})
	if err != nil {
		fmt.Fprintf(os.Stderr, "modelweave-gateway: %v\n", err)
		os.Exit(1)
	}
	log := obs.NewLogger(cfg.Server.LogLevel, cfg.Server.LogFormat)
	log.Info("engine protocol", "default", cfg.Engine.Protocol, "flag", *engineProtocol)
	srv, err := gateway.New(cfg, log, version)
	if err != nil {
		log.Error("startup failed", "error", err)
		os.Exit(1)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := srv.Run(ctx); err != nil {
		log.Error("server error", "error", err)
		os.Exit(1)
	}
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
