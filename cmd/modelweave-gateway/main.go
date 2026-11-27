// Command modelweave-gateway runs the ModelWeave gateway and scheduler. It
// serves the OpenAI and Anthropic APIs over HTTP and the mwip Registry, where
// engine instances register, over gRPC.
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
	"github.com/leonyang215/ModelWeave/internal/scheduler"
)

var version = "dev"

func main() {
	cfgPath := flag.String("config", envOr("MW_GATEWAY_CONFIG", "configs/gateway.yaml"), "path of the gateway YAML config")
	showVersion := flag.Bool("version", false, "print the version and exit")
	checkConfig := flag.Bool("check-config", false, "validate configuration and scheduling plugins without connecting to backends")
	flag.Parse()
	if *showVersion {
		fmt.Println(version)
		return
	}
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "modelweave-gateway: %v\n", err)
		os.Exit(1)
	}
	if *checkConfig {
		fmt.Printf("configuration valid: %d models, %d scoring plugins\n", len(cfg.Models), len(scheduler.RegisteredScorers()))
		return
	}
	log := obs.NewLogger(cfg.Server.LogLevel, cfg.Server.LogFormat)
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
