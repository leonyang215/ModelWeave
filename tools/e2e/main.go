// Command e2e prepares isolated test configs without modifying local-stack
// YAML or checking generated credentials/artifacts into the repository.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"

	"github.com/leonyang215/ModelWeave/internal/config"
)

func main() {
	dir := flag.String("output", "", "artifact/config output directory")
	backend := flag.String("backend", "memory", "memory or redis")
	flag.Parse()
	if *dir == "" || (*backend != "memory" && *backend != "redis") {
		fail(fmt.Errorf("output is required; backend must be memory or redis"))
	}
	if err := os.MkdirAll(*dir, 0755); err != nil {
		fail(err)
	}
	for _, name := range []string{"gateway", "gateway-events"} {
		data, err := os.ReadFile(filepath.Join("deploy", "compose", name+".yaml"))
		if err != nil {
			fail(err)
		}
		var document map[string]any
		if err := yaml.Unmarshal(data, &document); err != nil {
			fail(err)
		}
		if name == "gateway" {
			models, ok := document["models"].([]any)
			if !ok {
				fail(fmt.Errorf("%s: models is not a sequence", name))
			}
			document["models"] = append(models, map[string]any{
				"name": "e2e-weighted", "mode": "normal", "tokenizer_model": "Qwen3-8B",
				"routing": map[string]any{"default": map[string]any{"scorers": []any{
					map[string]any{"name": "label", "weight": 3, "params": map[string]any{"key": "pool", "value": "preferred"}},
					map[string]any{"name": "load", "weight": 1},
				}}},
			})
		}
		if *backend == "redis" {
			document["state"] = map[string]any{"backend": "redis", "redis": map[string]any{
				"url": "redis://redis:6379/0", "namespace": "modelweave-e2e-" + name,
			}}
		}
		data, err = yaml.Marshal(document)
		if err != nil {
			fail(err)
		}
		if _, err := config.Parse(data); err != nil {
			fail(fmt.Errorf("validate generated %s: %w", name, err))
		}
		if err := os.WriteFile(filepath.Join(*dir, name+".yaml"), data, 0644); err != nil {
			fail(err)
		}
	}
	fmt.Printf("prepared isolated %s e2e configs in %s\n", *backend, *dir)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "e2e config:", err)
	os.Exit(1)
}
