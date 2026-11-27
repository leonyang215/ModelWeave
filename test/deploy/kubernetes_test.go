// Package deploy performs offline rendering/configuration checks only. It
// neither reads a cluster kubeconfig nor calls a Kubernetes API server.
package deploy

import (
	"bytes"
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/leonyang215/ModelWeave/internal/config"
	"gopkg.in/yaml.v3"
)

func TestGatewayDeploymentConfig(t *testing.T) {
	setConfigEnv(t)
	c, err := config.Load("../../deploy/kubernetes/base/config/gateway.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if c.State.Backend != "redis" || c.Server.AdminToken == "" || c.Server.AdminToken == c.Server.ControlToken || !c.Auth.Enabled || len(c.Models) != 3 {
		t.Fatal("HA deployment must have shared state, separate admin auth and model config")
	}
}

func setConfigEnv(t *testing.T) {
	t.Helper()
	for key, value := range map[string]string{
		"MW_REDIS_URL": "redis://127.0.0.1:1/0", "MW_STATE_NAMESPACE": "offline-v3",
		"MW_CONTROL_TOKEN": "test-engine", "MW_ADMIN_TOKEN": "test-operator", "MW_API_KEY": "test-client",
	} {
		t.Setenv(key, value)
	}
}

func at(v any, keys ...string) any {
	for _, key := range keys {
		m, ok := v.(map[string]any)
		if !ok {
			return nil
		}
		v = m[key]
	}
	return v
}
func mapping(v any) map[string]any { m, _ := v.(map[string]any); return m }
func sequence(v any) []any         { s, _ := v.([]any); return s }
func text(v any) string            { s, _ := v.(string); return s }
func matches(selector, labels map[string]any) bool {
	if len(selector) == 0 {
		return false
	}
	for key, value := range selector {
		if labels[key] != value {
			return false
		}
	}
	return true
}

func TestKustomizeOfflineReferencesAndSecurity(t *testing.T) {
	tool := os.Getenv("KUSTOMIZE")
	if tool == "" {
		var err error
		tool, err = exec.LookPath("kustomize")
		if err != nil {
			t.Skip("set KUSTOMIZE to run optional offline manifest rendering")
		}
	}
	setConfigEnv(t)
	for _, profile := range []struct {
		path, namespace string
		deployments     int
	}{
		{"base", "modelweave", 2}, {"overlays/demo", "modelweave-demo", 6},
	} {
		t.Run(profile.path, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, tool, "build", filepath.Join("../../deploy/kubernetes", profile.path))
			cmd.Env = append(os.Environ(), "KUBECONFIG=/dev/null")
			data, err := cmd.Output()
			if err != nil {
				t.Fatalf("offline rendering: %v", err)
			}
			var resources []map[string]any
			decoder := yaml.NewDecoder(bytes.NewReader(data))
			for {
				var resource map[string]any
				err := decoder.Decode(&resource)
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatal(err)
				}
				if len(resource) > 0 {
					resources = append(resources, resource)
				}
			}
			known := make(map[string]map[string]any)
			var deployments, services []map[string]any
			for _, resource := range resources {
				kind, name := text(resource["kind"]), text(at(resource, "metadata", "name"))
				key := kind + "/" + name
				if kind == "" || name == "" || resource["apiVersion"] == nil || known[key] != nil {
					t.Fatalf("invalid/duplicate resource: %s", key)
				}
				known[key] = resource
				if kind != "Namespace" && at(resource, "metadata", "namespace") != profile.namespace {
					t.Errorf("wrong namespace: %s", key)
				}
				switch kind {
				case "Deployment":
					deployments = append(deployments, resource)
				case "Service":
					services = append(services, resource)
				case "ConfigMap":
					if value := at(resource, "data", "gateway.yaml"); value != nil {
						if _, err := config.Parse([]byte(text(value))); err != nil {
							t.Fatalf("rendered gateway config: %v", err)
						}
					}
				case "Secret":
					if profile.path == "base" {
						t.Fatal("production base must not ship credentials")
					}
				}
			}
			if len(deployments) != profile.deployments {
				t.Fatalf("deployments=%d want=%d", len(deployments), profile.deployments)
			}
			for _, deployment := range deployments {
				name := text(at(deployment, "metadata", "name"))
				pod := mapping(at(deployment, "spec", "template", "spec"))
				labels := mapping(at(deployment, "spec", "template", "metadata", "labels"))
				selector := mapping(at(deployment, "spec", "selector", "matchLabels"))
				if !matches(selector, labels) {
					t.Errorf("selector mismatch: %s", name)
				}
				for _, other := range deployments {
					if other != nil && at(other, "metadata", "name") != name && matches(selector, mapping(at(other, "spec", "template", "metadata", "labels"))) {
						t.Errorf("overlapping Deployment selectors: %s", name)
					}
				}
				if pod["automountServiceAccountToken"] != false || at(pod, "securityContext", "runAsNonRoot") != true || known["ServiceAccount/"+text(pod["serviceAccountName"])] == nil {
					t.Errorf("unsafe service account/security context: %s", name)
				}
				for _, volume := range sequence(pod["volumes"]) {
					if cm := text(at(volume, "configMap", "name")); cm != "" && known["ConfigMap/"+cm] == nil {
						t.Errorf("missing ConfigMap %s in %s", cm, name)
					}
				}
				for _, container := range sequence(pod["containers"]) {
					if at(container, "securityContext", "readOnlyRootFilesystem") != true || at(container, "securityContext", "allowPrivilegeEscalation") != false || at(container, "readinessProbe") == nil || at(container, "livenessProbe") == nil || at(container, "resources", "requests") == nil || text(at(container, "image")) == "" {
						t.Errorf("incomplete container safeguards: %s", name)
					}
					for _, env := range sequence(at(container, "env")) {
						secret := text(at(env, "valueFrom", "secretKeyRef", "name"))
						if secret == "" || (profile.path == "base" && secret == "modelweave-secrets") {
							continue
						}
						if known["Secret/"+secret] == nil {
							t.Errorf("missing Secret %s in %s", secret, name)
						}
					}
				}
			}
			for _, service := range services {
				if at(service, "spec", "type") != "ClusterIP" {
					t.Error("unexpected external service exposure")
				}
				found := false
				for _, deployment := range deployments {
					found = found || matches(mapping(at(service, "spec", "selector")), mapping(at(deployment, "spec", "template", "metadata", "labels")))
				}
				if !found {
					t.Errorf("Service %s has no matching Deployment", at(service, "metadata", "name"))
				}
			}
			t.Logf("offline %s: %d resources, %d Deployments; references/selectors/config/security verified", profile.path, len(resources), len(deployments))
		})
	}
}
