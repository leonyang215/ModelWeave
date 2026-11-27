package scheduling_test

import (
	"context"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestRepositoryPrivateLayout(t *testing.T) {
	root := filepath.Join("..", "..")
	for _, name := range []string{"pkg", "plugins"} {
		if _, err := os.Stat(filepath.Join(root, name)); !os.IsNotExist(err) {
			t.Errorf("implementation directory must live below internal, not %s", name)
		}
	}
	for _, dir := range []string{"api", "cmd", "internal", "tools", "test"} {
		err := filepath.WalkDir(filepath.Join(root, dir), func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				if entry.Name() == "node_modules" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") {
				return nil
			}
			file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
			if err != nil {
				return err
			}
			for _, spec := range file.Imports {
				name, err := strconv.Unquote(spec.Path.Value)
				if err != nil {
					return err
				}
				if strings.HasPrefix(name, "github.com/leonyang215/ModelWeave/pkg/") || strings.HasPrefix(name, "github.com/leonyang215/ModelWeave/plugins/") {
					t.Errorf("obsolete implementation import in %s: %s", path, name)
				}
				if dir == "api" && strings.HasPrefix(name, "github.com/leonyang215/ModelWeave/internal/") {
					t.Errorf("public engine protocol depends on private implementation: %s", name)
				}
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
}

// Verify the compiler boundary from a genuinely separate module, not merely
// by checking a directory name. No network or remote dependencies are needed.
func TestExternalModuleCannotImportPrivateScoringContract(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	module := "module example.com/modelweave-consumer\n\ngo 1.26.0\n\nrequire github.com/leonyang215/ModelWeave v0.0.0\nreplace github.com/leonyang215/ModelWeave => " + strconv.Quote(root) + "\n"
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(module), 0600); err != nil {
		t.Fatal(err)
	}
	program := "package consumer\nimport _ \"github.com/leonyang215/ModelWeave/internal/scheduling\"\n"
	if err := os.WriteFile(filepath.Join(dir, "consumer.go"), []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, filepath.Join(runtime.GOROOT(), "bin", "go"), "test", "-mod=mod", ".")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off", "GOPROXY=off", "GOSUMDB=off", "GOTOOLCHAIN=local")
	output, err := cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(output), "use of internal package") || !strings.Contains(string(output), "not allowed") {
		t.Fatalf("expected compiler-enforced internal boundary; err=%v output=%s", err, output)
	}
	t.Log("external module rejected by Go's internal-package import rule")
}
