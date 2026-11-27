package scheduler_test

import (
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// The framework/runtime layer must not install built-in algorithms or depend
// back on the application assembly. Keep this boundary enforceable, not just
// a directory diagram that can silently drift into an import cycle.
func TestFrameworkDependencyDirection(t *testing.T) {
	err := filepath.WalkDir("framework", func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
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
			if strings.Contains(name, "/internal/scheduler/plugins") || strings.HasSuffix(name, "/scheduler/profile") || strings.HasSuffix(name, "/scheduler/policy") || strings.HasSuffix(name, "/internal/config") || strings.HasSuffix(name, "/internal/gateway") {
				t.Errorf("framework dependency points upward: %s imports %s", path, name)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
