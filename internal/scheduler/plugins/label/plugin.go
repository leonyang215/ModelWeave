// Package label adds a soft preference for an instance label, such as zone
// or accelerator class. It never excludes otherwise eligible instances.
package label

import (
	"fmt"
	"strings"

	"github.com/leonyang215/ModelWeave/internal/scheduling"
)

type Scorer struct{ key, value string }

func (Scorer) Name() string    { return "label" }
func (Scorer) UsesCache() bool { return false }
func (s Scorer) Score(_ *scheduling.Request, c *scheduling.Candidate, _ []*scheduling.Candidate) float64 {
	if value, ok := c.Labels[s.key]; ok && value == s.value {
		return 1
	}
	return 0
}

func init() {
	err := scheduling.Register(scheduling.Descriptor{
		Name: "label", Version: "1.0.0", Source: "internal/scheduler/plugins/label",
		Description: "Soft preference for an instance label, e.g. zone or GPU class; non-matching instances remain eligible.",
		Parameters: []scheduling.Parameter{
			{Name: "key", Type: "string", Required: true, Description: "Instance label key."},
			{Name: "value", Type: "string", Required: true, Description: "Preferred label value (exact match)."},
		},
	}, func(c scheduling.Config) (scheduling.Scorer, error) {
		var p struct {
			Key   string  `json:"key"`
			Value *string `json:"value"`
		}
		if err := scheduling.DecodeParams(c.Params, &p); err != nil {
			return nil, err
		}
		if strings.TrimSpace(p.Key) == "" || p.Value == nil {
			return nil, fmt.Errorf("params.key (non-empty) and params.value are required")
		}
		return Scorer{key: p.Key, value: *p.Value}, nil
	})
	if err != nil {
		panic(err)
	}
}
