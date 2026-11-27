// Package load rewards spare capacity in the role's normalized load unit.
package load

import "github.com/leonyang215/ModelWeave/pkg/scheduling"

type Scorer struct{}

func (Scorer) Name() string    { return "load" }
func (Scorer) UsesCache() bool { return false }
func (Scorer) Score(_ *scheduling.Request, c *scheduling.Candidate, _ []*scheduling.Candidate) float64 {
	return 1 - c.NormLoad
}

func init() {
	err := scheduling.Register(scheduling.Descriptor{
		Name: "load", Version: "1.0.0", Source: "plugins/scheduling/load",
		Description: "Prefer spare capacity; combines running requests, prefill backlog and KV utilization by role.",
	}, func(c scheduling.Config) (scheduling.Scorer, error) {
		if err := scheduling.DecodeParams(c.Params, &struct{}{}); err != nil {
			return nil, err
		}
		return Scorer{}, nil
	})
	if err != nil {
		panic(err)
	}
}
