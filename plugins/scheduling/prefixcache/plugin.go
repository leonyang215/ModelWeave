// Package prefixcache rewards the fraction of the prompt already cached.
package prefixcache

import "github.com/leonyang215/ModelWeave/pkg/scheduling"

type Scorer struct{}

func (Scorer) Name() string    { return "prefix-cache" }
func (Scorer) UsesCache() bool { return true }
func (Scorer) Score(_ *scheduling.Request, c *scheduling.Candidate, _ []*scheduling.Candidate) float64 {
	return c.CacheRatio
}

func init() {
	err := scheduling.Register(scheduling.Descriptor{
		Name: "prefix-cache", Version: "1.0.0", Source: "plugins/scheduling/prefixcache",
		Description: "Prefer instances holding a larger cached fraction of the prompt.",
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
