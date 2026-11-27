// Package affinity supplies a composable rendezvous-hash session preference.
package affinity

import (
	"github.com/zeebo/xxh3"

	"github.com/leonyang215/ModelWeave/pkg/scheduling"
)

type Scorer struct{}

func (Scorer) Name() string    { return "affinity" }
func (Scorer) UsesCache() bool { return false }
func (Scorer) Score(req *scheduling.Request, c *scheduling.Candidate, cs []*scheduling.Candidate) float64 {
	if req.SessionKey == "" {
		return 0
	}
	weight := xxh3.HashString(req.SessionKey + "\x00" + c.ID)
	for _, other := range cs {
		if xxh3.HashString(req.SessionKey+"\x00"+other.ID) > weight {
			return 0
		}
	}
	return 1
}

func init() {
	err := scheduling.Register(scheduling.Descriptor{
		Name: "affinity", Version: "1.0.0", Source: "plugins/scheduling/affinity",
		Description: "Prefer a stable rendezvous-hashed instance for the request's session key; zero without a session key.",
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
