// Package runtime executes Filter and Score extension points and constructs
// policies from registries. It does not install any in-tree algorithms.
package runtime

import (
	"errors"

	"github.com/leonyang215/ModelWeave/internal/scheduler/framework"
)

type Framework struct {
	policy  framework.Policy
	filters []framework.FilterPlugin
}

// New assembles a per-model/role runtime. The core supplies mandatory filters;
// scoring plugins receive only the candidates that pass all of them.
func New(policy framework.Policy, filters ...framework.FilterPlugin) *Framework {
	return &Framework{policy: policy, filters: append([]framework.FilterPlugin{}, filters...)}
}

func (f *Framework) RunFilterPlugins(req *framework.Request, c *framework.Candidate) framework.Status {
	for _, plugin := range f.filters {
		if status := plugin.Filter(req, c); status.Code != framework.Success {
			return status
		}
	}
	return framework.Status{}
}

func (f *Framework) RunScorePlugins(req *framework.Request, candidates []*framework.Candidate) (*framework.Candidate, string, error) {
	if len(candidates) == 0 {
		return nil, "", errors.New("cannot score an empty candidate list")
	}
	pick, reason := f.policy.Pick(req, candidates)
	for _, candidate := range candidates {
		if pick != nil && candidate == pick {
			return pick, reason, nil
		}
	}
	return nil, "", errors.New("scheduling policy returned an ineligible candidate")
}
