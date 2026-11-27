package plugins

import (
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler/framework"
)

type Capacity struct{}

func (Capacity) Name() string { return "Capacity" }
func (Capacity) Filter(req *framework.Request, c *framework.Candidate) framework.Status {
	s := &c.Snap
	full := func(reason string) framework.Status {
		return framework.Status{Code: framework.Unschedulable, Reason: reason}
	}
	if s.MaxRunning > 0 && s.Running() >= int64(s.MaxRunning) {
		return full("running request capacity")
	}
	switch req.Role {
	case registry.RoleDecode, registry.RoleUnified:
		if s.KVCapacity > 0 && s.KVFree() < req.KVReserve {
			return full("KV capacity")
		}
		if req.Role == registry.RoleUnified || req.RequireNormal {
			if s.MaxPrefillTokens > 0 && s.QueuedPrefillTokens() >= s.MaxPrefillTokens {
				return full("prefill capacity")
			}
		}
	case registry.RolePrefill:
		if s.MaxPrefillTokens > 0 && s.QueuedPrefillTokens() >= s.MaxPrefillTokens {
			return full("prefill capacity")
		}
	}
	return framework.Status{}
}
