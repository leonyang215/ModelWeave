// Package plugins implements the mandatory internal Filter extension points.
// These safety checks are separate from configurable weighted scoring plugins.
package plugins

import "github.com/leonyang215/ModelWeave/internal/scheduler/framework"

type Eligibility struct{}

func (Eligibility) Name() string { return "Eligibility" }
func (Eligibility) Filter(req *framework.Request, c *framework.Candidate) framework.Status {
	s := &c.Snap
	if req.Exclude[s.Spec.ID] || (req.Pin != "" && req.Pin != s.Spec.ID) {
		return framework.Status{Code: framework.UnschedulableAndUnresolvable, Reason: "excluded or not pinned instance"}
	}
	if s.Gone || s.Spec.Model != req.Model || s.Spec.Role != req.Role || !s.Healthy || s.Draining || s.Blocked {
		return framework.Status{Code: framework.UnschedulableAndUnresolvable, Reason: "instance unavailable"}
	}
	if req.RequireNormal && !s.Spec.ServesNormal() {
		return framework.Status{Code: framework.UnschedulableAndUnresolvable, Reason: "normal execution unsupported"}
	}
	return framework.Status{}
}
