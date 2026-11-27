package framework

// Code distinguishes filters that eliminate an instance from those that
// find a healthy instance temporarily full. This preserves the queue's
// ErrNoWorkers versus ErrNoCapacity behaviour.
type Code int

const (
	Success Code = iota
	Unschedulable
	UnschedulableAndUnresolvable
)

type Status struct {
	Code   Code
	Reason string
}

// FilterPlugin is an internal extension point. Mandatory admission filters
// are assembled by the scheduler, never disabled by a scoring configuration.
type FilterPlugin interface {
	Name() string
	Filter(*Request, *Candidate) Status
}
