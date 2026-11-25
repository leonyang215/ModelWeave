package qos

import (
	"math/rand/v2"
	"sync"
)

// Action is the overload decision for a request.
type Action int

const (
	Admit Action = iota
	Degrade
	Reject
)

// matrix[level][priority]: load level L0..L4 against priority classes
// (critical, high, normal, low). Degrade means probabilistic dropping.
var matrix = [5][numPriorities]Action{
	{Admit, Admit, Admit, Admit},
	{Admit, Admit, Admit, Degrade},
	{Admit, Admit, Degrade, Reject},
	{Admit, Degrade, Reject, Reject},
	{Degrade, Reject, Reject, Reject},
}

// Overload is a utilization-level load shedder: pool utilization maps to
// levels L0..L4 with hysteresis, and a priority matrix
// decides per request whether to admit, drop probabilistically or reject.
// Requests whose prefix is mostly cached are cheap and exempt from drops.
type Overload struct {
	mu          sync.Mutex
	thresholds  [4]float64
	hysteresis  float64
	cacheExempt float64
	level       int
	util        float64
	rand        func() float64
}

// NewOverload returns a controller. thresholds are the utilizations at
// which levels L1..L4 start.
func NewOverload(thresholds []float64, hysteresis, cacheExempt float64) *Overload {
	o := &Overload{hysteresis: hysteresis, cacheExempt: cacheExempt, rand: rand.Float64}
	copy(o.thresholds[:], thresholds)
	return o
}

// Update feeds the current utilization and returns the new level. Levels
// rise immediately and fall only once utilization is hysteresis below the
// current level's threshold.
func (o *Overload) Update(util float64) int {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.util = util
	for o.level < 4 && util >= o.thresholds[o.level] {
		o.level++
	}
	for o.level > 0 && util < o.thresholds[o.level-1]-o.hysteresis {
		o.level--
	}
	return o.level
}

// Level returns the current level and utilization.
func (o *Overload) Level() (int, float64) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.level, o.util
}

// Decide returns whether to admit a request and why.
func (o *Overload) Decide(p Priority, cacheRatio float64) (bool, string) {
	if p < 0 || p >= numPriorities {
		p = PriorityNormal
	}
	o.mu.Lock()
	level, util := o.level, o.util
	o.mu.Unlock()
	switch matrix[level][p] {
	case Admit:
		return true, "admit"
	case Reject:
		return false, "overload_reject"
	}
	if cacheRatio >= o.cacheExempt {
		return true, "cache_exempt"
	}
	// Drop with a probability growing from 0 at the level's threshold to 1
	// at full utilization.
	thr := o.thresholds[level-1]
	prob := 1.0
	if thr < 1 {
		prob = (util - thr) / (1 - thr)
	}
	if prob < 0 {
		prob = 0
	}
	if o.rand() < prob {
		return false, "overload_drop"
	}
	return true, "degrade_pass"
}
