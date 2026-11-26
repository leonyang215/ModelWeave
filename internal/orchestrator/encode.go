package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/leonyang215/ModelWeave/internal/engine"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

// ErrEncodeUnavailable means media must be encoded but no encoder exists.
var ErrEncodeUnavailable = errors.New("no encode worker available")

// NeedsEncode reports whether the job's media go through the gateway's
// encode stage: under the model's policy, when encoders exist.
func (o *Orchestrator) NeedsEncode(job *Job) (bool, error) {
	if job.Media == nil || len(job.Media.Items) == 0 {
		return false, nil
	}
	switch job.Model.Cfg.Encode.Policy {
	case "never":
		return false, nil
	case "always":
		if !o.has(job.Model.Name, registry.RoleEncode, false) {
			return false, ErrEncodeUnavailable
		}
		return true, nil
	default: // auto: encode when an encoder pool exists, else let P/D do it
		return o.has(job.Model.Name, registry.RoleEncode, false), nil
	}
}

// EncodeStats summarizes an encode stage.
type EncodeStats struct {
	Items    int
	Hits     int
	Built    int
	Workers  []string
	Duration time.Duration
}

// Encode runs the encode stage before P/D reservations are taken, so that no
// prefill/decode capacity is held while encoders work: every item whose
// embedding is not cached is built on an encode instance with exactly its
// frozen plan. One packed location per item is then forwarded to both P/D.
func (o *Orchestrator) Encode(ctx context.Context, job *Job) (*EncodeStats, error) {
	start := time.Now()
	m := job.Model
	items := job.Media.Items
	st := &EncodeStats{Items: len(items)}
	if err := mwip.ValidateMedia(job.TokenIDs(), job.Media); err != nil {
		return st, fmt.Errorf("invalid media execution contract: %w", err)
	}
	var todo []int
	seen := map[string]bool{}
	for i := range items {
		key := ArtifactKey(&items[i])
		if embedding, ok := m.Artifacts.Get(key); ok {
			if err := items[i].Input().ValidateEmbedding(&embedding); err == nil {
				items[i].Embedding = &embedding
				st.Hits++
				o.met.ArtifactLookup.WithLabelValues(m.Name, "hit").Inc()
				continue
			}
			m.Artifacts.Invalidate(key)
		}
		o.met.ArtifactLookup.WithLabelValues(m.Name, "miss").Inc()
		if !seen[key] {
			seen[key] = true
			todo = append(todo, i)
		}
	}
	var (
		mu       sync.Mutex
		firstErr error
		wg       sync.WaitGroup
		built    = map[string]mwip.EmbeddingLocation{}
	)
	sem := make(chan struct{}, 4)
	parent := ctx
	ctx, cancel := context.WithDeadline(parent, job.FirstTokenDeadline)
	defer cancel()
	for _, idx := range todo {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				return
			}
			defer func() { <-sem }()
			res, worker, err := o.encodeItem(ctx, job, &items[idx])
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				if firstErr == nil {
					firstErr = err
					cancel()
				}
				return
			}
			key := ArtifactKey(&items[idx])
			m.Artifacts.Put(key, *res.Embedding)
			built[key] = *res.Embedding
			st.Built++
			st.Workers = append(st.Workers, worker)
		}(idx)
	}
	wg.Wait()
	if firstErr != nil {
		return st, firstErr
	}
	if ctx.Err() != nil {
		if parent.Err() != nil {
			return st, parent.Err()
		}
		return st, ErrFirstTokenTimeout
	}
	for i := range items {
		if items[i].Embedding == nil {
			e, ok := built[ArtifactKey(&items[i])]
			if !ok {
				return st, fmt.Errorf("encoder produced no item embedding")
			}
			items[i].Embedding = &e
		}
	}
	st.Duration = time.Since(start)
	o.met.StageDuration.WithLabelValues(m.Name, "encode").Observe(st.Duration.Seconds())
	return st, nil
}

// encodeItem encodes one item, retrying on other encoders for retriable
// errors. An outcome-unknown failure (connection broke after submission) is
// also retried: artifacts are content addressed, so a duplicate build is
// harmless.
func (o *Orchestrator) encodeItem(ctx context.Context, job *Job, it *mwip.MediaItem) (*engine.EncodeResult, string, error) {
	m := job.Model
	exclude := map[string]bool{}
	var lastErr error
	for attempt := 1; attempt <= m.Cfg.Retry.StageAttempts; attempt++ {
		req := job.request(registry.RoleEncode)
		req.Exclude = exclude
		req.Blocks = nil
		d, err := m.Encode.Schedule(ctx, req)
		if err != nil {
			if lastErr != nil {
				return nil, "", fmt.Errorf("%w (after %v)", err, lastErr)
			}
			return nil, "", err
		}
		o.met.EngineRequests.WithLabelValues(m.Name, string(registry.RoleEncode)).Inc()
		input := it.Input()
		res, err := o.eng.Encode(ctx, d.Spec, &engine.EncodeStage{
			RequestID: job.ID, AttemptID: fmt.Sprintf("%s-enc%d-%d", job.ID, it.Index, attempt),
			Model: m.Name, EngineModel: m.Cfg.EngineModel, Item: &input,
			TimeoutMS: time.Until(job.FirstTokenDeadline).Milliseconds(),
		})
		d.Reservation.Release(err == nil)
		if err == nil {
			d.Worker.ReportSuccess()
			return res, d.Spec.ID, nil
		}
		lastErr = err
		var e *engine.Error
		if !errors.As(err, &e) {
			return nil, "", err
		}
		o.met.UpstreamErrors.WithLabelValues(m.Name, "encode", e.Code).Inc()
		if countsAgainstHealth(e) {
			d.Worker.ReportFailure(time.Now(), o.health.FailureThreshold, o.health.Cooldown.D())
		}
		if !e.Retriable {
			return nil, "", err
		}
		exclude[d.Spec.ID] = true
		o.met.Retries.WithLabelValues(m.Name, "encode", e.Code).Inc()
	}
	return nil, "", lastErr
}

// InvalidateArtifacts drops the job's artifacts from the cache and the job
// (an engine reported them unavailable) so a rebuild happens.
func (o *Orchestrator) InvalidateArtifacts(job *Job) {
	if job.Media == nil {
		return
	}
	var keys []string
	for i := range job.Media.Items {
		keys = append(keys, ArtifactKey(&job.Media.Items[i]))
		job.Media.Items[i].Embedding = nil
	}
	job.Model.Artifacts.Invalidate(keys...)
	o.met.ArtifactLookup.WithLabelValues(job.Model.Name, "invalidated").Add(float64(len(keys)))
}
