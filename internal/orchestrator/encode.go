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
// encode stage: under the model's policy, when encoders exist and the
// engine protocol encodes outside the engine. SGLang encodes inside the
// engine (a language-only prefill server dispatches media to its own
// encoders), so its requests carry the raw media instead.
func (o *Orchestrator) NeedsEncode(job *Job) (bool, error) {
	if job.Media == nil || len(job.Media.Items) == 0 || !engine.EncodeInGateway(job.Model.Protocol) {
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
// prefill/decode capacity is held while encoders work.
//
//   - mwip: every item whose artifacts are not cached is built on an encoder
//     (one artifact per placement: an image, or each kept frame of a video);
//     the artifacts are attached to the job's media for P/D.
//   - vllm: every image is sent to an EC-producer instance; the
//     ec_transfer_params it returns go to P/D, which load the embeddings
//     through their EC connector. Handles are not cached across requests.
func (o *Orchestrator) Encode(ctx context.Context, job *Job) (*EncodeStats, error) {
	start := time.Now()
	m := job.Model
	items := job.Media.Items
	st := &EncodeStats{Items: len(items)}
	cacheable := m.Protocol == mwip.EngineMWIP
	var todo []int
	seen := map[string]bool{}
	for i := range items {
		key := ArtifactKey(&items[i])
		if cacheable {
			if arts, ok := m.Artifacts.Get(key); ok {
				items[i].Embeddings = arts
				st.Hits++
				o.met.ArtifactLookup.WithLabelValues(m.Name, "hit").Inc()
				continue
			}
			o.met.ArtifactLookup.WithLabelValues(m.Name, "miss").Inc()
		}
		if !seen[key] {
			seen[key] = true
			todo = append(todo, i)
		}
	}
	var (
		mu       sync.Mutex
		firstErr error
		wg       sync.WaitGroup
		ec       = map[string]any{}
		built    = map[string][]mwip.EmbeddingArtifact{}
	)
	sem := make(chan struct{}, 4)
	ctx, cancel := context.WithDeadline(ctx, job.FirstTokenDeadline)
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
			if cacheable {
				m.Artifacts.Put(key, res.Artifacts)
			}
			built[key] = res.Artifacts
			for k, v := range res.ECParams {
				ec[k] = v
			}
			st.Built++
			st.Workers = append(st.Workers, worker)
		}(idx)
	}
	wg.Wait()
	if firstErr != nil {
		return st, firstErr
	}
	if ctx.Err() != nil && time.Now().After(job.FirstTokenDeadline) {
		return st, ErrFirstTokenTimeout
	}
	if m.Protocol == mwip.EngineVLLM {
		job.ECParams = ec
	} else {
		for i := range items {
			if items[i].Embeddings == nil {
				items[i].Embeddings = built[ArtifactKey(&items[i])]
			}
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
		o.met.EngineRequests.WithLabelValues(m.Name, string(registry.RoleEncode), m.Protocol).Inc()
		res, err := o.eng.Encode(ctx, m.Protocol, d.Spec, &engine.EncodeStage{
			RequestID: job.ID, AttemptID: fmt.Sprintf("%s-enc%d-%d", job.ID, it.Index, attempt),
			Model: m.Name, EngineModel: m.Cfg.EngineModel, Items: []mwip.MediaItem{*it},
			PlaceholderTokenID: job.Media.PlaceholderTokenID,
			TimeoutMS:          time.Until(job.FirstTokenDeadline).Milliseconds(),
		})
		if err == nil && m.Protocol == mwip.EngineMWIP {
			res.Artifacts, err = orderArtifacts(d.Spec.ID, it, res.Artifacts)
		}
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

// orderArtifacts returns one artifact per placement, in placement order.
func orderArtifacts(worker string, it *mwip.MediaItem, arts []mwip.EmbeddingArtifact) ([]mwip.EmbeddingArtifact, error) {
	by := make(map[string]mwip.EmbeddingArtifact, len(arts))
	for _, a := range arts {
		by[a.Reference] = a
	}
	out := make([]mwip.EmbeddingArtifact, 0, len(it.Placements))
	for _, p := range it.Placements {
		a, ok := by[p.Reference]
		if !ok {
			return nil, &engine.Error{Role: "encode", WorkerID: worker, Code: mwip.ErrInternal, Retriable: true,
				Message: fmt.Sprintf("encoder returned no artifact for %s", p.Reference)}
		}
		out = append(out, a)
	}
	return out, nil
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
		job.Media.Items[i].Embeddings = nil
	}
	job.ECParams = nil
	job.Model.Artifacts.Invalidate(keys...)
	o.met.ArtifactLookup.WithLabelValues(job.Model.Name, "invalidated").Add(float64(len(keys)))
}
