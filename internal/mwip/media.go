package mwip

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strings"
)

const PadShift uint32 = 1 << 30

// MediaInput is independent of the prompt's absolute token positions. A
// cached item embedding can therefore be reused at a different prompt offset.
type MediaInput struct {
	Modality string      `json:"modality"`
	Digest   string      `json:"digest"`
	Source   MediaSource `json:"source"`
	Plan     MediaPlan   `json:"plan"`
	PlanID   string      `json:"plan_id"`
}

// EmbeddingLocation points at one item's packed visual tensor. Rows follow
// plan order; placements map slices of these rows to absolute prompt ranges.
type EmbeddingLocation struct {
	URI     string `json:"uri"`
	Tokens  int    `json:"tokens"`
	PlanID  string `json:"plan_id"`
	Encoder string `json:"encoder,omitempty"`
}

func (it *MediaItem) Input() MediaInput {
	return MediaInput{Modality: it.Modality, Digest: it.Digest, Source: it.Source, Plan: it.Plan, PlanID: it.PlanID}
}

func (p MediaPlan) Groups() int {
	if len(p.FrameIndices) == 0 {
		return 1
	}
	if p.TemporalPatchSize <= 0 {
		return 0
	}
	return len(p.FrameIndices) / p.TemporalPatchSize
}
func (p MediaPlan) EmbeddingTokens() int { return p.Tokens * p.Groups() }

// PlanID is a language-neutral fingerprint shared with the Rust tokenizer.
// Source URLs and prompt offsets are deliberately excluded: content+plan,
// not transport credentials or conversation position, identify an embedding.
func PlanID(modality, digest string, p MediaPlan) string {
	h := sha256.New()
	h.Write([]byte("mwip-media-plan-v2\x00"))
	var b [8]byte
	u64 := func(n uint64) { binary.LittleEndian.PutUint64(b[:], n); h.Write(b[:]) }
	str := func(s string) { u64(uint64(len(s))); h.Write([]byte(s)) }
	str(modality)
	str(digest)
	str(p.Processor)
	for _, n := range []int{p.Width, p.Height, p.Tokens, p.NFrames, p.PatchSize, p.SpatialMergeSize, p.TemporalPatchSize} {
		u64(uint64(n))
	}
	u64(uint64(len(p.FrameIndices)))
	for _, n := range p.FrameIndices {
		u64(n)
	}
	u64(uint64(len(p.Timestamps)))
	for _, n := range p.Timestamps {
		u64(math.Float64bits(n))
	}
	return hex.EncodeToString(h.Sum(nil))
}

func PlacementReference(planID, modality string, group int) string {
	if modality == "image" {
		return planID
	}
	return fmt.Sprintf("%s_kf_%04d", planID, group)
}
func PadValue(reference string) uint32 {
	h := sha256.Sum256([]byte(reference))
	return PadShift + uint32(binary.BigEndian.Uint64(h[:8])&((1<<30)-1))
}

func (p MediaPlan) Validate(modality string) error {
	if p.Width <= 0 || p.Height <= 0 || p.Tokens <= 0 || p.NFrames <= 0 || p.Width > math.MaxInt32 || p.Height > math.MaxInt32 || p.Tokens > math.MaxInt32 || p.NFrames > math.MaxInt32 {
		return fmt.Errorf("plan dimensions/tokens/nframes must be positive int32 values")
	}
	if p.PatchSize <= 0 || p.PatchSize > 1024 || p.SpatialMergeSize <= 0 || p.SpatialMergeSize > 64 || p.TemporalPatchSize <= 0 || p.TemporalPatchSize > 32 {
		return fmt.Errorf("plan patch/merge/temporal sizes are invalid")
	}
	factor := int64(p.PatchSize) * int64(p.SpatialMergeSize)
	var expected int64
	switch p.Processor {
	case "qwen_vl.v1":
		if int64(p.Width)%factor != 0 || int64(p.Height)%factor != 0 {
			return fmt.Errorf("Qwen plan dimensions must be multiples of patch*merge")
		}
		expected = (int64(p.Width) / factor) * (int64(p.Height) / factor)
	case "deepseek_v4_1.v1":
		if p.PatchSize != 14 || p.SpatialMergeSize != 3 {
			return fmt.Errorf("DeepSeek V4.1 plan must use patch=14, merge=3")
		}
		w := (int64(p.Width) + factor - 1) / factor
		h := (int64(p.Height) + factor - 1) / factor
		expected = h*(w+1) + 2
	default:
		return fmt.Errorf("unknown authoritative processor %q", p.Processor)
	}
	if expected != int64(p.Tokens) {
		return fmt.Errorf("plan tokens=%d, processor geometry requires %d", p.Tokens, expected)
	}
	switch modality {
	case "image":
		if p.NFrames != 1 || len(p.FrameIndices) != 0 || len(p.Timestamps) != 0 {
			return fmt.Errorf("image plan must declare nframes=1 and no video sampling")
		}
	case "video":
		if len(p.FrameIndices) != p.NFrames || p.NFrames%p.TemporalPatchSize != 0 || len(p.Timestamps) != p.Groups() {
			return fmt.Errorf("video plan requires nframes=len(frame_indices), explicit temporal padding, and one timestamp per group")
		}
		for i, index := range p.FrameIndices {
			if index > math.MaxInt64 || (i > 0 && index < p.FrameIndices[i-1]) {
				return fmt.Errorf("video frame indices must be ordered non-negative int64 values")
			}
		}
		for i, ts := range p.Timestamps {
			if math.IsNaN(ts) || math.IsInf(ts, 0) || ts < 0 || (i > 0 && ts < p.Timestamps[i-1]) {
				return fmt.Errorf("video timestamps must be finite, non-negative and ordered")
			}
		}
	default:
		return fmt.Errorf("unsupported media modality %q", modality)
	}
	if int64(p.Tokens)*int64(p.Groups()) > math.MaxInt32 {
		return fmt.Errorf("item embedding exceeds int32 row count")
	}
	return nil
}

func (in MediaInput) Validate() error {
	if len(in.Digest) != 64 || strings.ToLower(in.Digest) != in.Digest {
		return fmt.Errorf("media digest must be lowercase SHA-256 hex")
	}
	if _, err := hex.DecodeString(in.Digest); err != nil {
		return fmt.Errorf("media digest must be SHA-256 hex")
	}
	if in.Source.URL == "" || (in.Source.Kind != "url" && in.Source.Kind != "base64") {
		return fmt.Errorf("media source must be a URL or base64 data URL")
	}
	if err := in.Plan.Validate(in.Modality); err != nil {
		return err
	}
	if in.PlanID != PlanID(in.Modality, in.Digest, in.Plan) {
		return fmt.Errorf("media plan_id does not match content and execution plan")
	}
	return nil
}
func (in MediaInput) ValidateEmbedding(e *EmbeddingLocation) error {
	if e == nil || strings.TrimSpace(e.URI) == "" {
		return fmt.Errorf("encoder returned no embedding location")
	}
	if e.PlanID != in.PlanID {
		return fmt.Errorf("embedding plan_id does not match the execution plan")
	}
	if e.Tokens != in.Plan.EmbeddingTokens() {
		return fmt.Errorf("embedding has %d rows, plan requires %d", e.Tokens, in.Plan.EmbeddingTokens())
	}
	return nil
}

// ValidateMedia proves the scheduler's pad ranges, the encoder's tensor rows
// and the P/D scatter map describe the same execution. No hints/re-planning
// or collapsed-placeholder interpretation is allowed on the engine path.
func ValidateMedia(ids []uint32, m *Multimodal) error {
	if m == nil {
		for _, id := range ids {
			if id >= PadShift {
				return fmt.Errorf("media pads require a multimodal execution plan")
			}
		}
		return nil
	}
	if m.PadShift != PadShift {
		return fmt.Errorf("invalid media pad_shift")
	}
	var ranges []Span
	for i, it := range m.Items {
		if it.Index != i {
			return fmt.Errorf("media item index %d does not match position %d", it.Index, i)
		}
		in := it.Input()
		if err := in.Validate(); err != nil {
			return fmt.Errorf("media item %d: %w", i, err)
		}
		if len(it.Placements) != it.Plan.Groups() {
			return fmt.Errorf("media item %d: placement count does not match plan groups", i)
		}
		for k, p := range it.Placements {
			ref := PlacementReference(it.PlanID, it.Modality, k)
			if p.Length != it.Plan.Tokens || p.EmbeddingOffset != k*it.Plan.Tokens {
				return fmt.Errorf("media item %d: token/tensor slice differs from plan", i)
			}
			if p.Reference != ref || p.PadValue != PadValue(ref) {
				return fmt.Errorf("media item %d: placement reference/pad is not bound to the plan", i)
			}
			if it.Modality == "image" {
				if p.Modality != "image" || p.Timestamp != nil {
					return fmt.Errorf("image placement has video fields")
				}
			} else if p.Modality != "frame" || p.Timestamp == nil || *p.Timestamp != it.Plan.Timestamps[k] {
				return fmt.Errorf("frame placement differs from plan timestamp")
			}
			if p.Offset < 0 || p.Offset > len(ids) || p.Length > len(ids)-p.Offset {
				return fmt.Errorf("media item %d: placement is outside token_ids", i)
			}
			for _, id := range ids[p.Offset : p.Offset+p.Length] {
				if id != p.PadValue {
					return fmt.Errorf("media item %d: token_ids do not match the placement pad", i)
				}
			}
			ranges = append(ranges, Span{p.Offset, p.Length})
		}
		if it.Embedding != nil {
			if err := in.ValidateEmbedding(it.Embedding); err != nil {
				return fmt.Errorf("media item %d: %w", i, err)
			}
		}
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].Offset < ranges[j].Offset })
	end := 0
	for _, r := range ranges {
		if r.Offset < end {
			return fmt.Errorf("media placements overlap")
		}
		for _, id := range ids[end:r.Offset] {
			if id >= PadShift {
				return fmt.Errorf("unmapped media pad in token_ids")
			}
		}
		end = r.Offset + r.Length
	}
	for _, id := range ids[end:] {
		if id >= PadShift {
			return fmt.Errorf("unmapped media pad in token_ids")
		}
	}
	return nil
}
