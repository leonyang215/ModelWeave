package mwip

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

// These fixtures are rendered by Rust, including its content+plan hash and
// pad ranges. This test is intentionally cross-language, not Go self-checking.
func mediaFixture(t *testing.T) *RenderResult {
	t.Helper()
	b, err := os.ReadFile("testdata/qwen3vl_media_json.json")
	if err != nil {
		t.Fatal(err)
	}
	var rr RenderResult
	if err := json.Unmarshal(b, &rr); err != nil {
		t.Fatal(err)
	}
	return &rr
}

func TestFrozenPlansAgreeWithRustAndMapPackedRows(t *testing.T) {
	r := mediaFixture(t)
	if err := ValidateMedia(r.TokenIDs, r.Multimodal); err != nil {
		t.Fatal(err)
	}
	v := r.Multimodal.Items[1]
	if !reflect.DeepEqual(v.Plan.FrameIndices, []uint64{0, 20, 39, 59}) || v.Plan.NFrames != 4 || v.Plan.TemporalPatchSize != 2 || v.Plan.Groups() != 2 {
		t.Fatalf("exact temporal execution plan: %+v", v.Plan)
	}
	for _, it := range r.Multimodal.Items {
		if it.PlanID != PlanID(it.Modality, it.Digest, it.Plan) {
			t.Fatal("Rust and Go disagree on plan identity")
		}
		for k, p := range it.Placements {
			if p.EmbeddingOffset != k*it.Plan.Tokens {
				t.Fatal("tensor row slice is not explicit")
			}
		}
	}
}

func TestPlanIdentityCoversExecutionButNotTransportOrPromptPosition(t *testing.T) {
	r := mediaFixture(t)
	in := r.Multimodal.Items[1].Input()
	original := in.PlanID
	in.Source.URL = "https://rotated.example/video?signature=new"
	if err := in.Validate(); err != nil {
		t.Fatalf("rotating source URL changed content identity: %v", err)
	}
	for name, change := range map[string]func(*MediaInput){
		"frames": func(in *MediaInput) {
			in.Plan.FrameIndices = append([]uint64(nil), in.Plan.FrameIndices...)
			in.Plan.FrameIndices[1] = 19
		},
		"timestamps": func(in *MediaInput) { in.Plan.Timestamps = []float64{0.5} },
		"resize":     func(in *MediaInput) { in.Plan.Width = 640; in.Plan.Height = 96 },
		"processor":  func(in *MediaInput) { in.Plan.PatchSize = 8; in.Plan.SpatialMergeSize = 4 },
		"digest":     func(in *MediaInput) { in.Digest = strings.Repeat("f", 64) },
	} {
		t.Run(name, func(t *testing.T) {
			copy := in
			change(&copy)
			if PlanID(copy.Modality, copy.Digest, copy.Plan) == original {
				t.Fatal("different execution reused the old hash")
			}
			if err := copy.Validate(); err == nil {
				t.Fatal("changed plan retained a valid old plan_id")
			}
		})
	}
}

func TestRejectInconsistentPlansPlacementsAndLocations(t *testing.T) {
	for name, mut := range map[string]func(*RenderResult){
		"plan id":                    func(r *RenderResult) { r.Multimodal.Items[0].PlanID = "wrong" },
		"unknown processor":          func(r *RenderResult) { r.Multimodal.Items[0].Plan.Processor = "some-default" },
		"wrong token geometry":       func(r *RenderResult) { r.Multimodal.Items[0].Plan.Tokens++ },
		"missing nframes":            func(r *RenderResult) { r.Multimodal.Items[1].Plan.NFrames = 0 },
		"nframes/index disagreement": func(r *RenderResult) { r.Multimodal.Items[1].Plan.NFrames = 2 },
		"image frame count":          func(r *RenderResult) { r.Multimodal.Items[0].Plan.NFrames = 2 },
		"changed sampling":           func(r *RenderResult) { r.Multimodal.Items[1].Plan.FrameIndices[0] = 1 },
		"frame count":                func(r *RenderResult) { r.Multimodal.Items[1].Plan.FrameIndices = []uint64{0} },
		"timestamp":                  func(r *RenderResult) { r.Multimodal.Items[1].Placements[0].Timestamp = new(float64) },
		"tensor offset":              func(r *RenderResult) { r.Multimodal.Items[1].Placements[0].EmbeddingOffset = 1 },
		"wrong pad":                  func(r *RenderResult) { p := r.Multimodal.Items[0].Placements[0]; r.TokenIDs[p.Offset]++ },
		"unmapped pad":               func(r *RenderResult) { r.TokenIDs[0] = PadShift },
		"missing mapping":            func(r *RenderResult) { r.Multimodal.Items[1].Placements = nil },
		"outside prompt":             func(r *RenderResult) { r.Multimodal.Items[0].Placements[0].Offset = len(r.TokenIDs) },
		"empty URI": func(r *RenderResult) {
			it := &r.Multimodal.Items[0]
			it.Embedding = &EmbeddingLocation{PlanID: it.PlanID, Tokens: it.Plan.EmbeddingTokens()}
		},
		"embedding rows": func(r *RenderResult) {
			it := &r.Multimodal.Items[0]
			it.Embedding = &EmbeddingLocation{URI: "mock://e/item", PlanID: it.PlanID, Tokens: it.Plan.EmbeddingTokens() + 1}
		},
		"embedding plan": func(r *RenderResult) {
			it := &r.Multimodal.Items[0]
			it.Embedding = &EmbeddingLocation{URI: "mock://e/item", PlanID: "other", Tokens: it.Plan.EmbeddingTokens()}
		},
	} {
		t.Run(name, func(t *testing.T) {
			r := mediaFixture(t)
			mut(r)
			if err := ValidateMedia(r.TokenIDs, r.Multimodal); err == nil {
				t.Fatal("inconsistent execution was accepted")
			}
		})
	}
}
