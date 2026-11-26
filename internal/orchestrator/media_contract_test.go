package orchestrator_test

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/leonyang215/ModelWeave/internal/config"
	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/orchestrator"
)

func (e *env) mediaJob() *orchestrator.Job {
	e.t.Helper()
	b, err := os.ReadFile("../mwip/testdata/qwen3vl_media_json.json")
	if err != nil {
		e.t.Fatal(err)
	}
	var rr mwip.RenderResult
	if err := json.Unmarshal(b, &rr); err != nil {
		e.t.Fatal(err)
	}
	j := e.job(4, false)
	j.Render = &rr
	j.Media = rr.Multimodal.Clone()
	j.Blocks = kvindex.BlockHashes(rr.TokenIDs, j.Model.BlockSize, j.Model.Seed)
	return j
}

func TestFrozenSingleItemPlanAndLocationsReachBothPDLegs(t *testing.T) {
	e := newEnv(t, func(m *config.Model) { m.Mode = "pd" })
	e.worker("e0", "encode", nil)
	e.worker("p0", "prefill", nil)
	e.worker("d0", "decode", nil)
	j := e.mediaJob()
	plans := []mwip.MediaPlan{j.Media.Items[0].Plan, j.Media.Items[1].Plan}
	st, err := e.orch.Encode(context.Background(), j)
	if err != nil || st.Built != 2 {
		t.Fatalf("encode: %+v %v", st, err)
	}
	for i, it := range j.Media.Items {
		if !reflect.DeepEqual(it.Plan, plans[i]) {
			t.Fatal("gateway altered tokenizer's execution plan")
		}
		if err := it.Input().ValidateEmbedding(it.Embedding); err != nil {
			t.Fatal(err)
		}
	}
	last := e.stats("e0")["last_encode"].(map[string]any)
	if _, batch := last["items"]; batch {
		t.Fatal("encoder still receives batches")
	}
	input := last["item"].(map[string]any)
	for _, legacy := range []string{"metadata", "pixel_control_options", "frame_extract_options", "placements"} {
		if _, ok := input[legacy]; ok {
			t.Fatalf("encoder can re-plan using %s", legacy)
		}
	}
	s := &sink{}
	out := e.run(j, s)
	expectOK(t, out, s, orchestrator.ModePD)
	for _, worker := range []string{"p0", "d0"} {
		last := lastRequest(e.stats(worker))
		mm := last["multimodal"].(map[string]any)
		got := mm["items"].([]any)
		for i, raw := range got {
			item := raw.(map[string]any)
			want := j.Media.Items[i]
			emb := item["embedding"].(map[string]any)
			if item["plan_id"] != want.PlanID || emb["plan_id"] != want.PlanID || emb["uri"] != want.Embedding.URI || emb["tokens"] != float64(want.Plan.EmbeddingTokens()) {
				t.Fatalf("%s lost item embedding identity: %+v", worker, item)
			}
			for k, raw := range item["placements"].([]any) {
				p := raw.(map[string]any)
				if p["offset"] != float64(want.Placements[k].Offset) || p["embedding_offset"] != float64(k*want.Plan.Tokens) || p["length"] != float64(want.Plan.Tokens) {
					t.Fatalf("%s scatter map differs: %+v", worker, p)
				}
			}
		}
	}
	if n := num(e.stats("p0"), "artifacts_verified"); n != 2 {
		t.Fatalf("P did not load/verify both item tensors: %v", n)
	}
}

func TestEmbeddingCacheRebindsPromptPositionsButNotExecutionPlans(t *testing.T) {
	e := newEnv(t, func(m *config.Model) { m.Mode = "pd" })
	e.worker("e0", "encode", nil)
	e.worker("p0", "prefill", nil)
	e.worker("d0", "decode", nil)
	first := e.mediaJob()
	if _, err := e.orch.Encode(context.Background(), first); err != nil {
		t.Fatal(err)
	}
	second := e.mediaJob()
	const shift = 17
	prefix := make([]uint32, shift)
	for i := range prefix {
		prefix[i] = 123
	}
	second.Render.TokenIDs = append(prefix, second.Render.TokenIDs...)
	second.Render.PromptTokens = len(second.Render.TokenIDs)
	for i := range second.Media.Items {
		second.Media.Items[i].Source.URL += "?new-signature"
		for k := range second.Media.Items[i].Placements {
			second.Media.Items[i].Placements[k].Offset += shift
		}
	}
	second.Blocks = kvindex.BlockHashes(second.TokenIDs(), second.Model.BlockSize, second.Model.Seed)
	st, err := e.orch.Encode(context.Background(), second)
	if err != nil || st.Hits != 2 || st.Built != 0 {
		t.Fatalf("same plan at new prompt offset missed cache: %+v %v", st, err)
	}
	for i, it := range second.Media.Items {
		if *it.Embedding != *first.Media.Items[i].Embedding {
			t.Fatal("cached tensor was rebound to absolute prompt offsets")
		}
	}
	s := &sink{}
	expectOK(t, e.run(second, s), s, orchestrator.ModePD)
	// Same dimensions and output count, DIFFERENT frame selection. The old
	// width/height/count cache key would have incorrectly reused this tensor.
	third := e.mediaJob()
	video := &third.Media.Items[1]
	video.Plan.FrameIndices = []uint64{0, 20, 38, 59}
	video.Plan.Timestamps = []float64{20.0 / 30 / 2, (38.0 + 59) / 30 / 2}
	video.PlanID = mwip.PlanID(video.Modality, video.Digest, video.Plan)
	for k := range video.Placements {
		p := &video.Placements[k]
		p.Reference = mwip.PlacementReference(video.PlanID, "video", k)
		p.PadValue = mwip.PadValue(p.Reference)
		p.Timestamp = &video.Plan.Timestamps[k]
		for n := p.Offset; n < p.Offset+p.Length; n++ {
			third.Render.TokenIDs[n] = p.PadValue
		}
	}
	st, err = e.orch.Encode(context.Background(), third)
	if err != nil || st.Hits != 1 || st.Built != 1 {
		t.Fatalf("different sampling reused old embedding: %+v %v", st, err)
	}
	if third.Media.Items[1].Embedding.URI == first.Media.Items[1].Embedding.URI {
		t.Fatal("new frame plan returned the old tensor")
	}
}

func TestEmbeddingCacheRejectsPoisonedLocation(t *testing.T) {
	e := newEnv(t, nil)
	e.worker("e0", "encode", nil)
	j := e.mediaJob()
	it := &j.Media.Items[0]
	e.model.Artifacts.Put(orchestrator.ArtifactKey(it), mwip.EmbeddingLocation{URI: "mock://wrong/item", Tokens: it.Plan.EmbeddingTokens(), PlanID: "wrong"})
	st, err := e.orch.Encode(context.Background(), j)
	if err != nil || st.Hits != 0 || st.Built != 2 {
		t.Fatalf("invalid cache entry was trusted: %+v %v", st, err)
	}
}
