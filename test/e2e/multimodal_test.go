//go:build e2e

package e2e

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// An inline image: the digest is the sha256 of its bytes, the size is read
// from the image, and its positions hold its pad value.
func TestTokenizerImagePlan(t *testing.T) {
	img := tinyPNG(320, 240, 10, 20, 30)
	status, rr := render(t, "chat_completions", userBlocks("deepseek-v4.1", text("Describe"), map[string]any{"type": "image_url", "image_url": map[string]any{"url": img}}))
	if status != 200 {
		t.Fatalf("render: %d %v", status, rr)
	}
	if path(rr, "multimodal", "placeholder_token_id") != nil || path(rr, "multimodal", "pad_shift") != float64(1<<30) {
		t.Fatalf("multimodal header: %v", rr["multimodal"])
	}
	it := mediaItem(t, rr, 0)
	ps := checkPlacements(t, rr, it)
	plan := it["plan"].(map[string]any)
	if it["modality"] != "image" || it["digest"] != dataURLDigest(t, img) || path(it, "source", "kind") != "base64" ||
		it["metadata"] != nil || it["placeholder"] != nil || plan["processor"] != "deepseek_v4_1.v1" {
		t.Fatalf("item: %v", it)
	}
	if len(ps) != 1 || ps[0]["reference"] != it["plan_id"] || ps[0]["modality"] != "image" ||
		ps[0]["length"] != plan["tokens"] || it["span"] != nil || plan["tokens"].(float64) < 50 {
		t.Fatalf("plan %v placements %v", plan, ps)
	}
	// The same bytes always give the same tokens (prefix caching).
	_, again := render(t, "chat_completions", userBlocks("deepseek-v4.1", text("Describe"), map[string]any{"type": "image_url", "image_url": map[string]any{"url": img}}))
	if fmt.Sprint(again["token_ids"]) != fmt.Sprint(rr["token_ids"]) {
		t.Fatal("rendering the same image twice differs")
	}
}

// Remote media carry their metadata; detail and pixel controls cap the
// size before the model's own resize.
func TestTokenizerPixelControls(t *testing.T) {
	digest := sha256Hex([]byte("e2e-photo"))
	url := "https://img.example/photo.jpg"
	tokens := func(extra map[string]any) float64 {
		t.Helper()
		status, rr := render(t, "chat_completions", userBlocks("deepseek-v4.1", remoteImage(url, digest, 1920, 1080, extra)))
		if status != 200 {
			t.Fatalf("render %v: %d %v", extra, status, rr)
		}
		it := mediaItem(t, rr, 0)
		checkPlacements(t, rr, it)
		if it["digest"] != digest || path(it, "source", "kind") != "url" || path(it, "source", "url") != url {
			t.Fatalf("item %v", it)
		}
		return path(it, "plan", "tokens").(float64)
	}
	full := tokens(nil)
	low := tokens(map[string]any{"image_url": map[string]any{"url": url, "detail": "low"}})
	capped := tokens(map[string]any{"pixel_control_options": map[string]any{"max_long_side": 1024}})
	if !(low < capped && capped < full) {
		t.Fatalf("tokens: full %v, max_long_side 1024 %v, detail low %v", full, capped, low)
	}
	t.Logf("1920x1080: %v tokens, max_long_side=1024: %v, detail=low: %v", full, capped, low)

	for name, tc := range map[string]struct {
		block map[string]any
		want  string
	}{
		"no metadata":      {map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}, "metadata.digest"},
		"no size":          {map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}, "metadata": map[string]any{"digest": digest}}, "metadata.width"},
		"bad digest":       {remoteImage(url, "abc", 10, 10, nil), "64 lowercase hex"},
		"too many pixels":  {remoteImage(url, digest, 1920, 1080, map[string]any{"pixel_control_options": map[string]any{"max_total_pixels": 100000}}), "max_total_pixels"},
		"frames for image": {remoteImage(url, digest, 64, 64, map[string]any{"frame_extract_options": map[string]any{"fps": 1}}), "videos only"},
	} {
		status, rr := render(t, "chat_completions", userBlocks("deepseek-v4.1", tc.block))
		if status != 400 || !strings.Contains(fmt.Sprint(rr), tc.want) {
			t.Errorf("%s: want 400 mentioning %q, got %d %v", name, tc.want, status, rr)
		}
	}
	// The gateway passes tokenizer rejections through in the client's shape.
	r := post(t, gateway, "/v1/chat/completions", keyA, userBlocks("deepseek-v4.1", map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}), nil)
	if r.status != 400 || !strings.Contains(string(r.body), "metadata.digest") {
		t.Fatalf("gateway: %d %s", r.status, r.body)
	}
}

// A video becomes its kept frames: each frame is its timestamp text plus a
// pad region whose reference is "{plan_id}_kf_{k:04}".
func TestTokenizerVideoPlan(t *testing.T) {
	digest := sha256Hex([]byte("e2e-video"))
	status, rr := render(t, "chat_completions", userBlocks("deepseek-v4.1", text("What happens?"),
		remoteVideo("https://video.example/clip.mp4", digest, map[string]any{"fps": 1})))
	if status != 200 {
		t.Fatalf("render: %d %v", status, rr)
	}
	it := mediaItem(t, rr, 0)
	ps := checkPlacements(t, rr, it)
	plan := it["plan"].(map[string]any)
	if it["modality"] != "video" || plan["nframes"] != float64(10) || len(plan["frame_indices"].([]any)) != 10 || plan["temporal_patch_size"] != float64(1) || len(ps) != 10 {
		t.Fatalf("video plan %v, %d placements", plan, len(ps))
	}
	ids := rr["token_ids"].([]any)
	tsBody := map[string]any{"model": "deepseek-v4.1", "text": fmt.Sprintf("%.1f seconds", ps[0]["timestamp"].(float64)), "parse_special": false}
	tsReply := post(t, tokURL, "/v1/tokenize_text", "", tsBody, nil)
	if tsReply.status != 200 {
		t.Fatalf("timestamp tokenize: %s", tsReply.body)
	}
	spanStart := int(ps[0]["offset"].(float64)) - len(tsReply.json(t)["token_ids"].([]any))
	prevEnd, prevTS := spanStart, -1.0
	for k, p := range ps {
		ts := p["timestamp"].(float64)
		if p["reference"] != fmt.Sprintf("%s_kf_%04d", it["plan_id"], k) || p["modality"] != "frame" || p["length"] != plan["tokens"] || ts <= prevTS {
			t.Fatalf("frame %d: %v", k, p)
		}
		off := int(p["offset"].(float64))
		if got, want := detokenize(t, ids[prevEnd:off]), fmt.Sprintf("%.1f seconds", ts); got != want {
			t.Fatalf("frame %d timestamp text %q, want %q", k, got, want)
		}
		prevEnd, prevTS = off+int(p["length"].(float64)), ts
	}
	t.Logf("10 s video at fps 1: %d groups of %v tokens, %v prompt tokens", len(ps), plan["tokens"], rr["prompt_tokens"])

	// nframes fixes the frame count. Frames default to a 672 px long side;
	// the frame pixel controls replace that (V4.1 never goes below 544² px,
	// so smaller limits end at the same 192 tokens).
	_, fixed := render(t, "chat_completions", userBlocks("deepseek-v4.1",
		remoteVideo("https://video.example/clip.mp4", digest, map[string]any{"nframes": 4, "pixel_control_options": map[string]any{"max_long_side": 1344}})))
	fp := path(mediaItem(t, fixed, 0), "plan").(map[string]any)
	if fp["nframes"] != float64(4) || len(fp["frame_indices"].([]any)) != 4 || fp["tokens"].(float64) <= plan["tokens"].(float64) || fmt.Sprint(fp["timestamps"]) != "[0 2.5 5 7.5]" {
		t.Fatalf("nframes=4, max_long_side=1344: %v", fp)
	}
	t.Logf("nframes=4, max_long_side=1344: %vx%v, %v tokens per frame", fp["width"], fp["height"], fp["tokens"])
	noDuration := remoteVideo("https://video.example/clip.mp4", digest, nil)
	delete(noDuration["metadata"].(map[string]any), "duration_ms")
	if status, out := render(t, "chat_completions", userBlocks("deepseek-v4.1", noDuration)); status != 400 || !strings.Contains(fmt.Sprint(out), "duration_ms") {
		t.Fatalf("video without duration: %d %v", status, out)
	}
	// Anthropic video blocks and Responses input_video are planned the same way.
	_, msg := render(t, "messages", map[string]any{"model": "deepseek-v4.1", "max_tokens": 32, "messages": []any{map[string]any{"role": "user", "content": []any{
		map[string]any{"type": "video", "source": map[string]any{"type": "url", "url": "https://video.example/clip.mp4"},
			"metadata": remoteVideo("", digest, nil)["metadata"], "frame_extract_options": map[string]any{"fps": 1}},
	}}}})
	_, resp := render(t, "responses", map[string]any{"model": "deepseek-v4.1", "input": []any{map[string]any{"role": "user", "content": []any{
		map[string]any{"type": "input_video", "video_url": "https://video.example/clip.mp4",
			"metadata": remoteVideo("", digest, nil)["metadata"], "frame_extract_options": map[string]any{"fps": 1}},
	}}}})
	for name, out := range map[string]map[string]any{"messages": msg, "responses": resp} {
		if p := path(mediaItem(t, out, 0), "plan").(map[string]any); len(p["frame_indices"].([]any)) != 10 || p["tokens"] != plan["tokens"] {
			t.Fatalf("%s video plan %v", name, p)
		}
	}
}

// mwip EPD: one input item yields one packed location, cached by content+plan.
func TestEPDImage(t *testing.T) {
	img := tinyPNG(320, 240, byte(time.Now().UnixNano()), 40, 200)
	body := userBlocks("deepseek-v4.1", text("What is in this image?"), map[string]any{"type": "image_url", "image_url": map[string]any{"url": img}})
	miss0 := metric(t, gateway, "modelweave_embedding_artifact_lookups_total", `model="deepseek-v4.1"`, `result="miss"`)
	hit0 := metric(t, gateway, "modelweave_embedding_artifact_lookups_total", `model="deepseek-v4.1"`, `result="hit"`)
	epd0 := metric(t, gateway, "modelweave_requests_total", `model="deepseek-v4.1"`, `mode="epd_`)
	enc0 := num(workerStats(t, "enc-0"), "encoded")
	_, rr := render(t, "chat_completions", body)
	for i := 0; i < 2; i++ {
		r := post(t, gateway, "/v1/chat/completions", keyA, body, nil)
		if r.status != 200 || !strings.Contains(string(r.body), "Hello from ModelWeave") {
			t.Fatalf("image request %d: %d %s", i, r.status, r.body)
		}
		if pt := path(r.json(t), "usage", "prompt_tokens"); pt != rr["prompt_tokens"] {
			t.Fatalf("engine prompt_tokens %v, tokenizer %v", pt, rr["prompt_tokens"])
		}
		// The worker that ran prefill got the intermediate state with the
		// embedding artifacts attached.
		w := r.header.Get("X-ModelWeave-Prefill-Worker")
		if w == "" {
			w = r.header.Get("X-ModelWeave-Decode-Worker")
		}
		req := lastRequest(t, w)
		item := path(req, "multimodal", "items").([]any)[0].(map[string]any)
		checkEmbedding(t, item)
		if len(req["token_ids"].([]any)) != int(rr["prompt_tokens"].(float64)) {
			t.Fatalf("%s request: token count", w)
		}
	}
	if d := metric(t, gateway, "modelweave_embedding_artifact_lookups_total", `model="deepseek-v4.1"`, `result="miss"`) - miss0; d != 1 {
		t.Fatalf("want exactly one encode miss, got %v", d)
	}
	if d := metric(t, gateway, "modelweave_embedding_artifact_lookups_total", `model="deepseek-v4.1"`, `result="hit"`) - hit0; d != 1 {
		t.Fatalf("want one artifact hit, got %v", d)
	}
	if d := metric(t, gateway, "modelweave_requests_total", `model="deepseek-v4.1"`, `mode="epd_`) - epd0; d != 2 {
		t.Fatalf("want 2 epd requests, got %v", d)
	}
	if d := num(workerStats(t, "enc-0"), "encoded") - enc0; d != 1 {
		t.Fatalf("enc-0 encoded %v items, want 1", d)
	}
}

// A whole video is encoded by one call; temporal slices share one location.
func TestEPDVideo(t *testing.T) {
	digest := sha256Hex([]byte("e2e-video-" + nonce()))
	body := userBlocks("deepseek-v4.1", text("What happens in the clip?"), remoteVideo("https://video.example/"+digest+".mp4", digest, map[string]any{"fps": 1}))
	_, rr := render(t, "chat_completions", body)
	r := post(t, gateway, "/v1/chat/completions", keyA, body, nil)
	if r.status != 200 || r.header.Get("X-ModelWeave-Mode") != "pd" || !strings.Contains(string(r.body), "Hello from ModelWeave") {
		t.Fatalf("video request: %d %v %s", r.status, r.header, r.body)
	}
	if pt := path(r.json(t), "usage", "prompt_tokens"); pt != rr["prompt_tokens"] {
		t.Fatalf("engine prompt_tokens %v, tokenizer %v", pt, rr["prompt_tokens"])
	}
	p := r.header.Get("X-ModelWeave-Prefill-Worker")
	req := lastRequest(t, p)
	item := path(req, "multimodal", "items").([]any)[0].(map[string]any)
	checkEmbedding(t, item)
	placements := item["placements"].([]any)
	if item["modality"] != "MODALITY_VIDEO" || len(placements) != 10 {
		t.Fatalf("%s: modality %v, %d groups", p, item["modality"], len(placements))
	}
	for k, a := range placements {
		if path(a, "reference") != fmt.Sprintf("%s_kf_%04d", item["plan_id"], k) {
			t.Fatalf("placement %d: %v", k, a)
		}
	}
	if metric(t, gateway, "modelweave_media_items_total", `model="deepseek-v4.1"`, `modality="video"`) == 0 ||
		metric(t, gateway, "modelweave_media_tokens_total", `model="deepseek-v4.1"`, `modality="video"`) < 1000 {
		t.Fatal("video media metrics missing")
	}
}

func num(st map[string]any, key string) float64 {
	v, _ := st[key].(float64)
	return v
}
