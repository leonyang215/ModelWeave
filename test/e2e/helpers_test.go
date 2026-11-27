//go:build e2e

// Package e2e tests a running ModelWeave stack (deploy/compose): the gateway,
// the Rust tokenizer serving DeepSeek V4.1, Qwen3 and Qwen3-VL with their
// real tokenizers, and mock engines.
//
//	docker compose -f deploy/compose/docker-compose.yaml --profile test run --rm e2e
//	MW_E2E_GATEWAY=http://127.0.0.1:38080 MW_E2E_GATEWAY_EVENTS=http://127.0.0.1:38081 \
//	  MW_E2E_TOKENIZER=http://127.0.0.1:38090 go test -tags e2e -v ./test/e2e
package e2e

import (
	"bufio"
	"bytes"
	"compress/zlib"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

var (
	gateway   = env("MW_E2E_GATEWAY", "http://127.0.0.1:38080")
	gatewayEv = env("MW_E2E_GATEWAY_EVENTS", "http://127.0.0.1:38081")
	tokURL    = env("MW_E2E_TOKENIZER", "http://127.0.0.1:38090")
	ctlToken  = env("MW_E2E_CONTROL_TOKEN", "local-control-token")
	workerFmt = env("MW_E2E_WORKER_URL", "http://%s:9000") // mock workers by ID (compose network)
	client    = &http.Client{Timeout: 60 * time.Second}
)

// instanceURL is the HTTP base of a mock instance (one per container).
func instanceURL(id string) string { return fmt.Sprintf(workerFmt, id) }

// workerStats returns a mock instance's counters, including the last
// Generate request it received (in its JSON form, with proto field names).
func workerStats(t *testing.T, id string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(get(t, instanceURL(id)+"/mock/stats"), &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func lastRequest(t *testing.T, id string) map[string]any {
	t.Helper()
	m, _ := workerStats(t, id)["last_request"].(map[string]any)
	if m == nil {
		t.Fatalf("worker %s has no last request", id)
	}
	return m
}

const keyA = "sk-team-a"

type resp struct {
	status int
	header http.Header
	body   []byte
}

func (r *resp) json(t *testing.T) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(r.body, &m); err != nil {
		t.Fatalf("status %d, invalid JSON %q: %v", r.status, r.body, err)
	}
	return m
}

func post(t *testing.T, base, path, key string, body any, hdr map[string]string) *resp {
	t.Helper()
	b, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPost, base+path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	r, err := client.Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	defer r.Body.Close()
	data, _ := io.ReadAll(r.Body)
	return &resp{status: r.StatusCode, header: r.Header, body: data}
}

func get(t *testing.T, url string) []byte {
	t.Helper()
	req, _ := http.NewRequest(http.MethodGet, url, nil)
	req.Header.Set("X-ModelWeave-Token", ctlToken)
	r, err := client.Do(req)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer r.Body.Close()
	b, _ := io.ReadAll(r.Body)
	if r.StatusCode != 200 {
		t.Fatalf("GET %s: %d %s", url, r.StatusCode, b)
	}
	return b
}

type sseEvent struct {
	event string
	data  string
}

// stream posts a request and parses the SSE response.
func stream(t *testing.T, base, path, key string, body any, hdr map[string]string) (http.Header, []sseEvent) {
	t.Helper()
	b, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPost, base+path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	r, err := client.Do(req)
	if err != nil {
		t.Fatalf("stream %s: %v", path, err)
	}
	defer r.Body.Close()
	if r.StatusCode != 200 {
		data, _ := io.ReadAll(r.Body)
		t.Fatalf("stream %s: status %d: %s", path, r.StatusCode, data)
	}
	if ct := r.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content type %q", ct)
	}
	var out []sseEvent
	sc := bufio.NewScanner(r.Body)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	cur := sseEvent{}
	for sc.Scan() {
		line := sc.Text()
		switch {
		case line == "":
			if cur.data != "" || cur.event != "" {
				out = append(out, cur)
			}
			cur = sseEvent{}
		case strings.HasPrefix(line, "event: "):
			cur.event = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			cur.data = strings.TrimPrefix(line, "data: ")
		}
	}
	return r.Header, out
}

func nonce() string {
	var b [6]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// longPrompt is ~900 tokens, unique per call so tests do not share caches.
func longPrompt() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "Session %s. ", nonce())
	for i := 0; i < 60; i++ {
		fmt.Fprintf(&sb, "Paragraph %d: ModelWeave routes requests across prefill and decode pools. ", i)
	}
	return sb.String()
}

// metric sums the samples of a metric whose label set contains all labels.
func metric(t *testing.T, base, name string, labels ...string) float64 {
	t.Helper()
	total := 0.0
	for _, line := range strings.Split(string(get(t, base+"/metrics")), "\n") {
		if !strings.HasPrefix(line, name+"{") && !strings.HasPrefix(line, name+" ") {
			continue
		}
		ok := true
		for _, l := range labels {
			if !strings.Contains(line, l) {
				ok = false
				break
			}
		}
		if !ok {
			continue
		}
		f := strings.Fields(line)
		if v, err := strconv.ParseFloat(f[len(f)-1], 64); err == nil {
			total += v
		}
	}
	return total
}

type worker struct {
	ID             string         `json:"id"`
	Model          string         `json:"model"`
	Role           string         `json:"role"`
	Target         string         `json:"target"` // the instance's mwip target
	Source         string         `json:"source"`
	Normal         bool           `json:"normal"`
	Healthy        bool           `json:"healthy"`
	Draining       bool           `json:"draining"`
	AdminDrained   bool           `json:"admin_drained"`
	Blocked        bool           `json:"blocked"`
	HeartbeatAgeMS int64          `json:"heartbeat_age_ms"`
	IndexBlocks    int            `json:"index_blocks"`
	State          map[string]any `json:"state"`
}

func workers(t *testing.T, base string) map[string]worker {
	t.Helper()
	var out struct {
		Workers []worker `json:"workers"`
	}
	if err := json.Unmarshal(get(t, base+"/admin/workers"), &out); err != nil {
		t.Fatal(err)
	}
	m := map[string]worker{}
	for _, w := range out.Workers {
		m[w.ID] = w
	}
	return m
}

func chatBody(model, content string, extra map[string]any) map[string]any {
	b := map[string]any{"model": model, "messages": []map[string]any{{"role": "user", "content": content}}}
	for k, v := range extra {
		b[k] = v
	}
	return b
}

func message(t *testing.T, m map[string]any) map[string]any {
	t.Helper()
	ch, ok := m["choices"].([]any)
	if !ok || len(ch) == 0 {
		t.Fatalf("no choices in %v", m)
	}
	return ch[0].(map[string]any)
}

// tinyPNG builds a solid-color PNG data URL.
func tinyPNG(w, h int, r, g, b byte) string {
	raw := &bytes.Buffer{}
	for y := 0; y < h; y++ {
		raw.WriteByte(0)
		for x := 0; x < w; x++ {
			raw.Write([]byte{r, g, b})
		}
	}
	var z bytes.Buffer
	zw := zlib.NewWriter(&z)
	zw.Write(raw.Bytes())
	zw.Close()
	chunk := func(typ string, data []byte) []byte {
		var c bytes.Buffer
		l := make([]byte, 4)
		putU32(l, uint32(len(data)))
		c.Write(l)
		c.WriteString(typ)
		c.Write(data)
		crc := make([]byte, 4)
		putU32(crc, crc32.ChecksumIEEE(append([]byte(typ), data...)))
		c.Write(crc)
		return c.Bytes()
	}
	ihdr := make([]byte, 13)
	putU32(ihdr[0:], uint32(w))
	putU32(ihdr[4:], uint32(h))
	ihdr[8], ihdr[9] = 8, 2
	var png bytes.Buffer
	png.WriteString("\x89PNG\r\n\x1a\n")
	png.Write(chunk("IHDR", ihdr))
	png.Write(chunk("IDAT", z.Bytes()))
	png.Write(chunk("IEND", nil))
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(png.Bytes())
}

func putU32(b []byte, v uint32) {
	b[0], b[1], b[2], b[3] = byte(v>>24), byte(v>>16), byte(v>>8), byte(v)
}

// padValue is the protocol's pad value of a media reference, computed
// independently of the tokenizer: 2^30 + (sha256(reference)[0..8] as big-endian u64 & (2^30-1)).
func padValue(reference string) uint32 {
	h := sha256.Sum256([]byte(reference))
	return 1<<30 + uint32(binary.BigEndian.Uint64(h[:8])&(1<<30-1))
}

func sha256Hex(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// dataURLDigest is the sha256 of a data URL's bytes.
func dataURLDigest(t *testing.T, u string) string {
	t.Helper()
	b, err := base64.StdEncoding.DecodeString(u[strings.Index(u, ",")+1:])
	if err != nil {
		t.Fatal(err)
	}
	return sha256Hex(b)
}

// render calls the tokenizer directly and returns the intermediate state.
func render(t *testing.T, protocol string, body map[string]any) (int, map[string]any) {
	t.Helper()
	r := post(t, tokURL, "/v1/render", "", map[string]any{
		"request_id": "e2e-" + nonce(), "protocol": protocol, "model": "deepseek-v4.1", "body": body,
	}, nil)
	return r.status, r.json(t)
}

func detokenize(t *testing.T, ids []any) string {
	t.Helper()
	r := post(t, tokURL, "/v1/detokenize", "", map[string]any{"model": "deepseek-v4.1", "token_ids": ids}, nil)
	if r.status != 200 {
		t.Fatalf("detokenize: %d %s", r.status, r.body)
	}
	return r.json(t)["text"].(string)
}

// mediaItem returns item i of an intermediate state.
func mediaItem(t *testing.T, rr map[string]any, i int) map[string]any {
	t.Helper()
	mm, ok := rr["multimodal"].(map[string]any)
	if !ok {
		t.Fatalf("no multimodal section: %v", rr)
	}
	items := mm["items"].([]any)
	if len(items) <= i {
		t.Fatalf("want item %d, have %d", i, len(items))
	}
	return items[i].(map[string]any)
}

// checkPlacements verifies that every placement of an item fills its range
// of the token list with its pad value, computed with the protocol formula.
func checkPlacements(t *testing.T, rr, item map[string]any) []map[string]any {
	t.Helper()
	b, _ := json.Marshal(item)
	var typed mwip.MediaItem
	if err := json.Unmarshal(b, &typed); err != nil {
		t.Fatal(err)
	}
	if err := typed.Input().Validate(); err != nil {
		t.Fatalf("not an authoritative item plan: %v", err)
	}
	ids := rr["token_ids"].([]any)
	if int(rr["prompt_tokens"].(float64)) != len(ids) {
		t.Fatalf("prompt_tokens %v != len(token_ids) %d", rr["prompt_tokens"], len(ids))
	}
	var out []map[string]any
	for k, p := range item["placements"].([]any) {
		pl := p.(map[string]any)
		off, n := int(pl["offset"].(float64)), int(pl["length"].(float64))
		pv := uint32(pl["pad_value"].(float64))
		if want := padValue(pl["reference"].(string)); pv != want {
			t.Fatalf("pad_value of %s: %d, the protocol formula gives %d", pl["reference"], pv, want)
		}
		if pl["reference"] != mwip.PlacementReference(typed.PlanID, typed.Modality, k) || pl["embedding_offset"] != float64(k*typed.Plan.Tokens) {
			t.Fatalf("placement is not bound to plan/tensor rows: %v", pl)
		}
		if n <= 0 || off+n > len(ids) {
			t.Fatalf("placement out of range: %v", pl)
		}
		for i := off; i < off+n; i++ {
			if uint32(ids[i].(float64)) != pv {
				t.Fatalf("token %d is %v, want pad value %d", i, ids[i], pv)
			}
		}
		out = append(out, pl)
	}
	return out
}

func checkEmbedding(t *testing.T, item map[string]any) {
	t.Helper()
	e, ok := item["embedding"].(map[string]any)
	if !ok || e["uri"] == "" || e["uri"] == nil || e["plan_id"] != item["plan_id"] {
		t.Fatalf("missing/wrong embedding location: %v", item)
	}
	p := item["plan"].(map[string]any)
	placements := item["placements"].([]any)
	if e["tokens"] != float64(len(placements))*p["tokens"].(float64) {
		t.Fatalf("embedding rows differ from plan: %v", item)
	}
	for k, raw := range placements {
		pl := raw.(map[string]any)
		if pl["embedding_offset"] != float64(k)*p["tokens"].(float64) {
			t.Fatalf("bad tensor scatter offset: %v", pl)
		}
	}
}

// remoteImage is a chat image block with caller-measured metadata.
func remoteImage(url, digest string, w, h int, extra map[string]any) map[string]any {
	b := map[string]any{"type": "image_url", "image_url": map[string]any{"url": url},
		"metadata": map[string]any{"digest": digest, "width": w, "height": h}}
	for k, v := range extra {
		b[k] = v
	}
	return b
}

// remoteVideo is a 10 s, 30 fps 1280x720 chat video block.
func remoteVideo(url, digest string, frameOpts map[string]any) map[string]any {
	b := map[string]any{"type": "video_url", "video_url": map[string]any{"url": url},
		"metadata": map[string]any{"digest": digest, "width": 1280, "height": 720, "duration_ms": 10000, "total_frames": 300, "video_fps": 30}}
	if frameOpts != nil {
		b["frame_extract_options"] = frameOpts
	}
	return b
}

func userBlocks(model string, blocks ...any) map[string]any {
	return map[string]any{"model": model, "thinking": map[string]any{"type": "disabled"},
		"messages": []any{map[string]any{"role": "user", "content": blocks}}}
}

func text(s string) map[string]any { return map[string]any{"type": "text", "text": s} }

// path walks nested JSON objects.
func path(v any, keys ...string) any {
	for _, k := range keys {
		m, ok := v.(map[string]any)
		if !ok {
			return nil
		}
		v = m[k]
	}
	return v
}

func countToken(ids any, tok float64) int {
	n := 0
	list, _ := ids.([]any)
	for _, v := range list {
		if v == tok {
			n++
		}
	}
	return n
}
