package mockengine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// modelInfo is what the tokenizer reports about the worker's model
// (GET /v1/models): how its output is marked up.
type modelInfo struct {
	Name            string   `json:"name"`
	ReasoningParser string   `json:"reasoning_parser"`
	ToolCallParser  string   `json:"tool_call_parser"`
	Thinking        string   `json:"thinking"` // toggle | always | none
	ThinkStart      *uint32  `json:"think_start_token_id"`
	ThinkEnd        *uint32  `json:"think_end_token_id"`
	EOS             []uint32 `json:"eos_token_ids"`
}

// deepseekInfo describes DeepSeek V4.1, the model of a worker without a
// tokenizer (unit tests).
var deepseekInfo = modelInfo{
	ReasoningParser: mwip.ParserDeepseekV4, ToolCallParser: mwip.ParserDeepseekV4, Thinking: "toggle",
	ThinkStart: ptr(uint32(128821)), ThinkEnd: ptr(uint32(128822)), EOS: []uint32{1},
}

func ptr[T any](v T) *T { return &v }

// vocab turns canned output text into token IDs and per-token text through
// the tokenizer service, so special-token spellings such as </think>,
// <tool_call> and DSML tags become the model's real tokens. Without a
// tokenizer it falls back to a word-level stand-in (unit tests).
type vocab struct {
	engine *Engine
	mu     sync.Mutex
	cache  map[string]tokenized
	info   *modelInfo
	texts  map[uint32]string
}

type tokenized struct {
	ids    []uint32
	pieces []string
}

const filler = "ModelWeave weaves gateways, schedulers and tokenizers into one serving fabric. "

func (v *vocab) post(ctx context.Context, path string, req, out any) error {
	body, _ := json.Marshal(req)
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, v.engine.cfg.TokenizerURL+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	hreq.Header.Set("Content-Type", "application/json")
	resp, err := v.engine.http.Do(hreq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("tokenizer %s: status %d", path, resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// model returns the worker model's markup, from the tokenizer.
func (v *vocab) model(ctx context.Context) (modelInfo, error) {
	v.mu.Lock()
	if v.info != nil {
		defer v.mu.Unlock()
		return *v.info, nil
	}
	v.mu.Unlock()
	if v.engine.cfg.TokenizerURL == "" {
		return deepseekInfo, nil
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, v.engine.cfg.TokenizerURL+"/v1/models", nil)
	if err != nil {
		return modelInfo{}, err
	}
	resp, err := v.engine.http.Do(req)
	if err != nil {
		return modelInfo{}, err
	}
	defer resp.Body.Close()
	var list struct {
		Models []struct {
			modelInfo
			Aliases []string `json:"aliases"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return modelInfo{}, err
	}
	want := v.engine.cfg.TokenizerModel
	for _, m := range list.Models {
		match := want == "" || m.Name == want
		for _, a := range m.Aliases {
			match = match || a == want
		}
		if match {
			info := m.modelInfo
			v.mu.Lock()
			v.info = &info
			v.mu.Unlock()
			return info, nil
		}
	}
	return modelInfo{}, fmt.Errorf("the tokenizer does not serve model %q", want)
}

// tokenText is the text of one token (logprob entries of stop tokens).
func (v *vocab) tokenText(ctx context.Context, id uint32) string {
	v.mu.Lock()
	t, ok := v.texts[id]
	v.mu.Unlock()
	if ok {
		return t
	}
	t = fmt.Sprintf("<token:%d>", id)
	if v.engine.cfg.TokenizerURL != "" {
		var dec struct {
			Text string `json:"text"`
		}
		if v.post(ctx, "/v1/detokenize", map[string]any{"model": v.engine.cfg.TokenizerModel, "token_ids": []uint32{id}}, &dec) == nil {
			t = dec.Text
		}
	}
	v.mu.Lock()
	if v.texts == nil {
		v.texts = map[uint32]string{}
	}
	v.texts[id] = t
	v.mu.Unlock()
	return t
}

func (v *vocab) tokenize(ctx context.Context, text string) (tokenized, error) {
	v.mu.Lock()
	if v.cache == nil {
		v.cache = map[string]tokenized{}
	}
	if t, ok := v.cache[text]; ok {
		v.mu.Unlock()
		return t, nil
	}
	v.mu.Unlock()
	var t tokenized
	if v.engine.cfg.TokenizerURL == "" {
		t = fallbackTokenize(text)
	} else {
		model := v.engine.cfg.TokenizerModel
		var enc struct {
			TokenIDs []uint32 `json:"token_ids"`
		}
		if err := v.post(ctx, "/v1/tokenize_text", map[string]any{"model": model, "text": text, "parse_special": true}, &enc); err != nil {
			return t, err
		}
		var dec struct {
			Pieces []string `json:"pieces"`
		}
		if err := v.post(ctx, "/v1/detokenize", map[string]any{"model": model, "token_ids": enc.TokenIDs, "skip_special": false, "per_token": true}, &dec); err != nil {
			return t, err
		}
		if len(dec.Pieces) != len(enc.TokenIDs) {
			return t, fmt.Errorf("tokenizer returned %d pieces for %d tokens", len(dec.Pieces), len(enc.TokenIDs))
		}
		t = tokenized{ids: enc.TokenIDs, pieces: dec.Pieces}
	}
	v.mu.Lock()
	v.cache[text] = t
	v.mu.Unlock()
	return t, nil
}

var fallbackSplit = regexp.MustCompile(`</?think>|</?tool_call>|</?｜DSML｜|\s*[\p{L}\p{N}_]+|\s+|.`)

// fallbackTokenize splits text into word-like pieces with stable fake IDs.
func fallbackTokenize(text string) tokenized {
	var t tokenized
	for _, p := range fallbackSplit.FindAllString(text, -1) {
		h := fnv.New32a()
		_, _ = h.Write([]byte(p))
		t.ids = append(t.ids, 1000+h.Sum32()%100000)
		t.pieces = append(t.pieces, p)
	}
	return t
}

const thought = "The user is talking to the ModelWeave mock engine; I will answer briefly."

// answer is the canned output for a request, in the markup of its model:
// what the model "generates".
func (v *vocab) answer(g *genReq) string {
	id := v.engine.cfg.ID
	var b strings.Builder
	hermes := g.toolParser == mwip.ParserHermes
	required := g.callsOpened && !hermes
	if g.reasoning && !required {
		if g.openThink {
			b.WriteString(thinkStart + "\n")
		}
		b.WriteString(thought)
		if g.reasoningParser == mwip.ParserQwen3 {
			b.WriteString("\n" + thinkEnd + "\n\n")
		} else {
			b.WriteString(thinkEnd)
		}
	}
	switch {
	case len(g.tools) > 0 && g.toolChoice != "none" && hermes:
		args, _ := json.Marshal(map[string]string{"query": "ModelWeave", "worker": id})
		fmt.Fprintf(&b, "%s\n{\"name\": %q, \"arguments\": %s}\n%s", toolCallOpen, g.tools[0], args, toolCallClose)
	case len(g.tools) > 0 && g.toolChoice != "none":
		if !required {
			b.WriteString("\n\n<｜DSML｜ calls>\n")
		}
		fmt.Fprintf(&b, "<｜DSML｜ invoke name=\"%s\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">ModelWeave</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"worker\" string=\"true\">%s</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>", g.tools[0], id)
	case g.structured:
		fmt.Fprintf(&b, `{"answer": "ModelWeave mock", "worker": "%s"}`, id)
	default:
		fmt.Fprintf(&b, "Hello from ModelWeave mock worker %s. Count: one two three four five six seven eight nine ten.", id)
		for i := 0; i < v.engine.cfg.AnswerRepeat; i++ {
			b.WriteString(" " + filler)
		}
	}
	return b.String()
}
