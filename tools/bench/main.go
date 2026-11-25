// Command bench is a small load generator for ModelWeave. It sends streaming
// chat requests in prefix groups (requests of a group share a long system
// prompt, as with agents or shared documents) and reports client-side TTFT
// and the gateway's routing decisions (X-ModelWeave-* headers).
//
//	go run ./tools/bench -url http://127.0.0.1:38080 -groups 8 -requests 160 -concurrency 16
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand/v2"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

type result struct {
	status        int
	ttft, total   time.Duration
	mode, reason  string
	decode        string
	cachedHeader  string
	promptTokens  float64
	cachedTokens  float64
	completionTok float64
	err           string
}

func main() {
	url := flag.String("url", "http://127.0.0.1:38080", "gateway URL")
	key := flag.String("key", "sk-team-a", "API key")
	model := flag.String("model", "deepseek-v4.1", "model")
	groups := flag.Int("groups", 8, "number of shared-prefix groups")
	prefixWords := flag.Int("prefix-words", 700, "words in each group's shared prefix")
	requests := flag.Int("requests", 160, "total requests")
	conc := flag.Int("concurrency", 16, "concurrent requests")
	flag.Parse()

	prefixes := make([]string, *groups)
	run := rand.Uint64()
	for g := range prefixes {
		var sb strings.Builder
		fmt.Fprintf(&sb, "Run %x document %d. ", run, g)
		for i := 0; i < *prefixWords; i++ {
			fmt.Fprintf(&sb, "w%d ", (g*7919+i)%997)
		}
		prefixes[g] = sb.String()
	}
	client := &http.Client{Timeout: 2 * time.Minute}
	jobs := make(chan int)
	results := make([]result, *requests)
	var wg sync.WaitGroup
	start := time.Now()
	for w := 0; w < *conc; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				results[i] = one(client, *url, *key, *model, prefixes[i%*groups], i)
			}
		}()
	}
	for i := 0; i < *requests; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	elapsed := time.Since(start)
	report(results, elapsed)
}

func one(c *http.Client, url, key, model, prefix string, i int) result {
	body, _ := json.Marshal(map[string]any{
		"model": model, "stream": true, "stream_options": map[string]any{"include_usage": true},
		"thinking": map[string]any{"type": "disabled"},
		"messages": []map[string]any{
			{"role": "system", "content": prefix},
			{"role": "user", "content": fmt.Sprintf("Question %d: summarize the document.", i)},
		},
	})
	req, _ := http.NewRequest(http.MethodPost, url+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	t0 := time.Now()
	resp, err := c.Do(req)
	if err != nil {
		return result{err: err.Error()}
	}
	defer resp.Body.Close()
	r := result{
		status: resp.StatusCode, mode: resp.Header.Get("X-ModelWeave-Mode"), reason: resp.Header.Get("X-ModelWeave-Reason"),
		decode: resp.Header.Get("X-ModelWeave-Decode-Worker"), cachedHeader: resp.Header.Get("X-ModelWeave-Cached-Tokens"),
	}
	if resp.StatusCode != 200 {
		var b bytes.Buffer
		b.ReadFrom(resp.Body)
		r.err = strings.TrimSpace(b.String())
		return r
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 8<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content *string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     float64 `json:"prompt_tokens"`
				CompletionTokens float64 `json:"completion_tokens"`
				Details          struct {
					Cached float64 `json:"cached_tokens"`
				} `json:"prompt_tokens_details"`
			} `json:"usage"`
		}
		if json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk) != nil {
			continue
		}
		if r.ttft == 0 && len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != nil && *chunk.Choices[0].Delta.Content != "" {
			r.ttft = time.Since(t0)
		}
		if chunk.Usage != nil {
			r.promptTokens, r.completionTok, r.cachedTokens = chunk.Usage.PromptTokens, chunk.Usage.CompletionTokens, chunk.Usage.Details.Cached
		}
	}
	r.total = time.Since(t0)
	return r
}

func pct(ds []time.Duration, p float64) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
	return ds[min(len(ds)-1, int(float64(len(ds))*p))]
}

func report(rs []result, elapsed time.Duration) {
	byMode := map[string][]time.Duration{}
	reasons := map[string]int{}
	workers := map[string]int{}
	codes := map[int]int{}
	var prompt, cached, completion float64
	var all []time.Duration
	for _, r := range rs {
		codes[r.status]++
		if r.status != 200 {
			if r.err != "" {
				fmt.Fprintln(os.Stderr, "error:", r.status, r.err)
			}
			continue
		}
		byMode[r.mode] = append(byMode[r.mode], r.ttft)
		all = append(all, r.ttft)
		reasons[r.mode+"/"+r.reason]++
		workers[r.decode]++
		prompt += r.promptTokens
		cached += r.cachedTokens
		completion += r.completionTok
	}
	fmt.Printf("requests: %d in %s (%.1f req/s), status codes %v\n", len(rs), elapsed.Round(time.Millisecond), float64(len(rs))/elapsed.Seconds(), codes)
	fmt.Printf("tokens: prompt %.0f, cached %.0f (%.1f%% of prompt tokens served from KV cache), completion %.0f\n",
		prompt, cached, 100*cached/max(prompt, 1), completion)
	fmt.Printf("TTFT all: p50 %s  p95 %s\n", pct(all, 0.5).Round(time.Microsecond*100), pct(all, 0.95).Round(time.Microsecond*100))
	modes := make([]string, 0, len(byMode))
	for m := range byMode {
		modes = append(modes, m)
	}
	sort.Strings(modes)
	for _, m := range modes {
		ds := byMode[m]
		fmt.Printf("  mode %-10s n=%-4d TTFT p50 %-8s p95 %s\n", m, len(ds), pct(ds, 0.5).Round(time.Microsecond*100), pct(ds, 0.95).Round(time.Microsecond*100))
	}
	keys := make([]string, 0, len(reasons))
	for k := range reasons {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Println("plans (mode/reason):")
	for _, k := range keys {
		fmt.Printf("  %-28s %d\n", k, reasons[k])
	}
	fmt.Println("decode workers:", workers)
}
