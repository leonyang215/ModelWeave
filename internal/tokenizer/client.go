// Package tokenizer is the gateway's client of the tokenizer service.
//
// The tokenizer only converts and tokenizes: POST /v1/render turns an API
// request into the intermediate state (the token list of the model's prompt
// format, media planned and expanded, generation and parsing options). It is
// stateless, so any replica can serve any request: a render that cannot reach
// a replica is retried on the next one, and the unreachable replica is
// skipped for a short while.
package tokenizer

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// Error is a tokenizer-side failure. Body is the protocol-shaped error body
// to return to the client when Status is set.
type Error struct {
	Status int
	Body   json.RawMessage
	Err    error
}

func (e *Error) Error() string {
	if e.Err != nil {
		return "tokenizer: " + e.Err.Error()
	}
	return fmt.Sprintf("tokenizer: status %d: %s", e.Status, e.Body)
}

func (e *Error) Unwrap() error { return e.Err }

// Client talks to one or more tokenizer replicas.
type Client struct {
	endpoints []string
	next      atomic.Uint64
	http      *http.Client
	timeout   time.Duration
	// Replicas that failed at transport level are skipped until then.
	mu       sync.Mutex
	downTill map[string]time.Time
	// Cooldown is how long an unreachable replica is skipped.
	Cooldown time.Duration
	// OnUnreachable, when set, is called for a replica that could not be
	// reached; the render moves on to the next replica.
	OnUnreachable func(endpoint string, err error)
	now           func() time.Time
}

// New returns a client. timeout bounds one render call.
func New(endpoints []string, timeout time.Duration) *Client {
	dialer := &net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}
	tr := &http.Transport{
		DialContext:         dialer.DialContext,
		MaxIdleConns:        256,
		MaxIdleConnsPerHost: 64,
		IdleConnTimeout:     90 * time.Second,
	}
	// h2c multiplexes concurrent renders over few connections.
	tr.Protocols = new(http.Protocols)
	tr.Protocols.SetUnencryptedHTTP2(true)
	tr.HTTP2 = &http.HTTP2Config{SendPingTimeout: 15 * time.Second, PingTimeout: 10 * time.Second}
	return &Client{endpoints: endpoints, http: &http.Client{Transport: tr}, timeout: timeout,
		downTill: map[string]time.Time{}, Cooldown: 5 * time.Second, now: time.Now}
}

// order returns the replicas to try for one render: round-robin from the
// next one, replicas in their cooldown last.
func (c *Client) order() []string {
	n := len(c.endpoints)
	start := int((c.next.Add(1) - 1) % uint64(n))
	now := c.now()
	up := make([]string, 0, n)
	var down []string
	c.mu.Lock()
	for i := 0; i < n; i++ {
		e := c.endpoints[(start+i)%n]
		if now.Before(c.downTill[e]) {
			down = append(down, e)
		} else {
			up = append(up, e)
		}
	}
	c.mu.Unlock()
	return append(up, down...)
}

func (c *Client) markDown(endpoint string) {
	c.mu.Lock()
	c.downTill[endpoint] = c.now().Add(c.Cooldown)
	c.mu.Unlock()
}

func (c *Client) markUp(endpoint string) {
	c.mu.Lock()
	delete(c.downTill, endpoint)
	c.mu.Unlock()
}

// Render converts an API request. It returns an *Error, with the
// protocol-shaped body when the tokenizer rejected the request. A replica
// that cannot be reached is skipped for the next one; answers, including
// rejections, are final.
func (c *Client) Render(ctx context.Context, req *mwip.RenderRequest) (*mwip.RenderResult, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, &Error{Err: err}
	}
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	var last error
	for _, endpoint := range c.order() {
		rr, reached, err := c.render(ctx, endpoint, req.RequestID, body)
		if reached {
			c.markUp(endpoint)
			return rr, err
		}
		last = err
		if ctx.Err() != nil {
			break
		}
		c.markDown(endpoint)
		if c.OnUnreachable != nil {
			c.OnUnreachable(endpoint, err)
		}
	}
	return nil, last
}

// render posts to one replica; reached is false when the replica could not
// be reached.
func (c *Client) render(ctx context.Context, endpoint, requestID string, body []byte) (*mwip.RenderResult, bool, error) {
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint+"/v1/render", bytes.NewReader(body))
	if err != nil {
		return nil, true, &Error{Err: err}
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("X-Request-Id", requestID)
	resp, err := c.http.Do(hreq)
	if err != nil {
		return nil, false, &Error{Err: err}
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, true, &Error{Err: err}
	}
	if resp.StatusCode != http.StatusOK {
		return nil, true, &Error{Status: resp.StatusCode, Body: b}
	}
	var rr mwip.RenderResult
	if err := json.Unmarshal(b, &rr); err != nil {
		return nil, true, &Error{Err: fmt.Errorf("decode render result: %w", err)}
	}
	return &rr, true, nil
}
