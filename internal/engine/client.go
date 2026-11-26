// Package engine is the gateway's client of the mwip Engine service
// (proto/modelweave/mwip/v1/engine.proto), the protocol every engine
// instance implements. Generate always streams, whatever the client asked
// for; Encode builds the embeddings of the encode stage; Abort stops
// requests; GetState reports an instance's load.
package engine

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

// MaxMessageBytes bounds mwip messages both ways (a 1M-token prompt is
// about 5 MiB).
const MaxMessageBytes = 64 << 20

// Error is a failed engine call.
type Error struct {
	Role      string
	WorkerID  string
	Code      string // mwip error code
	Message   string
	Retriable bool
	// Transport is true when the instance could not be reached or the
	// connection broke; it counts against the instance's health.
	Transport bool
}

func (e *Error) Error() string {
	return fmt.Sprintf("%s worker %s: %s: %s", e.Role, e.WorkerID, e.Code, e.Message)
}

// IsCode reports whether err is an *Error with the given code.
func IsCode(err error, code string) bool {
	var e *Error
	return errors.As(err, &e) && e.Code == code
}

// Stage is one Generate call of an attempt: the intermediate state plus the
// part this call plays in the execution.
type Stage struct {
	Render      *mwip.RenderResult
	AttemptID   string // request_id of the call; P and D share it
	EngineModel string
	Disagg      mwip.Disagg
	// Media is the request's media with the embedding artifacts of the
	// encode stage attached; defaults to Render.Multimodal.
	Media      *mwip.Multimodal
	TimeoutMS  int64
	SessionKey string
	Priority   string
}

func (s *Stage) media() *mwip.Multimodal {
	if s.Media != nil {
		return s.Media
	}
	return s.Render.Multimodal
}

func (s *Stage) role(spec registry.Spec) string {
	if s.Disagg.Role != "" {
		return s.Disagg.Role
	}
	return string(spec.Role)
}

// request is the mwip GenerateRequest of a stage: the intermediate state as
// is, with its media (and their embedding artifacts) and the execution
// fields.
func (s *Stage) request() *mwip.GenerateRequest {
	r := s.Render
	return &mwip.GenerateRequest{
		RequestID: s.AttemptID, Model: s.EngineModel, TokenIDs: r.TokenIDs, Multimodal: s.media(),
		Conversation: r.Conversation, InferenceOptions: r.InferenceOptions, ParsingOptions: r.ParsingOptions,
		Disagg: s.Disagg, TimeoutMS: s.TimeoutMS, SessionKey: s.SessionKey, Priority: s.Priority,
	}
}

// EncodeStage asks an encode instance to execute one item's frozen plan.
type EncodeStage struct {
	RequestID   string
	AttemptID   string
	Model       string
	EngineModel string
	Item        *mwip.MediaInput
	TimeoutMS   int64
}

// EncodeResult locates one packed item embedding, readable by P and D.
type EncodeResult struct {
	Embedding *mwip.EmbeddingLocation
	EncodeMS  float64
}

// Client calls the Engine service of instances, with one gRPC connection
// per target.
type Client struct {
	mu    sync.Mutex
	conns map[string]*grpc.ClientConn
}

// New returns a client.
func New() *Client { return &Client{conns: map[string]*grpc.ClientConn{}} }

// Close closes the connections.
func (c *Client) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for t, cc := range c.conns {
		_ = cc.Close()
		delete(c.conns, t)
	}
}

// DialTarget turns an mwip target into a gRPC target and its credentials:
// "host:port" and "http://host:port" are plaintext HTTP/2,
// "https://host:port" is TLS.
func DialTarget(target string) (string, credentials.TransportCredentials) {
	switch {
	case strings.HasPrefix(target, "https://"):
		return strings.TrimRight(strings.TrimPrefix(target, "https://"), "/"), credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12})
	case strings.HasPrefix(target, "http://"):
		return strings.TrimRight(strings.TrimPrefix(target, "http://"), "/"), insecure.NewCredentials()
	}
	return target, insecure.NewCredentials()
}

func (c *Client) conn(target string) (*grpc.ClientConn, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if cc, ok := c.conns[target]; ok {
		return cc, nil
	}
	addr, creds := DialTarget(target)
	cc, err := grpc.NewClient(addr, grpc.WithTransportCredentials(creds),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(MaxMessageBytes), grpc.MaxCallSendMsgSize(MaxMessageBytes)))
	if err != nil {
		return nil, err
	}
	c.conns[target] = cc
	return cc, nil
}

func (c *Client) engine(role string, spec registry.Spec) (mwipv1.EngineClient, error) {
	cc, err := c.conn(spec.Target)
	if err != nil {
		return nil, &Error{Role: role, WorkerID: spec.ID, Code: mwip.ErrBadRequest, Message: "mwip target " + spec.Target + ": " + err.Error()}
	}
	return mwipv1.NewEngineClient(cc), nil
}

// Generate starts an Engine.Generate stream on an instance.
func (c *Client) Generate(ctx context.Context, spec registry.Spec, st *Stage) (*Stream, error) {
	role := st.role(spec)
	req, err := mwip.RequestToProto(st.request())
	if err != nil {
		return nil, &Error{Role: role, WorkerID: spec.ID, Code: mwip.ErrBadRequest, Message: err.Error()}
	}
	ec, err := c.engine(role, spec)
	if err != nil {
		return nil, err
	}
	sctx, cancel := context.WithCancel(ctx)
	rpc, err := ec.Generate(sctx, req)
	if err != nil {
		cancel()
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, statusErr(role, spec.ID, err)
	}
	return &Stream{role: role, worker: spec.ID, prefill: role == "prefill", rpc: rpc, cancel: cancel}, nil
}

// Encode runs Engine.Encode on an encode instance.
func (c *Client) Encode(ctx context.Context, spec registry.Spec, st *EncodeStage) (*EncodeResult, error) {
	if st.Item == nil {
		return nil, &Error{Role: "encode", WorkerID: spec.ID, Code: mwip.ErrBadRequest, Message: "Encode requires one item"}
	}
	if err := st.Item.Validate(); err != nil {
		return nil, &Error{Role: "encode", WorkerID: spec.ID, Code: mwip.ErrBadRequest, Message: err.Error()}
	}
	ec, err := c.engine("encode", spec)
	if err != nil {
		return nil, err
	}
	resp, err := ec.Encode(ctx, mwip.EncodeRequestToProto(&mwip.EncodeRequest{
		RequestID: st.RequestID, AttemptID: st.AttemptID, Model: st.EngineModel, Item: st.Item, TimeoutMS: st.TimeoutMS,
	}))
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, statusErr("encode", spec.ID, err)
	}
	embedding := mwip.EmbeddingFromProto(resp.GetEmbedding())
	if err := st.Item.ValidateEmbedding(embedding); err != nil {
		return nil, &Error{Role: "encode", WorkerID: spec.ID, Code: mwip.ErrInternal, Message: "invalid encoder result: " + err.Error(), Retriable: true}
	}
	embedding.Encoder = spec.ID
	return &EncodeResult{Embedding: embedding, EncodeMS: resp.GetEncodeMs()}, nil
}

// Abort asks an instance to stop a request (Engine.Abort). It is best
// effort: the gateway also cancels the call.
func (c *Client) Abort(ctx context.Context, spec registry.Spec, requestID string) error {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	ec, err := c.engine("abort", spec)
	if err != nil {
		return err
	}
	_, err = ec.Abort(ctx, &mwipv1.AbortRequest{RequestIds: []string{requestID}})
	return err
}

// State asks an instance for its load (Engine.GetState).
func (c *Client) State(ctx context.Context, spec registry.Spec) (registry.EngineState, error) {
	ec, err := c.engine("state", spec)
	if err != nil {
		return registry.EngineState{}, err
	}
	resp, err := ec.GetState(ctx, &mwipv1.GetStateRequest{})
	if err != nil {
		return registry.EngineState{}, err
	}
	return registry.StateFromProto(resp.GetState()), nil
}

func transportErr(role, worker string, err error) *Error {
	return &Error{Role: role, WorkerID: worker, Code: "unavailable", Message: err.Error(), Retriable: true, Transport: true}
}

// statusErr classifies a failed call: an mwip Error detail of the status
// decides; otherwise the status code does.
func statusErr(role, worker string, err error) *Error {
	st, ok := status.FromError(err)
	if !ok {
		return transportErr(role, worker, err)
	}
	for _, d := range st.Details() {
		if me, ok := d.(*mwipv1.Error); ok {
			return &Error{Role: role, WorkerID: worker, Code: mwip.ErrorCodeFromProto(me.GetCode()), Message: me.GetMessage(), Retriable: me.GetRetriable()}
		}
	}
	e := &Error{Role: role, WorkerID: worker, Code: mwip.ErrInternal, Message: st.Message(), Retriable: true}
	switch st.Code() {
	case codes.Unavailable:
		e.Code, e.Transport = "unavailable", true
	case codes.ResourceExhausted:
		e.Code = mwip.ErrCapacity
	case codes.InvalidArgument, codes.OutOfRange, codes.FailedPrecondition:
		e.Code, e.Retriable = mwip.ErrBadRequest, false
	case codes.DeadlineExceeded:
		e.Code = mwip.ErrTimeout
	case codes.Canceled, codes.Aborted:
		e.Code = mwip.ErrAborted
	}
	return e
}
