package mockengine

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protojson"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
	"github.com/leonyang215/ModelWeave/internal/engine"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

// engineService implements the mwip Engine service of one instance.
type engineService struct {
	mwipv1.UnimplementedEngineServer
	e *Engine
}

// NewGRPCServer returns a gRPC server serving the instance's mwip Engine
// service.
func NewGRPCServer(e *Engine) *grpc.Server {
	srv := grpc.NewServer(grpc.MaxRecvMsgSize(engine.MaxMessageBytes), grpc.MaxSendMsgSize(engine.MaxMessageBytes))
	mwipv1.RegisterEngineServer(srv, &engineService{e: e})
	return srv
}

// recorded is the JSON form of a request with proto field names and every
// field (zero values included), kept as the last request for tests.
var recorded = protojson.MarshalOptions{UseProtoNames: true, EmitUnpopulated: true}

func (s *engineService) Generate(req *mwipv1.GenerateRequest, stream mwipv1.Engine_GenerateServer) error {
	e := s.e
	out := &grpcStream{stream: stream}
	raw, _ := recorded.Marshal(req)
	greq, err := mwip.RequestFromProto(req)
	if err != nil {
		_ = out.fail(mwip.ErrBadRequest, "invalid generate request: "+err.Error(), false)
		return nil
	}
	info, err := e.vocab.model(stream.Context())
	if err != nil {
		_ = out.fail(mwip.ErrInternal, "model info: "+err.Error(), true)
		return nil
	}
	e.serve(stream.Context(), out, fromMWIP(greq, info), raw)
	return nil
}

func (s *engineService) Encode(ctx context.Context, req *mwipv1.EncodeRequest) (*mwipv1.EncodeResponse, error) {
	raw, _ := recorded.Marshal(req)
	s.e.count(func() { s.e.stats.LastEncode = raw })
	resp, err := s.e.encode(ctx, mwip.EncodeRequestFromProto(req))
	if err != nil {
		return nil, codeStatus(err)
	}
	return &mwipv1.EncodeResponse{Embedding: mwip.EmbeddingToProto(resp.Embedding), EncodeMs: resp.EncodeMS}, nil
}

func (s *engineService) Abort(_ context.Context, req *mwipv1.AbortRequest) (*mwipv1.AbortResponse, error) {
	return &mwipv1.AbortResponse{Aborted: int32(s.e.abort(req.GetRequestIds()))}, nil
}

func (s *engineService) GetState(context.Context, *mwipv1.GetStateRequest) (*mwipv1.GetStateResponse, error) {
	return &mwipv1.GetStateResponse{State: registry.StateToProto(s.e.State())}, nil
}

// codeStatus turns a failure into a gRPC status carrying its mwip Error.
func codeStatus(err error) error {
	ce, ok := err.(*codeError)
	if !ok {
		return status.Error(codes.Internal, err.Error())
	}
	code := codes.Internal
	switch ce.code {
	case mwip.ErrBadRequest:
		code = codes.InvalidArgument
	case mwip.ErrCapacity:
		code = codes.ResourceExhausted
	case mwip.ErrAborted:
		code = codes.Canceled
	}
	st, derr := status.New(code, ce.msg).WithDetails(&mwipv1.Error{Code: mwip.ErrorCodeToProto(ce.code), Message: ce.msg, Retriable: ce.retriable})
	if derr != nil {
		return status.Error(code, ce.msg)
	}
	return st.Err()
}
