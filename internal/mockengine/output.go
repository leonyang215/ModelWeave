package mockengine

import (
	"fmt"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// tokenOut is one generated token.
type tokenOut struct {
	id      uint32
	text    string // detokenized text released by the stop-string check
	delta   parsed // engine-side parse of text
	logprob *mwip.LogprobContent
	finish  *finishOut // set on the last token
}

type finishOut struct {
	reason     string // stop | length | tool_calls
	stopReason any    // matched stop string, or nil
	usage      mwip.Usage
	timing     *mwip.Timing
}

// sink receives the output of a Generate call.
type sink interface {
	// accepted acknowledges a push decode leg with its KV receiver.
	accepted(rcv *mwip.KVReceiver) error
	// prefillDone ends a prefill leg.
	prefillDone(ev *mwip.GenerateEvent) error
	token(t tokenOut) error
	// fail sends an Error event (the call itself succeeds) and returns the
	// failure.
	fail(code, msg string, retriable bool) error
}

// grpcStream writes the events of an mwip Generate stream.
type grpcStream struct {
	stream mwipv1.Engine_GenerateServer
}

func (s *grpcStream) event(ev *mwip.GenerateEvent) error {
	return s.stream.Send(mwip.EventToProto(ev))
}

func (s *grpcStream) accepted(rcv *mwip.KVReceiver) error {
	return s.event(&mwip.GenerateEvent{Type: mwip.EventAccepted, Receiver: rcv})
}

func (s *grpcStream) prefillDone(ev *mwip.GenerateEvent) error { return s.event(ev) }

func (s *grpcStream) token(t tokenOut) error {
	ev := &mwip.GenerateEvent{Type: mwip.EventDelta, TokenIDs: []uint32{t.id},
		Content: t.delta.content, Reasoning: t.delta.reasoning, ToolCalls: t.delta.calls}
	if t.logprob != nil {
		ev.Logprobs = []mwip.LogprobContent{*t.logprob}
	}
	if err := s.event(ev); err != nil {
		return err
	}
	f := t.finish
	if f == nil {
		return nil
	}
	reason, _ := f.stopReason.(string)
	u := f.usage
	return s.event(&mwip.GenerateEvent{Type: mwip.EventFinish, Reason: f.reason, StopReason: reason, Usage: &u, Timing: f.timing})
}

func (s *grpcStream) fail(code, msg string, retriable bool) error {
	_ = s.event(&mwip.GenerateEvent{Type: mwip.EventError, Code: code, Message: msg, Retriable: retriable})
	return fmt.Errorf("%s: %s", code, msg)
}
