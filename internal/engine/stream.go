package engine

import (
	"context"
	"errors"
	"io"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// Stream is an open Generate call, read as gateway events: accepted (push
// D), prefill_done (end of a prefill leg), delta, finish and error.
type Stream struct {
	role, worker string
	prefill      bool
	done         bool
	rpc          mwipv1.Engine_GenerateClient
	cancel       context.CancelFunc
}

// Next returns the next event. An Error event is returned with an *Error,
// a broken stream as a transport *Error, and io.EOF follows the final
// event. Deltas of a prefill leg are dropped, and its finish is
// prefill_done.
func (s *Stream) Next() (*mwip.GenerateEvent, error) {
	for {
		if s.done {
			return nil, io.EOF
		}
		msg, err := s.rpc.Recv()
		if err != nil {
			s.done = true
			if errors.Is(err, io.EOF) {
				return nil, &Error{Role: s.role, WorkerID: s.worker, Code: mwip.ErrInternal, Message: "stream ended without finish", Retriable: true, Transport: true}
			}
			return nil, statusErr(s.role, s.worker, err)
		}
		ev := mwip.EventFromProto(msg)
		if ev == nil {
			continue // an event this gateway does not know
		}
		switch ev.Type {
		case mwip.EventAccepted:
			if ev.Receiver != nil && ev.Receiver.WorkerID == "" {
				ev.Receiver.WorkerID = s.worker
			}
		case mwip.EventDelta:
			if s.prefill {
				continue
			}
		case mwip.EventError:
			s.done = true
			ev.WorkerID = s.worker
			return ev, &Error{Role: s.role, WorkerID: s.worker, Code: ev.Code, Message: ev.Message, Retriable: ev.Retriable}
		case mwip.EventFinish:
			s.done = true
			if s.prefill {
				ev.Type = mwip.EventPrefillDone
			}
		case mwip.EventPrefillDone:
			s.done = true
		}
		return ev, nil
	}
}

// Close releases the stream; a call still running is canceled.
func (s *Stream) Close() {
	if s.cancel != nil {
		s.cancel()
	}
}
