package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"time"

	"github.com/leonyang215/ModelWeave/internal/engine"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/orchestrator"
	"github.com/leonyang215/ModelWeave/internal/qos"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
	"github.com/leonyang215/ModelWeave/internal/tokenizer"
)

// apiError is an error returned to API clients.
type apiError struct {
	Status     int
	Kind       string // OpenAI error type / code
	Message    string
	RetryAfter time.Duration
	Body       json.RawMessage // pre-shaped body (tokenizer errors)
}

func (e *apiError) Error() string { return fmt.Sprintf("%d %s: %s", e.Status, e.Kind, e.Message) }

func anthropicType(status int, kind string) string {
	switch {
	case status == http.StatusBadRequest:
		return "invalid_request_error"
	case status == http.StatusUnauthorized:
		return "authentication_error"
	case status == http.StatusNotFound:
		return "not_found_error"
	case status == http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case status == http.StatusTooManyRequests:
		return "rate_limit_error"
	case status == http.StatusServiceUnavailable || kind == "overloaded":
		return "overloaded_error"
	case status == http.StatusGatewayTimeout:
		return "timeout_error"
	}
	return "api_error"
}

// body renders the error in the request's protocol.
func (e *apiError) body(protocol string) []byte {
	if len(e.Body) > 0 {
		return e.Body
	}
	var v any
	if protocol == mwip.ProtocolMessages {
		v = map[string]any{"type": "error", "error": map[string]any{"type": anthropicType(e.Status, e.Kind), "message": e.Message}}
	} else {
		v = map[string]any{"error": map[string]any{"message": e.Message, "type": e.Kind, "param": nil, "code": e.Kind}}
	}
	b, _ := json.Marshal(v)
	return b
}

func writeError(w http.ResponseWriter, protocol string, e *apiError) {
	h := w.Header()
	h.Set("Content-Type", "application/json")
	if e.RetryAfter > 0 {
		h.Set("Retry-After", strconv.Itoa(int(math.Ceil(e.RetryAfter.Seconds()))))
	}
	w.WriteHeader(e.Status)
	_, _ = w.Write(e.body(protocol))
}

func badRequest(msg string) *apiError {
	return &apiError{Status: http.StatusBadRequest, Kind: "invalid_request_error", Message: msg}
}

// classify maps internal failures to API errors.
func classify(err error) *apiError {
	var te *tokenizer.Error
	if errors.As(err, &te) {
		if te.Status > 0 {
			return &apiError{Status: te.Status, Kind: "invalid_request_error", Message: string(te.Body), Body: te.Body}
		}
		return &apiError{Status: http.StatusBadGateway, Kind: "tokenizer_unavailable", Message: te.Error()}
	}
	var rj *qos.Rejection
	if errors.As(err, &rj) {
		return &apiError{Status: http.StatusTooManyRequests, Kind: "rate_limit_exceeded", Message: rj.Error(), RetryAfter: max(rj.RetryAfter, time.Second)}
	}
	switch {
	case errors.Is(err, qos.ErrQueueFull), errors.Is(err, qos.ErrShed):
		return &apiError{Status: http.StatusServiceUnavailable, Kind: "overloaded", Message: err.Error(), RetryAfter: 2 * time.Second}
	case errors.Is(err, qos.ErrTimeout):
		return &apiError{Status: http.StatusServiceUnavailable, Kind: "overloaded", Message: "no capacity became available in time", RetryAfter: 5 * time.Second}
	case errors.Is(err, scheduler.ErrNoCapacity):
		return &apiError{Status: http.StatusServiceUnavailable, Kind: "overloaded", Message: err.Error(), RetryAfter: 2 * time.Second}
	case errors.Is(err, scheduler.ErrNoWorkers), errors.Is(err, orchestrator.ErrNoRoute), errors.Is(err, orchestrator.ErrEncodeUnavailable):
		return &apiError{Status: http.StatusServiceUnavailable, Kind: "no_worker_available", Message: err.Error(), RetryAfter: 5 * time.Second}
	case errors.Is(err, orchestrator.ErrFirstTokenTimeout):
		return &apiError{Status: http.StatusGatewayTimeout, Kind: "timeout", Message: err.Error()}
	case errors.Is(err, orchestrator.ErrIdleTimeout):
		return &apiError{Status: http.StatusGatewayTimeout, Kind: "timeout", Message: err.Error()}
	}
	var ee *engine.Error
	if errors.As(err, &ee) {
		switch ee.Code {
		case mwip.ErrBadRequest:
			return &apiError{Status: http.StatusBadRequest, Kind: "invalid_request_error", Message: ee.Message}
		case mwip.ErrCapacity:
			return &apiError{Status: http.StatusServiceUnavailable, Kind: "overloaded", Message: ee.Error(), RetryAfter: 2 * time.Second}
		case mwip.ErrTimeout:
			return &apiError{Status: http.StatusGatewayTimeout, Kind: "timeout", Message: ee.Error()}
		}
		return &apiError{Status: http.StatusBadGateway, Kind: "upstream_error", Message: ee.Error()}
	}
	return &apiError{Status: http.StatusInternalServerError, Kind: "internal_error", Message: err.Error()}
}
