// Package admission carries an internal queue turn to resource reservation.
// Redis checks the turn in the same script that reserves worker capacity, so
// a canceled/expired turn cannot race a later turn into an engine assignment.
package admission

import (
	"context"
	"errors"
)

var ErrTurnLost = errors.New("admission queue turn is no longer valid")

type Permit struct {
	Control string // Redis control hash (same namespace/hash slot as workers)
	Ticket  string
	Token   string
}
type contextKey struct{}

func WithPermit(ctx context.Context, p Permit) context.Context {
	return context.WithValue(ctx, contextKey{}, p)
}
func FromContext(ctx context.Context) Permit { p, _ := ctx.Value(contextKey{}).(Permit); return p }
