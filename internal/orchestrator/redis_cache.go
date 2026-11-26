package orchestrator

import (
	"crypto/sha256"
	_ "embed"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/redisstate"
)

//go:embed redis_cache.lua
var redisCacheSource string
var cacheScript = redis.NewScript(redisCacheSource)

type sharedCache struct {
	state   *redisstate.Store
	root    string
	ttl     time.Duration
	max     int
	onError func(error)
}
type redisSessions struct{ sharedCache }
type redisArtifacts struct{ sharedCache }

func NewRedisSessionMap(s *redisstate.Store, model string, ttl time.Duration, onError func(error)) SessionStore {
	return &redisSessions{sharedCache{state: s, root: s.Key("sessions:" + base64.RawURLEncoding.EncodeToString([]byte(model)) + ":"), ttl: ttl, onError: onError}}
}
func NewRedisArtifactCache(s *redisstate.Store, model string, ttl time.Duration, max int, onError func(error)) ArtifactStore {
	return &redisArtifacts{sharedCache{state: s, root: s.Key("artifacts:" + base64.RawURLEncoding.EncodeToString([]byte(model)) + ":"), ttl: ttl, max: max, onError: onError}}
}
func cacheDigest(key string) string { h := sha256.Sum256([]byte(key)); return hex.EncodeToString(h[:]) }
func (s *sharedCache) command(op, key, payload string) (string, error) {
	ctx, cancel := s.state.Context()
	defer cancel()
	raw, err := cacheScript.Run(ctx, s.state.Client, []string{s.root + "expiry", s.root + "order", s.root + "seq"}, op, s.root, key, payload, max(s.ttl.Milliseconds(), 1), s.max).Text()
	if errors.Is(err, redis.Nil) {
		return "", nil
	}
	if err != nil && s.onError != nil {
		s.onError(err)
	}
	return raw, err
}
func (s *sharedCache) mutate(op, key, payload string) int {
	ctx, cancel := s.state.Context()
	defer cancel()
	n, err := cacheScript.Run(ctx, s.state.Client, []string{s.root + "expiry", s.root + "order", s.root + "seq"}, op, s.root, key, payload, max(s.ttl.Milliseconds(), 1), s.max).Int()
	if err != nil && s.onError != nil {
		s.onError(err)
	}
	return n
}
func (s *redisSessions) Get(key string) (string, bool) {
	if key == "" {
		return "", false
	}
	v, err := s.command("session_get", cacheDigest(key), "")
	return v, err == nil && v != ""
}
func (s *redisSessions) Set(key, worker string) {
	if key != "" {
		s.mutate("session_set", cacheDigest(key), worker)
	}
}
func (s *redisSessions) Delete(key string) {
	if key != "" {
		s.mutate("session_delete", cacheDigest(key), "")
	}
}
func (s *redisSessions) ForgetWorker(worker string) { s.mutate("session_forget", worker, "") }
func (s *redisSessions) Len() int                   { return s.mutate("session_len", "", "") }

func (s *redisArtifacts) Get(key string) (mwip.EmbeddingLocation, bool) {
	v, err := s.command("artifact_get", cacheDigest(key), "")
	if err != nil || v == "" {
		return mwip.EmbeddingLocation{}, false
	}
	var embedding mwip.EmbeddingLocation
	if err := json.Unmarshal([]byte(v), &embedding); err != nil {
		if s.onError != nil {
			s.onError(err)
		}
		return mwip.EmbeddingLocation{}, false
	}
	return embedding, true
}
func (s *redisArtifacts) Put(key string, embedding mwip.EmbeddingLocation) {
	b, err := json.Marshal(embedding)
	if err == nil {
		s.mutate("artifact_put", cacheDigest(key), string(b))
	}
}
func (s *redisArtifacts) Invalidate(keys ...string) {
	for _, key := range keys {
		s.mutate("artifact_delete", cacheDigest(key), "")
	}
}
func (s *redisArtifacts) Len() int { return s.mutate("artifact_len", "", "") }
