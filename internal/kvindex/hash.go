// Package kvindex tracks which workers hold which prompt prefixes in their KV
// cache, so the scheduler can route a request to the worker that already has
// most of its prefix cached.
//
// Prompts are split into fixed-size token blocks. Each block is identified by a
// chained hash: the hash of block i covers the hash of block i-1 plus the
// block's own tokens, so equal block hashes imply equal prefixes. Only full
// blocks are hashed; the partial tail of a prompt is never cached by engines
// either.
package kvindex

import (
	"encoding/binary"

	"github.com/zeebo/xxh3"
)

// DefaultBlockSize is the number of tokens per hashed block.
const DefaultBlockSize = 64

// BlockHashes returns the chained xxh3-64 hashes of the full blocks of tokens.
//
// h(-1) = seed
// h(i)  = xxh3_64( le_u64(h(i-1)) || le_u32(tok[i*B]) || ... || le_u32(tok[i*B+B-1]) )
//
// The algorithm is part of the MWIP specification (docs/protocol.md): engines
// that publish KV events must hash blocks the same way.
func BlockHashes(tokens []uint32, blockSize int, seed uint64) []uint64 {
	if blockSize <= 0 {
		blockSize = DefaultBlockSize
	}
	n := len(tokens) / blockSize
	if n == 0 {
		return nil
	}
	out := make([]uint64, n)
	buf := make([]byte, 8+4*blockSize)
	prev := seed
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint64(buf[0:8], prev)
		block := tokens[i*blockSize : (i+1)*blockSize]
		for j, t := range block {
			binary.LittleEndian.PutUint32(buf[8+4*j:], t)
		}
		prev = xxh3.Hash(buf)
		out[i] = prev
	}
	return out
}

// ModelSeed derives the hash seed for a model so that identical token
// sequences of different models never share cache entries.
func ModelSeed(model string) uint64 {
	return xxh3.HashString(model)
}
