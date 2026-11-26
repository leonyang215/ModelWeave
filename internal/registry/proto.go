package registry

import mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"

// StateToProto converts an engine state report to its mwip form.
func StateToProto(s EngineState) *mwipv1.EngineState {
	return &mwipv1.EngineState{
		RunningRequests: int32(s.RunningRequests), WaitingRequests: int32(s.WaitingRequests),
		KvUsedTokens: s.KVUsedTokens, KvCapacityTokens: s.KVCapacityTokens, KvUsage: s.KVUsage,
		PrefillQueueTokens: s.PrefillQueueTokens, MaxRunningRequests: int32(s.MaxRunningRequests),
		DecodeTokensPerSec: s.DecodeTokensPerSec, PrefillTokensPerSec: s.PrefillTokensPerSec,
		CachedBlocks: s.CachedBlocks, Draining: s.Draining,
	}
}

// StateFromProto converts an mwip engine state report.
func StateFromProto(p *mwipv1.EngineState) EngineState {
	return EngineState{
		RunningRequests: int(p.GetRunningRequests()), WaitingRequests: int(p.GetWaitingRequests()),
		KVUsedTokens: p.GetKvUsedTokens(), KVCapacityTokens: p.GetKvCapacityTokens(), KVUsage: p.GetKvUsage(),
		PrefillQueueTokens: p.GetPrefillQueueTokens(), MaxRunningRequests: int(p.GetMaxRunningRequests()),
		DecodeTokensPerSec: p.GetDecodeTokensPerSec(), PrefillTokensPerSec: p.GetPrefillTokensPerSec(),
		CachedBlocks: p.GetCachedBlocks(), Draining: p.GetDraining(),
	}
}
