-- Hashes are decimal strings: Lua doubles must never round uint64 hashes.
local op = ARGV[1]
local seen, order, meta = KEYS[1], KEYS[2], KEYS[3]
local t = redis.call('TIME')
local now = tonumber(t[1])*1000+math.floor(tonumber(t[2])/1000)
local function remove(hashes, s, o)
    for _,h in ipairs(hashes) do redis.call('ZREM',s,h); redis.call('ZREM',o,h) end
end
local function sweep(ttl, s, o)
    if ttl > 0 then remove(redis.call('ZRANGEBYSCORE',s,'-inf','('..tostring(now-ttl)),s,o) end
end
local function insert(first, last, ttl, defaultCap, s, o, m)
    if first > last then return end
    sweep(ttl,s,o)
    local seq = tonumber(redis.call('HGET',m,'seq') or '0')
    for i=last,first,-1 do
        seq = seq+1
        redis.call('ZADD',s,now,ARGV[i])
        redis.call('ZADD',o,seq,ARGV[i])
    end
    redis.call('HSET',m,'seq',seq)
    local cap = tonumber(redis.call('HGET',m,'capacity') or defaultCap)
    if cap > 0 then
        local n = redis.call('ZCARD',o)-cap
        if n > 0 then remove(redis.call('ZRANGE',o,0,n-1),s,o) end
    end
    if ttl > 0 then
        redis.call('PEXPIRE',s,ttl*2)
        redis.call('PEXPIRE',o,ttl*2)
        if redis.call('HEXISTS',m,'capacity') == 1 then redis.call('PERSIST',m)
        else redis.call('PEXPIRE',m,ttl*2) end
    end
end
if op == 'match' then
    local ttl, n = tonumber(ARGV[2]), 0
    for i=3,#ARGV do
        local at = redis.call('ZSCORE',seen,ARGV[i])
        if not at or (ttl > 0 and tonumber(at) < now-ttl) then break end
        n=n+1
    end
    return n
elseif op == 'observe' then
    insert(4,#ARGV,tonumber(ARGV[2]),ARGV[3],seen,order,meta)
elseif op == 'capacity' then
    redis.call('HSET',meta,'capacity',ARGV[2])
    local n = redis.call('ZCARD',order)-tonumber(ARGV[2])
    if tonumber(ARGV[2]) > 0 and n > 0 then remove(redis.call('ZRANGE',order,0,n-1),seen,order) end
elseif op == 'events' then
    local os,oo,om = KEYS[4],KEYS[5],KEYS[6]
    if ARGV[2] == '1' then redis.call('DEL',seen,order,meta,os,oo,om) end
    local stored,removed = tonumber(ARGV[4]),tonumber(ARGV[5])
    for i=6+stored,5+stored+removed do
        remove({ARGV[i]},seen,order); remove({ARGV[i]},os,oo)
    end
    insert(6,5+stored,0,'0',seen,order,meta)
elseif op == 'stats' then
    sweep(tonumber(ARGV[2]),seen,order)
    return redis.call('ZCARD',seen)
elseif op == 'forget' then
    for _,k in ipairs(KEYS) do redis.call('DEL',k) end
end
return 1
