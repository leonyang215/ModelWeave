-- All keys belong to one namespace/hash slot. Redis TIME is the clock for
-- liveness, state age, health cooldowns and gateway reservation leases.
local op, root, owner, id = ARGV[1], ARGV[2], ARGV[3], ARGV[4]
local members, version, epochs, leases, liveness = KEYS[1], KEYS[2], KEYS[3], KEYS[4], KEYS[5]
local data, claims = KEYS[6], KEYS[7]
-- Deliberately separate from worker records: deregistration/expiry must not
-- undo an operator block. All keys remain in this namespace's hash slot.
local blacklist = root..'blacklist'
local t = redis.call('TIME')
local now = tonumber(t[1])*1000 + math.floor(tonumber(t[2])/1000)
local function number(key, field)
    return tonumber(redis.call('HGET', key, field) or '0')
end
local function ver() return tonumber(redis.call('GET', version) or '0') end
local function current() return redis.call('HGET', data, 'epoch') == ARGV[5] end
local function live(who)
    return tonumber(redis.call('ZSCORE', leases, who) or '0') > now
end
local function accounting(who)
    local s = redis.call('HGET', data, 'r:'..who)
    if s then return cjson.decode(s) end
    return {0,0,0,0} -- requests, KV, prefill, KV pending since the last report
end
local function save(who, a)
    if a[1] == 0 and a[2] == 0 and a[3] == 0 and a[4] == 0 then
        redis.call('HDEL', data, 'r:'..who)
    else redis.call('HSET', data, 'r:'..who, cjson.encode(a)) end
end
local function totals()
    local out, dead = {0,0,0,0}, {}
    local fields = redis.call('HGETALL', data)
    for i=1,#fields,2 do
        if string.sub(fields[i],1,2) == 'r:' then
            local who = string.sub(fields[i],3)
            if live(who) then
                local a = cjson.decode(fields[i+1])
                for j=1,4 do out[j] = out[j]+a[j] end
            else
                dead[who] = true
                redis.call('HDEL', data, fields[i])
            end
        end
    end
    -- Reclaim crashed replicas lazily, once per affected instance. No
    -- process-local cleanup or reliable Pub/Sub delivery is required.
    if next(dead) ~= nil then
        local held = redis.call('HGETALL', claims)
        for i=1,#held,2 do
            if dead[cjson.decode(held[i+1])[1]] then redis.call('HDEL', claims, held[i]) end
        end
    end
    return out
end
local function report(state)
    redis.call('HSET', data, 'state', state, 'state_at', now)
    redis.call('HINCRBY', data, 'state_ver', 1)
    redis.call('HINCRBY', data, 'revision', 1)
    local fields = redis.call('HGETALL', data)
    for i=1,#fields,2 do
        if string.sub(fields[i],1,2) == 'r:' then
            local a = cjson.decode(fields[i+1]); a[4] = 0
            save(string.sub(fields[i],3), a)
        end
    end
end
local function remove(worker)
    local entry = redis.call('HGET', members, worker)
    redis.call('HDEL', members, worker)
    redis.call('ZREM', liveness, worker)
    redis.call('DEL', root..'worker:'..worker, root..'claims:'..worker)
    return entry
end

if op == 'join' then
    redis.call('ZREMRANGEBYSCORE', leases, '-inf', now)
    redis.call('ZADD', leases, now+tonumber(ARGV[5]), owner)
    return 1
elseif op == 'maintain' then
    if not live(owner) then return -3 end -- expired owners cannot resurrect
    redis.call('ZADD', leases, now+tonumber(ARGV[5]), owner)
    return 1
elseif op == 'leave' then
    redis.call('ZREM', leases, owner)
    return 1
elseif op == 'claim' then
    local ok = redis.call('SET', root..'poll:'..id, owner, 'NX', 'PX', ARGV[5])
    if ok then return 1 end
    return 0
elseif op == 'members' then
    local v = ver()
    local out = {tostring(v)}
    if v ~= tonumber(ARGV[5]) then
        local entries = redis.call('HVALS', members)
        for i=1,#entries do out[#out+1] = entries[i] end
    end
    return out
elseif op == 'blocks' then
    return redis.call('HVALS', blacklist)
elseif op == 'block' then
    local old = redis.call('HGET', blacklist, id)
    if old and cjson.decode(old).reason == ARGV[5] then return old end
    local b = cjson.encode({id=id,reason=ARGV[5],blocked_at_ms=now})
    redis.call('HSET', blacklist, id, b)
    if redis.call('HEXISTS', data, 'epoch') == 1 then redis.call('HINCRBY', data, 'revision', 1) end
    return b
elseif op == 'unblock' then
    if redis.call('HDEL', blacklist, id) == 1 and redis.call('HEXISTS', data, 'epoch') == 1 then
        redis.call('HINCRBY', data, 'revision', 1)
    end
    return 1
elseif op == 'upsert' then
    local spec, source = ARGV[5], ARGV[6]
    local epoch = redis.call('HGET', data, 'epoch')
    local created, changed = not epoch, false
    local oldSource, oldSpec = redis.call('HGET', data, 'source'), redis.call('HGET', data, 'spec')
    if oldSource == 'static' and source ~= 'static' then
        spec, source = oldSpec, oldSource
    end
    if created or spec ~= oldSpec or source ~= oldSource then
        changed = not created
        local identityChanged = false
        if oldSpec and spec ~= oldSpec then
            local a,b = cjson.decode(oldSpec), cjson.decode(spec)
            identityChanged = a.target ~= b.target or a.model ~= b.model or a.role ~= b.role
        end
        if created or identityChanged then
            local drain = redis.call('HGET', data, 'admin_drain') or '0'
            redis.call('DEL', data, claims)
            epoch = tostring(redis.call('INCR', epochs))
            redis.call('HSET', data, 'admin_drain', drain)
        end
        local entry = cjson.encode({spec=cjson.decode(spec),source=source,epoch=tonumber(epoch)})
        redis.call('HSET', data, 'epoch', epoch, 'spec', spec, 'source', source, 'entry', entry)
        redis.call('HINCRBY', data, 'revision', 1)
        redis.call('HSET', members, id, entry)
        redis.call('INCR', version)
    end
    redis.call('HSET', data, 'last_seen', now)
    if source == 'heartbeat' then redis.call('ZADD', liveness, now, id)
    else redis.call('ZREM', liveness, id) end
    if ARGV[7] ~= '' then report(ARGV[7]) end
    if ARGV[8] ~= '' and redis.call('HGET', data, 'engine_drain') ~= ARGV[8] then
        redis.call('HSET', data, 'engine_drain', ARGV[8])
        redis.call('HINCRBY', data, 'revision', 1)
    end
    return cjson.encode({entry=cjson.decode(redis.call('HGET', data, 'entry')),created=created,changed=changed,version=ver()})
elseif op == 'remove' then
    local e = remove(id)
    if e then redis.call('INCR', version) end
    return {tostring(ver()), e or ''}
elseif op == 'expire' then
    local expired = redis.call('ZRANGEBYSCORE', liveness, '-inf', '('..tostring(now-tonumber(ARGV[5])))
    local out = {''}
    for _,worker in ipairs(expired) do
        local e = remove(worker)
        if e then out[#out+1] = e end
    end
    if #out > 1 then redis.call('INCR', version) end
    out[1] = tostring(ver())
    return out
elseif op == 'load' then
    if not current() then return cjson.encode({gone=true}) end
    local a = totals()
    local state = redis.call('HGET', data, 'state')
    local block = redis.call('HGET', blacklist, id)
    return cjson.encode({
        entry=cjson.decode(redis.call('HGET', data, 'entry')), now=now,
        last_seen=number(data,'last_seen'), state_at=number(data,'state_at'),
        state=state and cjson.decode(state) or cjson.null,
        state_ver=number(data,'state_ver'), revision=number(data,'revision'),
        failures=number(data,'failures'), unhealthy_till=number(data,'unhealthy_till'),
        admin_drain=number(data,'admin_drain') == 1, engine_drain=number(data,'engine_drain') == 1,
        block=block and cjson.decode(block) or cjson.null,
        inflight=a[1], reserved_kv=a[2], prefill=a[3], pending_kv=a[4]
    })
elseif op == 'state' then
    if not current() then return -1 end
    if tonumber(ARGV[7]) >= 0 and number(data,'revision') ~= tonumber(ARGV[7]) then return -2 end
    report(ARGV[6])
    return 1
elseif op == 'drain' then
    if not current() then return -1 end
    local field = ARGV[6] == '0' and 'admin_drain' or 'engine_drain'
    if redis.call('HGET', data, field) ~= ARGV[7] then
        redis.call('HSET', data, field, ARGV[7])
        redis.call('HINCRBY', data, 'revision', 1)
    end
    return 1
elseif op == 'reserve' then
    if not live(owner) then return -3 end
    if ARGV[14] ~= '' and (redis.call('HGET',KEYS[8],'turn') ~= ARGV[14] or redis.call('HGET',KEYS[8],'token') ~= ARGV[15]) then return -4 end
    if not current() then return -1 end
    if redis.call('HEXISTS', claims, ARGV[6]) == 1 then return 1 end
    if redis.call('HEXISTS', blacklist, id) == 1 then return -2 end
    local revision = tonumber(ARGV[9])
    if revision >= 0 and number(data,'revision') ~= revision then return -2 end
    local a = totals()
    -- Bounds are on the totals before adding the new claim.
    for i=1,4 do if a[i] > tonumber(ARGV[9+i]) then return -2 end end
    local kv, prefill = tonumber(ARGV[7]), tonumber(ARGV[8])
    local own = accounting(owner)
    own[1],own[2],own[3],own[4] = own[1]+1,own[2]+kv,own[3]+prefill,own[4]+kv
    save(owner, own)
    redis.call('HSET', claims, ARGV[6], cjson.encode({owner,kv,prefill}))
    return 1
elseif op == 'release' then
    if not current() then return 1 end
    local raw = redis.call('HGET', claims, ARGV[6])
    if not raw then return 1 end
    local held = cjson.decode(raw)
    if held[1] ~= owner then return -1 end
    local own = accounting(owner)
    own[3] = math.max(0,own[3]-held[3])
    if ARGV[7] == '1' then
        own[1],own[2] = math.max(0,own[1]-1),math.max(0,own[2]-held[2])
        redis.call('HDEL', claims, ARGV[6])
    else
        held[3] = 0
        redis.call('HSET', claims, ARGV[6], cjson.encode(held))
    end
    save(owner, own)
    return 1
elseif op == 'health' then
    if not current() then return 0 end
    local failures = number(data,'failures')
    if ARGV[6] == '0' then
        if failures ~= 0 then
            redis.call('HSET',data,'failures',0)
            redis.call('HINCRBY',data,'revision',1)
        end
        return 0
    end
    failures = failures+1
    redis.call('HINCRBY',data,'revision',1)
    if tonumber(ARGV[7]) > 0 and failures >= tonumber(ARGV[7]) then
        redis.call('HSET',data,'failures',0,'unhealthy_till',now+tonumber(ARGV[8]))
        return 1
    end
    redis.call('HSET',data,'failures',failures)
    return 0
end
return redis.error_reply('unknown state operation')
