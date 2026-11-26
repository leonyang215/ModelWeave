local op,owner,id=ARGV[1],ARGV[2],ARGV[3]
local policy=cjson.decode(ARGV[4])
local data,leases,owners=KEYS[1],KEYS[2],KEYS[3]
local t=redis.call('TIME')
local now=tonumber(t[1])*1000+math.floor(tonumber(t[2])/1000)
local function live(who) return tonumber(redis.call('ZSCORE',owners,who) or '0')>now end
local function reply(code,reason,wait,n,rpm,tpm)
    return cjson.encode({code=code,reason=reason or '',retry_after_ms=wait or 0,inflight=n or 0,rpm_available=rpm or 0,tpm_available=tpm or 0})
end
if not live(owner) then return reply('owner_lost') end
local n=0
local held=redis.call('HGETALL',leases)
for i=1,#held,2 do
    if live(cjson.decode(held[i+1])[1]) then n=n+1
    else redis.call('HDEL',leases,held[i]) end
end
local last=tonumber(redis.call('HGET',data,'at') or now)
local elapsed=math.max(now-last,0)/60000
local bucketAt=math.max(now,last)
local rpm=tonumber(redis.call('HGET',data,'rpm') or policy.rpm)
local tpm=tonumber(redis.call('HGET',data,'tpm') or policy.tpm)
if policy.rpm>0 then rpm=math.min(policy.rpm,rpm+elapsed*policy.rpm) end
if policy.tpm>0 then tpm=math.min(policy.tpm,tpm+elapsed*policy.tpm) end
local function save() redis.call('HSET',data,'at',bucketAt,'rpm',rpm,'tpm',tpm) end
save()
if op=='stats' then return reply('ok','',0,n,rpm,tpm) end
if op=='finish' then
    local raw=redis.call('HGET',leases,id)
    if not raw then return reply('ok') end -- idempotent after an ambiguous result
    local lease=cjson.decode(raw)
    if lease[1]~=owner then return reply('gone') end
    local actual=tonumber(ARGV[6])
    if policy.tpm>0 and actual>=0 then tpm=math.min(policy.tpm,tpm+lease[2]-actual) end
    redis.call('HDEL',leases,id)
    save()
    return reply('ok')
end
if op=='admit' then
    if redis.call('HEXISTS',leases,id)==1 then return reply('ok') end
    if policy.max_concurrency>0 and n>=policy.max_concurrency then return reply('rejected','concurrency',1000) end
    if policy.rpm>0 and rpm<math.min(1,policy.rpm) then
        return reply('rejected','rpm',math.max(1,math.min(86400000,math.ceil((math.min(1,policy.rpm)-rpm)*60000/policy.rpm))))
    end
    local estimate=math.max(tonumber(ARGV[5]),0)
    local need=math.min(estimate,policy.tpm)
    if policy.tpm>0 and tpm<need then
        return reply('rejected','tpm',math.max(1,math.min(86400000,math.ceil((need-tpm)*60000/policy.tpm))))
    end
    -- No charge is made until all limits pass. Oversized requests borrow
    -- from a full bucket and leave debt, exactly like the memory backend.
    if policy.rpm>0 then rpm=rpm-1 end
    if policy.tpm>0 then tpm=tpm-estimate end
    redis.call('HSET',leases,id,cjson.encode({owner,estimate}))
    save()
    return reply('ok')
end
return reply('bad_operation')
