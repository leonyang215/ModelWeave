-- A queue is shared by canonical model name. Only ticket metadata crosses
-- gateways; the owning HTTP goroutine executes its own fenced scheduling turn.
local op,owner,id=ARGV[1],ARGV[2],ARGV[3]
local cfg=cjson.decode(ARGV[4])
local tickets,waiting,control,vtc,skipped,owners=KEYS[1],KEYS[2],KEYS[3],KEYS[4],KEYS[5],KEYS[6]
local t=redis.call('TIME')
local now=tonumber(t[1])*1000+math.floor(tonumber(t[2])/1000)
local function live(who) return tonumber(redis.call('ZSCORE',owners,who) or '0')>now end
if not live(owner) then return cjson.encode({code='owner_lost'}) end
local entries={}
local changed=false
local function load(ticket)
    if entries[ticket] then return entries[ticket] end
    local raw=redis.call('HGET',tickets,ticket)
    if raw then entries[ticket]=cjson.decode(raw) end
    return entries[ticket]
end
local function save(e) entries[e.id]=e;redis.call('HSET',tickets,e.id,cjson.encode(e));changed=true end
local function clearTurn(ticket)
    if redis.call('HGET',control,'turn')==ticket then redis.call('HDEL',control,'turn','token');changed=true end
end
local function drop(ticket)
    clearTurn(ticket)
    redis.call('ZREM',waiting,ticket)
    redis.call('SREM',skipped,ticket)
    redis.call('HDEL',tickets,ticket)
    entries[ticket]=nil
    changed=true
end
local function terminal(e,state)
    clearTurn(e.id)
    redis.call('ZREM',waiting,e.id)
    redis.call('SREM',skipped,e.id)
    e.state=state;e.retain_until=now+math.max(15000,cfg.tick_ms*4)
    save(e)
end
local function active(e) return e.state=='waiting' or (e.state=='turn' and not e.shed_pending) end
local all=redis.call('HGETALL',tickets)
for i=1,#all,2 do
    local e=cjson.decode(all[i+1]);entries[e.id]=e
    if not live(e.owner) then drop(e.id)
    elseif e.state=='waiting' or e.state=='turn' then
        if now>=e.deadline then terminal(e,'timeout') end
    elseif now>=(e.retain_until or 0) then drop(e.id) end
end
local function resetRound()
    redis.call('DEL',skipped)
    redis.call('HSET',control,'failures',0,'next_at',0)
end
local function counts()
    local n,depth=0,{0,0,0,0}
    for _,e in pairs(entries) do if active(e) then n=n+1;depth[e.priority+1]=depth[e.priority+1]+1 end end
    return n,depth
end
local function reply(code,e,includeOwn)
    local n,depth=counts()
    local own={}
    if includeOwn then
        for ticket,entry in pairs(entries) do
            if entry.owner==owner then own[ticket]={state=entry.state,token=entry.token or '',queued=entry.queued} end
        end
    end
    if next(own)==nil then own=cjson.null end
    return cjson.encode({code=code or 'ok',changed=changed,total=n,depth=depth,own=own,
        state=e and e.state or '',token=e and e.token or '',queued=e and e.queued or false})
end
local function take(e)
    redis.call('ZREM',waiting,e.id)
    local token=tostring(redis.call('HINCRBY',control,'fence',1))
    redis.call('HSET',control,'turn',e.id,'token',token)
    e.state='turn';e.token=token
    e.wake=tonumber(redis.call('HGET',control,'wake') or '0')
    save(e)
end
local function shed(priority)
    -- Newest waiting request in the lowest lower-priority class, or that
    -- class's current turn. A shed in-turn attempt may still succeed; its
    -- place is transferred, but its fencing token is not revoked early.
    for p=3,priority+1,-1 do
        local victim=nil
        for _,e in pairs(entries) do
            if e.state=='waiting' and e.priority==p and (not victim or e.seq>victim.seq) then victim=e end
        end
        if victim then terminal(victim,'shed');return true end
        local current=redis.call('HGET',control,'turn')
        local e=current and load(current)
        if e and e.priority==p and not e.shed_pending then e.shed_pending=true;save(e);return true end
    end
    return false
end
local function lift(e)
    local same,floor=false,nil
    for _,other in pairs(entries) do
        if active(other) and other.priority==e.priority then
            if other.tenant==e.tenant then same=true
            else
                local v=tonumber(redis.call('HGET',vtc,other.tenant) or '0')
                if not floor or v<floor then floor=v end
            end
        end
    end
    if not same and floor then
        local v=tonumber(redis.call('HGET',vtc,e.tenant) or '0')
        if v<floor then redis.call('HSET',vtc,e.tenant,floor) end
    end
end
if op=='enqueue' then
    local existing=load(id)
    if existing then return reply('ok',existing,false) end
    local input=cjson.decode(ARGV[5])
    local n=counts()
    if n>=cfg.max_size and not shed(input.priority) then return reply('full') end
    local seq=redis.call('HINCRBY',control,'seq',1)
    local fast=n==0 and not redis.call('HGET',control,'turn')
    local e={id=id,owner=owner,tenant=input.tenant,priority=input.priority,weight=input.weight,cost=input.cost,
        seq=seq,deadline=now+tonumber(ARGV[6]),state='waiting',queued=not fast,shed_pending=false}
    lift(e)
    save(e)
    redis.call('HINCRBY',control,'wake',1)
    resetRound()
    if fast then take(e) else redis.call('ZADD',waiting,seq,id) end
    return reply('ok',e,false)
elseif op=='cancel' then
    local e=load(id)
    if e and e.owner==owner then drop(id);resetRound() end
    return reply('ok')
elseif op=='drop_owner' then
    for ticket,e in pairs(entries) do if e.owner==owner then drop(ticket) end end
    resetRound()
    return reply('ok')
elseif op=='finish' then
    local e=load(id)
    if not e then return reply('gone') end
    if e.owner~=owner then return reply('gone') end
    if e.state=='timeout' or e.state=='shed' then return reply(e.state,e,false) end
    local token,outcome=ARGV[5],ARGV[6]
    if e.state=='done' and e.token==token then return reply('ok',e,false) end
    if e.state~='turn' or e.token~=token or redis.call('HGET',control,'turn')~=id or redis.call('HGET',control,'token')~=token then return reply('turn_lost',e,false) end
    clearTurn(id)
    if outcome=='granted' then
        local v=tonumber(redis.call('HGET',vtc,e.tenant) or '0')
        redis.call('HSET',vtc,e.tenant,v+e.cost/e.weight)
        terminal(e,'done')
    elseif outcome=='retry' then
        if e.shed_pending then terminal(e,'shed')
        else e.state='waiting';e.queued=true;save(e);redis.call('ZADD',waiting,e.seq,id) end
        redis.call('SADD',skipped,id)
        -- A capacity notification received WHILE this attempt was running
        -- must survive its failed result. Otherwise finish could overwrite
        -- wake's reset with a long retry delay (a lost-wakeup race).
        local wake=tonumber(redis.call('HGET',control,'wake') or '0')
        if wake>(e.wake or 0) then resetRound()
        else
            local failures=redis.call('HINCRBY',control,'failures',1)
            if failures>cfg.backfill_depth then redis.call('HSET',control,'next_at',now+cfg.tick_ms) end
        end
    else terminal(e,'error') end
    return reply('ok',e,false)
elseif op=='stats' then return reply('ok')
elseif op=='wake' then redis.call('HINCRBY',control,'wake',1);resetRound();return reply('ok')
elseif op=='step' then
    local current=redis.call('HGET',control,'turn')
    if current and not load(current) then clearTurn(current);current=nil end
    if not current then
        local nextAt=tonumber(redis.call('HGET',control,'next_at') or '0')
        if nextAt>0 and now>=nextAt then resetRound();nextAt=0 end
        if nextAt==0 then
            local ids=redis.call('ZRANGE',waiting,0,-1)
            local best,bestV=nil,nil
            for _,ticket in ipairs(ids) do
                local e=load(ticket)
                if e and e.state=='waiting' and redis.call('SISMEMBER',skipped,ticket)==0 then
                    local v=tonumber(redis.call('HGET',vtc,e.tenant) or '0')
                    if not best or e.priority<best.priority or (e.priority==best.priority and (v<bestV or (v==bestV and e.seq<best.seq))) then best,bestV=e,v end
                end
            end
            if best then take(best)
            elseif #ids>0 then redis.call('HSET',control,'next_at',now+cfg.tick_ms) end
        end
    end
    return reply('ok',nil,true)
end
return reply('bad_operation')
