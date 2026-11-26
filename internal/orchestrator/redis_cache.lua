local op,root,key = ARGV[1],ARGV[2],ARGV[3]
local t=redis.call('TIME')
local now=tonumber(t[1])*1000+math.floor(tonumber(t[2])/1000)
local value=root..'value:'..key
local function drop(k)
    redis.call('DEL',root..'value:'..k)
    redis.call('ZREM',KEYS[1],k)
    redis.call('ZREM',KEYS[2],k)
end
local function sweep()
    local expired=redis.call('ZRANGEBYSCORE',KEYS[1],'-inf',now)
    for _,k in ipairs(expired) do drop(k) end
end
if op=='session_get' then
    local v=redis.call('GET',value)
    if not v then redis.call('ZREM',KEYS[1],key) end
    return v
elseif op=='session_set' then
    local worker,ttl=ARGV[4],tonumber(ARGV[5])
    local old=redis.call('GET',value)
    if old and old~=worker then redis.call('SREM',root..'worker:'..old,key) end
    redis.call('SET',value,worker,'PX',ttl)
    redis.call('ZADD',KEYS[1],now+ttl,key)
    redis.call('SADD',root..'worker:'..worker,key)
    redis.call('PEXPIRE',root..'worker:'..worker,ttl)
elseif op=='session_delete' then
    local old=redis.call('GET',value)
    if old then redis.call('SREM',root..'worker:'..old,key) end
    redis.call('DEL',value)
    redis.call('ZREM',KEYS[1],key)
elseif op=='session_forget' then
    local sessions=redis.call('SMEMBERS',root..'worker:'..key)
    for _,k in ipairs(sessions) do
        if redis.call('GET',root..'value:'..k)==key then
            redis.call('DEL',root..'value:'..k)
            redis.call('ZREM',KEYS[1],k)
        end
    end
    redis.call('DEL',root..'worker:'..key)
elseif op=='session_len' then
    redis.call('ZREMRANGEBYSCORE',KEYS[1],'-inf',now)
    return redis.call('ZCARD',KEYS[1])
elseif op=='artifact_get' then
    local v=redis.call('GET',value)
    if not v then drop(key); return false end
    redis.call('ZADD',KEYS[2],redis.call('INCR',KEYS[3]),key)
    return v
elseif op=='artifact_put' then
    sweep()
    local ttl,cap=tonumber(ARGV[5]),tonumber(ARGV[6])
    redis.call('SET',value,ARGV[4],'PX',ttl)
    redis.call('ZADD',KEYS[1],now+ttl,key)
    redis.call('ZADD',KEYS[2],redis.call('INCR',KEYS[3]),key)
    local n=redis.call('ZCARD',KEYS[2])-cap
    if cap>0 and n>0 then
        for _,k in ipairs(redis.call('ZRANGE',KEYS[2],0,n-1)) do drop(k) end
    end
elseif op=='artifact_delete' then drop(key)
elseif op=='artifact_len' then sweep(); return redis.call('ZCARD',KEYS[1])
end
return 1
