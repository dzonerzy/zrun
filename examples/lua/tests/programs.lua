-- A JSON encoder and decoder
local json = {}
local function kind(v)
  if type(v) ~= "table" then return type(v) end
  local n = 0
  for _ in pairs(v) do n = n + 1 end
  return (n == #v) and "array" or "object"
end
local escapes = {['"'] = '\\"', ["\\"] = "\\\\", ["\n"] = "\\n", ["\t"] = "\\t"}
function json.encode(v)
  local k = kind(v)
  if k == "nil" then return "null"
  elseif k == "boolean" then return tostring(v)
  elseif k == "number" then
    if math.type(v) == "integer" then return tostring(v) end
    return string.format("%.14g", v)
  elseif k == "string" then return '"' .. v:gsub('[%c"\\]', escapes) .. '"'
  elseif k == "array" then
    local parts = {}
    for i = 1, #v do parts[i] = json.encode(v[i]) end
    return "[" .. table.concat(parts, ",") .. "]"
  else
    local keys = {}
    for key in pairs(v) do keys[#keys + 1] = key end
    table.sort(keys)
    local parts = {}
    for _, key in ipairs(keys) do parts[#parts + 1] = json.encode(key) .. ":" .. json.encode(v[key]) end
    return "{" .. table.concat(parts, ",") .. "}"
  end
end
function json.decode(s)
  local pos = 1
  local function ws() pos = s:find("[^ \t\r\n]", pos) or #s + 1 end
  local value
  local function str()
    local out = {}
    pos = pos + 1
    while true do
      local c = s:sub(pos, pos)
      if c == '"' then pos = pos + 1; return table.concat(out) end
      if c == "\\" then
        local e = s:sub(pos + 1, pos + 1)
        out[#out + 1] = ({n = "\n", t = "\t"})[e] or e
        pos = pos + 2
      else
        out[#out + 1] = c
        pos = pos + 1
      end
    end
  end
  function value()
    ws()
    local c = s:sub(pos, pos)
    if c == "{" then
      local obj = {}
      pos = pos + 1; ws()
      if s:sub(pos, pos) == "}" then pos = pos + 1; return obj end
      repeat
        ws(); local key = str(); ws(); pos = pos + 1
        obj[key] = value(); ws()
        local sep = s:sub(pos, pos); pos = pos + 1
      until sep == "}"
      return obj
    elseif c == "[" then
      local arr = {}
      pos = pos + 1; ws()
      if s:sub(pos, pos) == "]" then pos = pos + 1; return arr end
      repeat
        arr[#arr + 1] = value(); ws()
        local sep = s:sub(pos, pos); pos = pos + 1
      until sep == "]"
      return arr
    elseif c == '"' then return str()
    elseif s:find("^true", pos) then pos = pos + 4; return true
    elseif s:find("^false", pos) then pos = pos + 5; return false
    elseif s:find("^null", pos) then pos = pos + 4; return nil
    else
      local num = s:match("^-?%d+%.?%d*[eE]?[-+]?%d*", pos)
      pos = pos + #num
      return math.tointeger(tonumber(num)) or tonumber(num)
    end
  end
  return value()
end
local doc = {name = "zrun", tags = {"fast", "lua"}, version = 1, ratio = 0.5, nested = {ok = true, list = {1, 2, {3}}}, text = 'say "hi"\n'}
local enc = json.encode(doc)
print(enc)
local back = json.decode(enc)
print(json.encode(back) == enc, back.nested.list[3][1], back.text)

-- Binary trees (the benchmarks game)
local function bottom_up(depth)
  if depth == 0 then return {} end
  depth = depth - 1
  return {bottom_up(depth), bottom_up(depth)}
end
local function check(tree)
  if tree[1] then return 1 + check(tree[1]) + check(tree[2]) end
  return 1
end
local max_depth = 8
for d = 4, max_depth, 2 do
  local iters = 2 ^ (max_depth - d + 4)
  local c = 0
  for _ = 1, iters do c = c + check(bottom_up(d)) end
  print(string.format("%d trees of depth %d check: %d", iters, d, c))
end

-- Sieve of Eratosthenes
local function sieve(n)
  local is = {}
  for i = 2, n do is[i] = true end
  for i = 2, math.floor(math.sqrt(n)) do
    if is[i] then for j = i * i, n, i do is[j] = false end end
  end
  local primes = {}
  for i = 2, n do if is[i] then primes[#primes + 1] = i end end
  return primes
end
local p = sieve(200)
print(#p, p[#p], table.concat(p, " ", 1, 10))

-- Word frequencies
local text = [[the quick brown fox jumps over the lazy dog the dog barks
and the fox runs quick quick]]
local freq = {}
for w in text:gmatch("%a+") do freq[w] = (freq[w] or 0) + 1 end
local words = {}
for w, n in pairs(freq) do words[#words + 1] = {w = w, n = n} end
table.sort(words, function(a, b) if a.n ~= b.n then return a.n > b.n end return a.w < b.w end)
for i = 1, 4 do io.write(words[i].w, "=", words[i].n, " ") end
print()

-- A tokenizer with a closure-based scanner
local function tokens(src)
  local i = 1
  return function()
    i = src:find("%S", i)
    if not i then return nil end
    local s, e = src:find("^%d+%.?%d*", i)
    if s then i = e + 1; return "num", src:sub(s, e) end
    s, e = src:find("^[%a_][%w_]*", i)
    if s then i = e + 1; return "id", src:sub(s, e) end
    local c = src:sub(i, i); i = i + 1
    return "op", c
  end
end
local out = {}
for k, v in tokens("x1 = 3.5 * (y + 42)") do out[#out + 1] = k .. ":" .. v end
print(table.concat(out, " "))
