-- closures, iterators, varargs, recursion, multiple returns
local function range(n, step)
  step = step or 1
  local i = 0
  return function()
    i = i + step
    if i <= n then return i end
  end
end
local out = {}
for v in range(10, 3) do out[#out + 1] = v end
print(table.concat(out, " "))

local function sum(...)
  local s = 0
  for _, v in ipairs({...}) do s = s + v end
  return s, select("#", ...)
end
print(sum(1, 2, 3, 4), sum())
local function pack2(...) return {n = select("#", ...), ...} end
local t = pack2(1, nil, 3, nil)
print(t.n, t[1], t[2], t[3])
print(table.unpack({1, 2, 3}))
print(table.unpack({1, 2, 3}, 2))

local memo = {}
local function fibm(n)
  if n <= 2 then return 1 end
  if memo[n] then return memo[n] end
  local v = fibm(n - 1) + fibm(n - 2)
  memo[n] = v
  return v
end
print(fibm(80))

local function queens(n)
  local count, cols = 0, {}
  local function ok(r, c)
    for pr = 1, r - 1 do
      local pc = cols[pr]
      if pc == c or math.abs(pc - c) == r - pr then return false end
    end
    return true
  end
  local function place(r)
    if r > n then count = count + 1; return end
    for c = 1, n do
      if ok(r, c) then cols[r] = c; place(r + 1) end
    end
  end
  place(1)
  return count
end
print(queens(6), queens(8))

local function compose(f, g) return function(...) return f(g(...)) end end
local inc = function(x) return x + 1 end
local dbl = function(x) return x * 2 end
print(compose(inc, dbl)(5), compose(dbl, inc)(5))

local function swap(a, b) return b, a end
local x, y = swap(1, 2)
print(x, y)
local a, b, c = (function() return 1, 2 end)()
print(a, b, c)
print((swap(1, 2)))

-- generic for with pairs over a sorted copy
local scores = {alice = 3, bob = 5, carol = 4}
local names = {}
for name in pairs(scores) do names[#names + 1] = name end
table.sort(names)
for _, name in ipairs(names) do io.write(name, "=", scores[name], " ") end
print()
table.sort(names, function(p, q) return scores[p] > scores[q] end)
print(table.concat(names, ","))

-- upvalues shared between closures
local function counter()
  local n = 0
  return function() n = n + 1; return n end, function() return n end
end
local incr, get = counter()
incr(); incr()
print(get())
-- recursion through a local function
local function fact(n) if n <= 1 then return 1 end return n * fact(n - 1) end
print(fact(20), fact(21))
