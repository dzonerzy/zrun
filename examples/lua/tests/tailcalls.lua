-- tail calls: `return f(x)` gives up the caller's frame, so chains of them
-- run in constant stack, as Lua guarantees
local function count(n, acc)
  if n == 0 then return acc end
  return count(n - 1, acc + 1)
end
print(count(200000, 0))

-- mutual recursion
local is_even, is_odd
function is_even(n) if n == 0 then return true end return is_odd(n - 1) end
function is_odd(n) if n == 0 then return false end return is_even(n - 1) end
print(is_even(100001), is_odd(100001))

-- a method in tail position
local Counter = {}
Counter.__index = Counter
function Counter.new() return setmetatable({n = 0}, Counter) end
function Counter:run(k)
  if k == 0 then return self.n end
  self.n = self.n + 1
  return self:run(k - 1)
end
print(Counter.new():run(50000))

-- every value of the call returned, none for none
local function three() return 1, 2, 3 end
local function pass() return three() end
print(pass())
local function nothing() end
local function pass_nothing() return nothing() end
print(select('#', pass_nothing()))

-- a library function in tail position, and parentheses (one value: not a
-- tail call)
local function str(x) return tostring(x) end
local function first(...) return (three()) end
print(str(42), first())

-- a loop written as tail calls, with an accumulator table
local function collect(i, n, t)
  if i > n then return t end
  t[#t + 1] = i * i
  return collect(i + 1, n, t)
end
local squares = collect(1, 100000, {})
print(#squares, squares[1], squares[100000])

-- an error deep in a tail chain
local function fail_at(n)
  if n == 0 then error("bottom") end
  return fail_at(n - 1)
end
print(pcall(fail_at, 100000))
