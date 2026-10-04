-- classes, inheritance, metamethods
local Animal = {}
Animal.__index = Animal
function Animal.new(name, sound)
  local self = setmetatable({}, Animal)
  self.name = name
  self.sound = sound
  return self
end
function Animal:speak() return self.name .. " says " .. self.sound end
function Animal:__tostring() return "Animal(" .. self.name .. ")" end
Animal.__tostring = Animal.__tostring

local Dog = setmetatable({}, {__index = Animal})
Dog.__index = Dog
Dog.__tostring = function(d) return "Dog(" .. d.name .. ")" end
function Dog.new(name)
  local d = Animal.new(name, "woof")
  return setmetatable(d, Dog)
end
function Dog:fetch() return self.name .. " fetches" end

local a = Animal.new("cat", "meow")
local d = Dog.new("rex")
print(a:speak(), d:speak(), d:fetch())
print(tostring(a), tostring(d))
print(getmetatable(d) == Dog, getmetatable(a) == Animal)

local Vec = {}
Vec.__index = Vec
local function vec(x, y) return setmetatable({x = x, y = y}, Vec) end
Vec.__add = function(a, b) return vec(a.x + b.x, a.y + b.y) end
Vec.__sub = function(a, b) return vec(a.x - b.x, a.y - b.y) end
Vec.__mul = function(a, k) if type(a) == "number" then return vec(a * k.x, a * k.y) end return vec(a.x * k, a.y * k) end
Vec.__eq = function(a, b) return a.x == b.x and a.y == b.y end
Vec.__lt = function(a, b) return a.x * a.x + a.y * a.y < b.x * b.x + b.y * b.y end
Vec.__le = function(a, b) return not (b < a) end
Vec.__len = function(a) return 2 end
Vec.__unm = function(a) return vec(-a.x, -a.y) end
Vec.__concat = function(a, b) return tostring(a) .. "|" .. tostring(b) end
Vec.__tostring = function(a) return "(" .. a.x .. "," .. a.y .. ")" end
Vec.__call = function(self, k) return self.x * k end
local p, q = vec(1, 2), vec(3, 4)
print(tostring(p + q), tostring(q - p), tostring(p * 3), tostring(2 * q), tostring(-p))
print(p == vec(1, 2), p ~= q, p < q, q <= p, #p, p .. q, p(10))

-- __index / __newindex functions, rawget/rawset
local log = {}
local proxy = setmetatable({}, {
  __index = function(t, k) log[#log + 1] = "get " .. k; return k .. "!" end,
  __newindex = function(t, k, v) log[#log + 1] = "set " .. k; rawset(t, k, v) end,
})
print(proxy.a, proxy.b)
proxy.c = 1
proxy.c = 2
print(proxy.c, rawget(proxy, "a"), table.concat(log, ","))

-- default values
local defaults = setmetatable({}, {__index = function() return 0 end})
defaults.x = defaults.x + 5
print(defaults.x, defaults.y)
