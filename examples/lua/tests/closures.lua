local fs = {}
for i = 1, 3 do fs[i] = function() return i end end
print(fs[1](), fs[2](), fs[3]())
local gs = {}
for _, v in ipairs({"a", "b"}) do local w = v .. "!"; gs[#gs + 1] = function() return w end end
print(gs[1](), gs[2]())
local hs = {}
local k = 1
while k <= 3 do local x = k * 10; hs[k] = function() return x end; k = k + 1 end
print(hs[1](), hs[2](), hs[3]())
local function counter() local n = 0; return function() n = n + 1; return n end end
local c1, c2 = counter(), counter()
print(c1(), c1(), c2())
