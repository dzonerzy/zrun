-- A return at the top level ends the chunk, from inside a loop too
local x = 0
for i = 1, 5 do
  x = x + i
  if i == 3 then
    print("returning", x)
    return x
  end
end
print("not reached")
