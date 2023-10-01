#This file contains an introduction to the Julia langauge
## Repl
1 + 1
print("Hello World")

## Variables, unicode
a = 1
Δ = 1.0
Δ ≈ a 

## Defining functions
function f(x)
    return x^2
end
f2(x) = x^2
f3 = x -> x^2

f(2)
2 |> f

## Linear algebra
v = [1, 2, 3]
m = [1 2; 3 4]
m = reshape(1:9, 3, 3)
m * v
m * m == m^2
v'
m'
v' * v == dot(v, v)

v[1]

## Broadcasting
v = 1:10
f.(v)
map(f, v)
?map
map(f, m)
map(f, rand(2, 2, 2, 2))

## Julia is slow, then fast
@time m^.5
@time m^.5


## Package manager
]status
]activate --temp
]add ForwardDiff
ForwardDiff.gradient(v-> sum(v)*prod(v), 1:10)

## Threads
@time for i in 1:10
        sleep(.1)
end
using Base.Threads
nthreads()
@time @threads for i in 1:10
        sleep(.1)
end



