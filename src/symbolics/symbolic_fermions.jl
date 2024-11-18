
struct SymbolicFermionBasis
    name::Symbol
    universe::UInt64
end

"""
    @fermions a b ...

Create one or more fermion species with the given names. Indexing into fermions species
gives a concrete fermion. Fermions in one `@fermions` block anticommute with each other, 
and commute with fermions in other `@fermions` blocks.

# Examples:
- `@fermions a b` creates two species of fermions that anticommute:
    - `a[1]' * a[1] + a[1] * a[1]' == 1`
    - `a[1]' * b[1] + b[1] * a[1]' == 0`
- `@fermions a; @fermions b` creates two species of fermions that commute with each other:
    - `a[1]' * a[1] + a[1] * a[1]' == 1`
    - `a[1] * b[1] - b[1] * a[1] == 0`

See also [`@majoranas`](@ref), [`QuantumDots.eval_in_basis`](@ref).
"""
macro fermions(xs...)
    universe = hash(xs)
    defs = map(xs) do x
        :($(esc(x)) = SymbolicFermionBasis($(Expr(:quote, x)), $universe))
    end
    Expr(:block, defs...,
        :(tuple($(map(x -> esc(x), xs)...))))
end

Base.getindex(f::SymbolicFermionBasis, is...) = FermionSym(false, is, f.name, f.universe)
Base.getindex(f::SymbolicFermionBasis, i) = FermionSym(false, i, f.name, f.universe)

struct FermionSym{L} <: AbstractFermionSym
    creation::Bool
    label::L
    name::Symbol
    universe::UInt
end
Base.adjoint(x::FermionSym) = FermionSym(!x.creation, x.label, x.name, x.universe)
Base.iszero(x::FermionSym) = false
function Base.show(io::IO, x::FermionSym)
    print(io, x.name, x.creation ? "†" : "")
    if Base.isiterable(typeof(x.label))
        Base.show_delim_array(io, x.label, "[", ",", "]", false)
    else
        print(io, "[", x.label, "]")
    end
end
function Base.isless(a::FermionSym, b::FermionSym)
    if a.universe !== b.universe
        a.universe < b.universe
    elseif a.creation == b.creation
        a.name == b.name && return a.label < b.label
        a.name < b.name
    else
        a.creation > b.creation
    end
end
Base.:(==)(a::FermionSym, b::FermionSym) = a.creation == b.creation && a.label == b.label && a.name == b.name && a.universe == b.universe
Base.hash(a::FermionSym, h::UInt) = hash(hash(a.creation, hash(a.label, hash(a.name, h))))

function ordered_prod(a::FermionSym, b::FermionSym)
    if a == b
        0
    elseif a < b
        FermionMul(1, [a, b])
    elseif a > b
        FermionMul((-1)^(a.universe == b.universe), [b, a]) + Int(a.name == b.name && a.label == b.label && a.universe == b.universe)
    else
        throw(ArgumentError("Don't know how to multiply $a * $b"))
    end
end
function Base.:^(a::FermionSym, b)
    if b isa Number && iszero(b)
        1
    elseif b isa Number && b == 1
        a
    elseif b isa Integer && b >= 2
        0
    else
        throw(ArgumentError("Invalid exponent $b"))
    end
end

""" 
    eval_in_basis(a, f::AbstractBasis)

Evaluate an expression with fermions in a basis `f`. 

# Examples
```julia
@fermions a
f = FermionBasis(1:2)
QuantumDots.eval_in_basis(a[1]'*a[2] + hc, f)
```
"""
eval_in_basis(a::FermionSym, f::AbstractBasis) = a.creation ? f[a.label]' : f[a.label]


@testitem "SymbolicFermions" begin
    using Symbolics
    @fermions f c
    @fermions b
    @variables a
    f1 = f[:a]
    f2 = f[:b]
    f3 = f[1, :↑]

    # Test canonical commutation relations
    @test f1' * f1 + f1 * f1' == 1
    @test iszero(f1 * f2 + f2 * f1)
    @test iszero(f1' * f2 + f2 * f1')

    # c anticommutes with f
    @test iszero(f1' * c[1] + c[1] * f1')
    # b commutes with f
    @test iszero(f1' * b[1] - b[1] * f1')

    @test_nowarn display(f1)
    @test_nowarn display(f3)
    @test_nowarn display(1 * f1)
    @test_nowarn display(2 * f3)
    @test_nowarn display(1 + f1)
    @test_nowarn display(1 + f3)
    @test_nowarn display(1 + a * f2 - 5 * f1 + 2 * a * f1 * f2)

    @test iszero(f1 - f1)
    @test iszero(f1 * f1)
    @test f1 * f2 isa QuantumDots.FermionMul
    @test iszero(2 * f1 - 2 * f1)
    @test iszero(0 * f1)
    @test 2 * f1 isa QuantumDots.FermionMul
    @test iszero(f1 * 0)
    @test iszero(f1^2)
    @test iszero(0 * (f1 + f2))
    @test iszero((f1 + f2) * 0)
    @test iszero(f1 * f2 * f1)
    f12 = f1 * f2
    @test iszero(f12'' - f12)
    @test iszero(f12 * f12)
    @test iszero(f12' * f12')
    nf1 = f1' * f1
    @test nf1^2 == nf1
    @test f1' * f1 isa QuantumDots.FermionMul
    @test f1 * f1' isa QuantumDots.FermionAdd

    @test 1 + (f1 + f2) == 1 + f1 + f2 == f1 + f2 + 1 == f1 + 1 + f2 == 1 * f1 + f2 + 1 == f1 + 0.5 * f2 + 1 + (0 * f1 + 0.5 * f2) == (0.5 + 0.5 * f1 + 0.2 * f2) + 0.5 + (0.5 * f1 + 0.8 * f2) == (1 + f1' + (1 * f2)')'
    @test iszero((2 * f1) * (2 * f1))
    @test iszero((2 * f1)^2)
    @test (2 * f2) * (2 * f1) == -4 * f1 * f2
    @test f1 == (f1 * (f1 + 1)) == (f1 + 1) * f1
    @test iszero(f1 * (f1 + f2) * f1)
    @test (f1 * (f1 + f2)) == f1 * f2
    @test (2nf1 - 1) * (2nf1 - 1) == 1

    @test (1 * f1) * f2 == f1 * f2
    @test (1 * f1) * (1 * f2) == f1 * f2
    @test f1 * f2 == f1 * (1 * f2) == f1 * f2
    @test f1 - 1 == (1 * f1) - 1 == (0.5 + f1) - 1.5
end
