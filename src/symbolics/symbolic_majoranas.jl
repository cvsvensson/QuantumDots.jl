struct SymbolicMajoranaBasis
    name::Symbol
    universe::UInt64
end
macro majoranas(xs...)
    universe = hash(xs)
    defs = map(xs) do x
        :($(esc(x)) = SymbolicMajoranaBasis($(Expr(:quote, x)), $universe))
    end
    Expr(:block, defs...,
        :(tuple($(map(x -> esc(x), xs)...))))
end
Base.getindex(f::SymbolicMajoranaBasis, is...) = MajoranaSym(is, f.name, f.universe)
Base.getindex(f::SymbolicMajoranaBasis, i) = MajoranaSym(i, f.name, f.universe)

struct MajoranaSym{L} <: AbstractFermionSym
    label::L
    name::Symbol
    universe::UInt
end
Base.:(==)(a::MajoranaSym, b::MajoranaSym) = a.label == b.label && a.name == b.name && a.universe == b.universe
Base.hash(a::MajoranaSym, h::UInt) = hash(hash(a.label, hash(a.name, h)))
Base.adjoint(x::MajoranaSym) = MajoranaSym(x.label, x.name, x.universe)
Base.iszero(x::MajoranaSym) = false
function Base.show(io::IO, x::MajoranaSym)
    print(io, x.name)
    if Base.isiterable(typeof(x.label))
        Base.show_delim_array(io, x.label, "[", ",", "]", false)
    else
        print(io, "[", x.label, "]")
    end
end
function Base.isless(a::MajoranaSym, b::MajoranaSym)
    if a.universe !== b.universe
        a.universe < b.universe
    elseif a.name == b.name
        a.label < b.label
    else
        a.name < b.name
    end
end
function Base.:^(a::MajoranaSym, b)
    if b isa Number && iszero(b)
        1
    elseif b isa Number && b == 1
        a
    elseif b isa Integer && b >= 2
        1
    else
        throw(ArgumentError("Invalid exponent $b"))
    end
end
function ordered_prod(a::MajoranaSym, b::MajoranaSym)
    if a == b
        1
    elseif a < b
        FermionMul(1, [a, b])
    elseif a > b
        FermionMul((-1)^(a.universe == b.universe), [b, a]) + Int(a.name == b.name && a.label == b.label && a.universe == b.universe)
    else
        throw(ArgumentError("Don't know how to multiply $a * $b"))
    end
end
eval_in_basis(a::MajoranaSym, f::AbstractBasis) = f[a.label]


@testitem "MajoranaSym" begin
    @majoranas γ f
    #test canonical anticommutation relations
    @test γ[1] * γ[1] == 1
    @test γ[1] * γ[2] == -γ[2] * γ[1]
    @test γ[1] * f[1] + f[1] * γ[1] == 0

    @test γ[1] * γ[2] * γ[1] == -γ[2]
    @test γ[1] * γ[2] * γ[3] == -γ[3] * γ[2] * γ[1]

    f1 = (γ[1] + 1im * γ[2]) / 2
    f2 = (γ[3] + 1im * γ[4]) / 2
    @test iszero(f1 - f1)
    @test iszero(f1 * f1)
    @test iszero(2 * f1 - 2 * f1)
    @test iszero(0 * f1)
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