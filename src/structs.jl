abstract type AbstractBasis end
abstract type AbstractManyBodyBasis <: AbstractBasis end
abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry end

siteindex(label, b::AbstractManyBodyBasis) = siteindex(label, b.jw)
siteindices(labels, b::AbstractManyBodyBasis) = siteindices(labels, b.jw)

handle_labels(iter, iters...) = Base.product(iter, iters...)
handle_labels(iter) = iter

##BdG
abstract type AbstractBdGFermion end
"""
    struct BdGFermion{S,B,T} <: AbstractBdGFermion

The `BdGFermion` struct represents a basis fermion for BdG matrices.

# Fields
- `id::S`: The identifier of the fermion.
- `basis::B`: The fermion basis.
- `amp::T`: The amplitude of the fermion (default: `true`, i.e. 1).
- `hole::Bool`: Indicates whether the fermion is a hole (default: `true`).
"""
struct BdGFermion{S,B,T} <: AbstractBdGFermion
    id::S
    basis::B
    amp::T
    hole::Bool
    function BdGFermion(id::S, basis::B, amp::T=true, hole=true) where {S,B,T}
        new{S,B,T}(id, basis, amp, hole)
    end
end
Base.adjoint(f::BdGFermion) = BdGFermion(f.id, f.basis, f.amp, !f.hole)
basis(f::BdGFermion) = f.basis
"""
    rep(f::BdGFermion)

Constructs a sparse vector representation of a `BdGFermion` object.
"""
function rep(f::BdGFermion)
    b = basis(f)
    N = nbr_of_modes(b)
    sparsevec([indexpos(f, b)], f.amp, 2N)
end
"""
    *(f1::BdGFermion, f2::BdGFermion; symmetrize=true)

Multiply two `BdGFermion` objects `f1` and `f2`. By default, it symmetrizes the result, returning a BdG matrix in the convention used here.
"""
function Base.:*(f1::BdGFermion, f2::BdGFermion; symmetrize::Bool=true)
    if symmetrize
        return ((rep(f1') * transpose(rep(f2)) - rep(f2') * transpose(rep(f1)))) * !same_fermion(f1, f2)
    else
        return (rep(f1') * transpose(rep(f2))) * !same_fermion(f1, f2)
    end
end
same_fermion(f1::BdGFermion, f2::BdGFermion) = f1.id == f2.id && f1.hole == f2.hole

Base.:*(x::Number, f::BdGFermion) = BdGFermion(f.id, f.basis, x * f.amp, f.hole)
Base.:*(f::BdGFermion, x::Number) = BdGFermion(f.id, f.basis, f.amp * x, f.hole)
struct FermionBdGBasis{M,L} <: AbstractBasis
    position::Dictionary{L,Int}
    function FermionBdGBasis(labels)
        positions = map((l, n) -> l => n, labels, eachindex(labels))
        new{length(labels),eltype(labels)}(dictionary(positions))
    end
end
FermionBdGBasis(labels...) = FermionBdGBasis(collect(Base.product(labels...)))
nbr_of_modes(::FermionBdGBasis{M}) where {M} = M
Base.getindex(b::FermionBdGBasis, i) = BdGFermion(i, b)
Base.getindex(b::FermionBdGBasis, args...) = BdGFermion(args, b)
indexpos(f::BdGFermion, b::FermionBdGBasis) = b.position[f.id] + !f.hole * nbr_of_modes(b)

Base.iterate(b::FermionBdGBasis) = ((result, state) = iterate(keys(b.position)); (b[result], state));
Base.iterate(b::FermionBdGBasis, state) = (res = iterate(keys(b.position), state); isnothing(res) ? nothing : (b[res[1]], res[2]))
Base.length(::FermionBdGBasis{M}) where {M} = M


abstract type AbstractDiagonalHamiltonian end
