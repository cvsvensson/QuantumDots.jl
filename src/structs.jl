abstract type AbstractBasis end
const BasisOrMissing = Union{AbstractBasis,Missing}
basis(::AbstractArray) = missing

abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry end

struct Fermion{S}
    id::S
end
id(f::Fermion) = f.id

struct FermionBasis{M,S,T,Sym} <: AbstractBasis
    dict::Dictionary{S,T}
    symmetry::Sym
    function FermionBasis(fermions::NTuple{M,Fermion{S}}, sym::Sym=NoSymmetry()) where {M,S,Sym<:AbstractSymmetry}
        reps = ntuple(n->fermion_sparse_matrix(n,2^M,sym),M)
        # FermionBasis(fermionids,reps,sym)
        new{M,S,eltype(reps),Sym}(Dictionary(map(id,fermions), reps), sym)
    end
end
Base.getindex(b::FermionBasis,i) = b.dict[i]
Base.getindex(b::FermionBasis,args...) = b.dict[args]
Base.keys(b::FermionBasis) = keys(b.dict)
labels(b::FermionBasis) = keys(b).values
Base.show(io::IO, ::MIME"text/plain", b::FermionBasis) = show(io,b)
Base.show(io::IO, b::FermionBasis{M,S,T,Sym}) where {M,S,T,Sym} = print(io, "FermionBasis{$M,$S,$T,$Sym}:\nkeys = ", keys(b))


symmetry(::NTuple{M},::NoSymmetry) where M = NoSymmetry()
FermionBasis(iters...; qn = NoSymmetry()) = FermionBasis(Fermion.(Tuple(Base.product(iters...))), symmetry(Tuple(Base.product(iters...)), qn))
FermionBasis(iter; qn = NoSymmetry()) = FermionBasis(Fermion.(Tuple(iter)), symmetry(Tuple(iter),qn))
nbr_of_fermions(::FermionBasis{M}) where M = M


struct AbelianFockSymmetry{IF,FI,QN,QNfunc} <: AbstractSymmetry
    indtofockdict::IF
    focktoinddict::FI
    qntoblocksizes::Dictionary{QN,Int}
    qntofockstates::Dictionary{QN,Vector{Int}}
    qntoinds::Dictionary{QN,Vector{Int}}
    conserved_quantity::QNfunc
end

##BdG
struct BdGFermion{S,B,T}
    id::S
    basis::B
    amp::T
    hole::Bool
    function BdGFermion(id::S,basis::B,amp::T=true,hole=false) where {S,B,T}
        # pos = findfirst(==(id), labels(basis))
        new{S,B,T}(id,basis,amp,hole)
    end
end
Base.adjoint(f::BdGFermion) = BdGFermion(f.id,f.basis,f.amp,!f.hole)
basis(f::BdGFermion) = f.basis
function rep(f::BdGFermion)
    b = basis(f)
    N = nbr_of_fermions(b) 
    p = pos(f,b)
    sparsevec([p + f.hole*N], f.amp, 2N)
end
function Base.:*(f1::BdGFermion, f2::BdGFermion; symmetrize::Bool=true)
    if symmetrize
        return ((rep(f1')*transpose(rep(f2)) - rep(f2')*transpose(rep(f1)))) * !same_fermion(f1,f2)
    else
        return (rep(f1')*transpose(rep(f2)) ) * !same_fermion(f1,f2)
    end
end
same_fermion(f1::BdGFermion, f2::BdGFermion) = f1.id == f2.id && f1.hole == f2.hole

Base.:*(x::Number,f::BdGFermion) = BdGFermion(f.id,f.basis,x*f.amp,f.hole)
Base.:*(f::BdGFermion,x::Number) = BdGFermion(f.id,f.basis,f.amp*x,f.hole)
struct FermionBdGBasis{M,L} <: AbstractBasis
    position::Dictionary{L,Int}
    function FermionBdGBasis(labels::NTuple{M,L}) where {M,L}
        positions = map((l,n) -> l => n, labels, eachindex(labels))
        new{M,L}(dictionary(positions))
    end
end
nbr_of_fermions(::FermionBdGBasis{M}) where M = M
Base.getindex(b::FermionBdGBasis,i) = BdGFermion(i,b)
Base.getindex(b::FermionBdGBasis,args...) = BdGFermion(args,b) 
pos(f::BdGFermion, b::FermionBdGBasis) = b.position[f.id]