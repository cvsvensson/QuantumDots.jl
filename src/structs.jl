abstract type AbstractBasis end
const BasisOrMissing = Union{AbstractBasis,Missing}
basis(::AbstractArray) = missing

abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry end

struct FermionBasis{M,S,T,Sym} <: AbstractBasis
    dict::Dictionary{S,T}
    symmetry::Sym
    function FermionBasis(fermions, sym::Sym; sparse = true) where {Sym<:AbstractSymmetry}
        M = length(fermions)
        S = eltype(fermions)
        reps = ntuple(n->fermion_matrix(n,2^M,sym; sparse),M) ##TODO: Let user choose dense matrices
        new{M,S,eltype(reps),Sym}(Dictionary(fermions, reps), sym)
    end
end
Base.getindex(b::FermionBasis,i) = b.dict[i]
Base.getindex(b::FermionBasis,args...) = b.dict[args]
Base.keys(b::FermionBasis) = keys(b.dict)
labels(b::FermionBasis) = keys(b).values
Base.show(io::IO, ::MIME"text/plain", b::FermionBasis) = show(io,b)
Base.show(io::IO, b::FermionBasis{M,S,T,Sym}) where {M,S,T,Sym} = print(io, "FermionBasis{$M,$S,$T,$Sym}:\nkeys = ", keys(b))


symmetry(labels,::NoSymmetry) = NoSymmetry()
FermionBasis(iters...; qn = NoSymmetry(), sparse = true) = FermionBasis(Base.product(iters...), symmetry(Base.product(iters...), qn); sparse)
FermionBasis(iter; qn = NoSymmetry(), sparse = true) = FermionBasis(iter, symmetry(iter,qn); sparse)
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
abstract type AbstractBdGFermion end
struct BdGFermion{S,B,T} <: AbstractBdGFermion
    id::S
    basis::B
    amp::T
    hole::Bool
    function BdGFermion(id::S,basis::B,amp::T=true,hole=true) where {S,B,T}
        new{S,B,T}(id,basis,amp,hole)
    end
end
Base.adjoint(f::BdGFermion) = BdGFermion(f.id,f.basis,f.amp,!f.hole)
basis(f::BdGFermion) = f.basis
function rep(f::BdGFermion)
    b = basis(f)
    N = nbr_of_fermions(b) 
    sparsevec([indexpos(f,b)], f.amp, 2N)
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
    function FermionBdGBasis(labels)
        positions = map((l,n) -> l => n, labels, eachindex(labels))
        new{length(labels),eltype(labels)}(dictionary(positions))
    end
end
nbr_of_fermions(::FermionBdGBasis{M}) where M = M
Base.getindex(b::FermionBdGBasis,i) = BdGFermion(i,b)
Base.getindex(b::FermionBdGBasis,args...) = BdGFermion(args,b) 
indexpos(f::BdGFermion, b::FermionBdGBasis) = b.position[f.id] + !f.hole*nbr_of_fermions(b)