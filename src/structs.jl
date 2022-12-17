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

symmetry(::NTuple{M},::NoSymmetry) where M = NoSymmetry()
FermionBasis(iters...; qn = NoSymmetry()) = FermionBasis(Fermion.(Tuple(Base.product(iters...))), symmetry(Tuple(Base.product(iters...)), qn))
FermionBasis(iter; qn = NoSymmetry()) = FermionBasis(Fermion.(Tuple(iter)), symmetry(Tuple(iter),qn))
nbr_of_fermions(::FermionBasis{M}) where M = M


struct AbelianFockSymmetry{IF,FI,QN,QNfunc} <: AbstractSymmetry
    indtofock::IF
    focktoind::FI
    qntoblocksizes::Dictionary{QN,Int}
    qntofockstates::Dictionary{QN,Vector{Int}}
    qntoinds::Dictionary{QN,Vector{Int}}
    conserved_quantity::QNfunc
end