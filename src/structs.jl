abstract type AbstractBasis end
const BasisOrMissing = Union{AbstractBasis,Missing}
basis(::AbstractArray) = missing

abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry end

struct FermionBasis{M,S,T,Sym} <: AbstractBasis
    dict::Dictionary{S,T}
    symmetry::Sym
    function FermionBasis(fermionids::NTuple{M,S}, sym::Sym=NoSymmetry()) where {M,S,Sym<:AbstractSymmetry}
        reps = ntuple(n->fermion_sparse_matrix(n,2^M,sym),M)
        # FermionBasis(fermionids,reps,sym)
        new{M,S,eltype(reps),Sym}(Dictionary(fermionids, reps), sym)
    end
end
Base.getindex(b::FermionBasis,i) = b.dict[i]
Base.getindex(b::FermionBasis,args...) = b.dict[args]

# function FermionBasis(fermionids::NTuple{M}; sym) where M = FermionBasis(fermionids, ntuple(n->fermion_sparse_matrix(n,M),M), NoSymmetry())
# FermionBasis(iters...) = FermionBasis(Tuple(Base.product(iters)))
# FermionBasis(iter) = FermionBasis(Tuple(iter))
FermionBasisQN(iters...; qn) = FermionBasis(Tuple(Base.product(iters...)), symmetry(Tuple(Base.product(iters...)), qn))
FermionBasisQN(iter; qn) = FermionBasis(Tuple(iter), symmetry(Tuple(iter),qn))
# FermionBasis(fermionids::NTuple{M}) where M = FermionBasis(fermionids, ntuple(n->fermion_sparse_matrix(n,M),M), NoSymmetry())
FermionBasis(iters...) = FermionBasis(Tuple(Base.product(iters...)), NoSymmetry())
FermionBasis(iter) = FermionBasis(Tuple(iter), NoSymmetry())

# FermionBasis(fermionids::NTuple{M}) where M = blockbasis(fermionids, qn)

# BlockFermionBasis(qn,iters...) = blockbasis(Tuple(Base.product(iters...)),qn)
# BlockFermionBasis(qn,iter) = blockbasis(Tuple(iter),qn)
# FermionBasis(iters...; kwargs...) = FermionBasis(ntuple(identity,n),iters...)
# FermionBasis(n::Integer; kwargs...) = FermionBasis(ntuple(i->i,n))


nbr_of_fermions(::FermionBasis{M}) where M = M



struct AbelianFockSymmetry{IF,FI,QN} <: AbstractSymmetry
    indtofock::IF
    focktoind::FI
    blocksizes::Vector{Int}
    conserved_quantity::QN
end