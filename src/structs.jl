abstract type AbstractBasis end
const BasisOrMissing = Union{AbstractBasis,Missing}
basis(::AbstractArray) = missing

abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry end

struct FermionBasis{M,S,T,Sym} <: AbstractBasis
    dict::Dictionary{S,T}
    symmetry::Sym
    function FermionBasis(fermionids::NTuple{M,S}, reps::NTuple{M,T}, sym::Sym=NoSymmetry()) where {M,S,T,Sym}
        new{M,S,T,Sym}(Dictionary(fermionids, reps), sym)
    end
end
Base.getindex(b::FermionBasis,i) = b.dict[i]
Base.getindex(b::FermionBasis,args...) = b.dict[args]
FermionBasis(fermionids::NTuple{M}) where M = FermionBasis(fermionids, ntuple(n->fermion_sparse_matrix(n,M),M), NoSymmetry())
FermionBasis(iters...) = FermionBasis(Tuple(Base.product(iters...)))
FermionBasis(iter) = FermionBasis(Tuple(iter))

BlockFermionBasis(qn,iters...) = blockbasis(Tuple(Base.product(iters...)),qn)
BlockFermionBasis(qn,iter) = blockbasis(Tuple(iter),qn)
# FermionBasis(iters...; kwargs...) = FermionBasis(ntuple(identity,n),iters...)
# FermionBasis(n::Integer; kwargs...) = FermionBasis(ntuple(i->i,n))


nbr_of_fermions(::FermionBasis{M}) where M = M

function fermion_sparse_matrix(fermion_number, total_nbr_of_fermions)
    mat = spzeros(Int,2^total_nbr_of_fermions,2^total_nbr_of_fermions)
    _fill!(mat, fs -> removefermion(fermion_number,fs), NoSymmetry())
    return mat
end

function _fill!(mat,op,::NoSymmetry)
    for ind in axes(mat,2)
        newfockstate, amp = op(ind-1)
        newind = newfockstate + 1
        mat[newind,ind] += amp
    end
    return mat
end

function _fill!(mat,op,sym::AbelianFockSymmetry)
    for ind in axes(mat,2)
        newfockstate, amp = op(sym.indtofock(ind))
        newind = sym.focktoind(newfockstate)
        mat[newind,ind] += amp
    end
    return mat
end

function removefermion(digitposition,statefocknbr) #Currently only works for a single creation operator
    cdag = focknbr(digitposition)
    newfocknbr = cdag ⊻ statefocknbr
    allowed = !iszero(cdag & statefocknbr) #&& allunique(digitpositions) 
    fermionstatistics = jwstring(digitposition, statefocknbr) 
    return allowed * newfocknbr, allowed * fermionstatistics
end

function parityoperator(basis::FermionBasis{<:Any,<:Any,<:Any,NoSymmetry})
    mat = spzeros(Int,2^nbr_of_fermions(basis),2^nbr_of_fermions(basis))
    _fill!(mat, fs->(fs,parity(fs)), NoSymmetry())
    return mat
end