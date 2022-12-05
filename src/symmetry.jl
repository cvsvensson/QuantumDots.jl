abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry end
# symmetry(::FermionBasis) = NoSymmetry()
symmetry(::FermionBasis) = NoSymmetry()

struct FermionParityBasis{M,S,IF,FI} <: AbstractBasis
    fb::FermionBasis{M,S}
    indtofock::IF
    focktoind::FI
    blocksizes::Vector{Int}
    function FermionParityBasis(fb::FermionBasis{M,S}) where {M,S}
        dict = group(ind->parity(basisstate(ind,fb)),eachindex(fb))
        sortkeys!(dict)
        oldindfromnew = vcat(dict...)
        blocksizes = collect(values(length.(dict)))
        newindfromold = map(first,sort(collect(enumerate(oldindfromnew)),by=last))
        # newindfromold = eachindex(fb)[oldindfromnew]
        indtofocklist = map(ind->basisstate(ind,fb),oldindfromnew)
        indtofock(ind) = indtofocklist[ind]
        focktoind(f) = newindfromold[index(f,fb)]
        new{M,S,typeof(indtofock),typeof(focktoind)}(fb,indtofock,focktoind,blocksizes)
    end
end
index(basisstate::Integer,b::FermionParityBasis) = b.focktoind(basisstate)
basisstate(ind::Integer,b::FermionParityBasis) = b.indtofock(ind)
Base.parent(fpb::FermionParityBasis) = fpb.fb
preimagebasis(fpb::FermionParityBasis) = preimagebasis(parent(fpb))
imagebasis(fpb::FermionParityBasis) = imagebasis(parent(fpb))
nbr_of_fermions(fpb::FermionParityBasis) = nbr_of_fermions(parent(fpb))
Base.length(fpb::FermionParityBasis) = length(parent(fpb))
siteindex(f::Fermion,b::FermionParityBasis) = siteindex(f,parent(b))
Base.eachindex(fpb::FermionParityBasis) = eachindex(parent(fpb))
blocksizes(fpb::FermionParityBasis) = (fpb.blocksizes)
blocksizes(fpb::FermionBasis) = fill(length(fpb),1)

BlockDiagonals.BlockDiagonal(op::AbstractFockOperator) = BlockDiagonal(op,preimagebasis(op),imagebasis(op))
function BlockDiagonals.BlockDiagonal(op::AbstractFockOperator,bin::AbstractBasis,bout::AbstractBasis)
    mat = Matrix(bin*op*bout)
    inblocksizes = deepcopy(blocksizes(bin))
    outblocksizes = deepcopy(blocksizes(bout))
    instarts = cumsum(pushfirst!(inblocksizes,1))
    outstarts =  cumsum(pushfirst!(outblocksizes,1))
    instartends = zip(instarts,Iterators.drop(instarts,1) .- 1)
    outstartends = zip(outstarts,Iterators.drop(outstarts,1) .- 1)
    blocks = [mat[os:oe,is:ie] for ((os,oe),(is,ie)) in zip(outstartends,instartends)]
    BlockDiagonal(blocks)
end