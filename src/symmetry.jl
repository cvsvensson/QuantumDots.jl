abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry end
# symmetry(::FermionBasis) = NoSymmetry()
symmetry(::FermionBasis) = NoSymmetry()
using Dictionaries

struct FermionParityBasis{M,S} <: AbstractBasis
    fb::FermionBasis{M,S}
    indtofock::Array
    focktoind::Array
    function FermionParityBasis(fb::FermionBasis{M,S}) where {M,S}
        eveninds = Int[]
        oddinds = Int[]
        dict = group(Iterators.map(ind->parity(basisstate(ind,fb)),eachindex(fb)))
        sortkeys!(dict)
        oldindfromnew = vcat(dict...)
        indtofock = map(ind->basisstate(ind,fb),oldindfromnew)
        newfromold = map(findfirst()
        focktoind = map(ind->basisstate(ind,fb),oldindfromnew)
        for ind in eachindex(fb)
            
            push!(eveninds)
        end
        new{M,S}(fb,)
    end
end

function blockify(f)
    group(f,0:2^4-1)
end

Base.parent(fpb::FermionParityBasis) = fpb.fb
preimagebasis(fpb::FermionParityBasis) = preimagebasis(parent(fpb))
imagebasis(fpb::FermionParityBasis) = imagebasis(parent(fpb))


# focknbr(ind::Integer,::Val{S},::NoSymmetry) where S = ind-1
# index(::NoSymmetry,basisstate::Integer,::FermionBasis) = basisstate+1
# basisstate(::NoSymmetry,ind::Integer,::FermionBasis) = ind-1

struct ParitySymmetry <: AbstractSymmetry end

# struct QN{F}
#     func::F
# end

# using BlockArrays
using BlockDiagonals

function blocks(m,by)

end

bm = BlockDiagonal([rand(2^8, 2^8), rand(2^8, 2^8)])