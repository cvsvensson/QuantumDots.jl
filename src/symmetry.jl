abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry end
# symmetry(::FermionBasis) = NoSymmetry()
symmetry(::FermionBasis) = NoSymmetry()

struct FermionParityBasis{M,S,IF,FI} <: AbstractBasis
    fb::FermionBasis{M,S}
    indtofock::IF
    focktoind::FI
    function FermionParityBasis(fb::FermionBasis{M,S}) where {M,S}
        dict = group(ind->parity(basisstate(ind,fb)),eachindex(fb))
        sortkeys!(dict)
        oldindfromnew = vcat(dict...)
        newindfromold = map(first,sort(collect(enumerate(oldindfromnew)),by=last))
        # newindfromold = eachindex(fb)[oldindfromnew]
        indtofocklist = map(ind->basisstate(ind,fb),oldindfromnew)
        indtofock(ind) = indtofocklist[ind]
        focktoind(f) = newindfromold[index(f,fb)]
        new{M,S,typeof(indtofock),typeof(focktoind)}(fb,indtofock,focktoind)
    end
end
index(basisstate::Integer,b::FermionParityBasis) = b.focktoind(basisstate)
basisstate(ind::Integer,b::FermionParityBasis) = b.indtofock(ind)
Base.parent(fpb::FermionParityBasis) = fpb.fb
preimagebasis(fpb::FermionParityBasis) = preimagebasis(parent(fpb))
imagebasis(fpb::FermionParityBasis) = imagebasis(parent(fpb))
nbr_of_fermions(fpb::FermionParityBasis) = nbr_of_fermions(parent(fpb))
length(fpb::FermionParityBasis) = length(parent(fpb))
siteindex(f::Fermion,b::FermionParityBasis) = siteindex(f,parent(b))
