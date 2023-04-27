abstract type AbstractBasis end
const BasisOrMissing = Union{AbstractBasis,Missing}
basis(::AbstractArray) = missing

abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry 
    size::Union{Int,Missing}
end
Base.size(sym::NoSymmetry) = sym.size

struct Fermion{S}
    id::S
end
id(f::Fermion) = f.id

struct FermionBasis{M,S,T,Sym} <: AbstractBasis
    dict::Dictionary{S,T}
    symmetry::Sym
    function FermionBasis(fermions::NTuple{M,Fermion{S}}, sym::Sym=NoSymmetry(2^M)) where {M,S,Sym<:AbstractSymmetry}
        reps = ntuple(n->fermion_sparse_matrix(n,sym),M)
        new{M,S,eltype(reps),Sym}(Dictionary(map(id,fermions), reps), sym)
    end
end
Base.getindex(b::FermionBasis,i) = b.dict[i]
Base.getindex(b::FermionBasis,args...) = b.dict[args]
Base.keys(b::FermionBasis) = keys(b.dict)

Base.show(io::IO, ::MIME"text/plain", b::FermionBasis) = show(io,b)
Base.show(io::IO, b::FermionBasis{M,S,T,Sym}) where {M,S,T,Sym} = print(io, "FermionBasis{$M,$S,$T,$Sym}:\nkeys = ", keys(b))

NoSymmetry() = NoSymmetry(missing)
symmetry(::NTuple{M},::NoSymmetry) where M = NoSymmetry(2^M)
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
Base.size(sym::AbelianFockSymmetry) = length(sym.indtofockdict)
# function FermionBasis(fermions::NTuple{M,Fermion{S}}, sym::Z2Symmetry{M}) where {M,S,Sym<:AbstractSymmetry}

#     reps = ntuple(n->fermion_sparse_matrix(n,2^M,sym),M)
#     # FermionBasis(fermionids,reps,sym)
#     new{M,S,eltype(reps),Sym}(Dictionary(map(id,fermions), reps), sym)
# end

struct QArray{N,QNs,A,S}
    blocks::Dictionary{QNs,A}
    symmetry::NTuple{N,S}
    dirs::NTuple{N,Bool}
    function QArray(blocks::Dictionary{QNs,A}, sym::NTuple{N,S},dirs = ntuple(i->false,N)) where {A<:AbstractArray,QNs, S,N}
        new{ndims(A) ,QNs,A, S}(blocks, sym, dirs)
    end
end