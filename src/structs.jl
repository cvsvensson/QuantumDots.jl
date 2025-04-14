abstract type AbstractBasis end
abstract type AbstractManyBodyBasis <: AbstractBasis end

abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry end

struct FockNumber
    f::Int
end
FockNumber(f::FockNumber) = f
struct JordanWignerOrdering{L}
    labels::Vector{L}
    ordering::OrderedDict{L,Int}
    function JordanWignerOrdering(labels)
        ls = collect(labels)
        dict = OrderedDict(zip(ls, Base.OneTo(length(ls))))
        new{eltype(ls)}(ls, dict)
    end
end
Base.length(jw::JordanWignerOrdering) = length(jw.labels)
Base.:(==)(jw1::JordanWignerOrdering, jw2::JordanWignerOrdering) = jw1.labels == jw2.labels && jw1.ordering == jw2.ordering
Base.keys(jw::JordanWignerOrdering) = jw.labels

struct FermionBasisTemplate{L,S}
    jw::JordanWignerOrdering{L}
    sym::S
end
Base.keys(b::FermionBasisTemplate) = keys(b.jw)
indtofock(ind, b::FermionBasisTemplate) = indtofock(ind, b.sym)

"""
    struct FermionBasis{M,D,Sym,L} <: AbstractManyBodyBasis

Fermion basis for representing many-body fermions.

## Fields
- `dict::OrderedDict`: A dictionary that maps fermion labels to a representation of the fermion.
- `symmetry::Sym`: The symmetry of the basis.
- `jw::JordanWignerOrdering{L}`: The Jordan-Wigner ordering of the basis.
"""
struct FermionBasis{M,D,Sym,L} <: AbstractManyBodyBasis
    dict::D
    symmetry::Sym
    jw::JordanWignerOrdering{L}
end
function FermionBasis(iters...; qn=NoSymmetry(), kwargs...)
    labels = handle_labels(iters...)
    labelvec = collect(labels)[:]
    jw = JordanWignerOrdering(labelvec)
    fockstates = map(FockNumber, get(kwargs, :fockstates, 0:2^length(labels)-1))
    M = length(labels)
    labelled_symmetry = instantiate(qn, jw)
    sym_concrete = focksymmetry(fockstates, labelled_symmetry)
    # sym_more_concrete = symmetry(fockstates, sym_concrete)
    reps = ntuple(n -> fermion_sparse_matrix(n, length(fockstates), sym_concrete), M)
    d = OrderedDict(zip(labelvec, reps))
    FermionBasis{M,typeof(d),typeof(sym_concrete),_label_type(jw)}(d, sym_concrete, jw)
end
Base.getindex(b::FermionBasis, i) = b.dict[i]
Base.getindex(b::FermionBasis, args...) = b.dict[args]
Base.keys(b::FermionBasis) = keys(b.dict)
Base.show(io::IO, ::MIME"text/plain", b::FermionBasis) = show(io, b)
Base.show(io::IO, b::FermionBasis{M,D,Sym}) where {M,D,Sym} = print(io, "FermionBasis{$M,$D,$Sym}:\nkeys = ", keys(b))
Base.iterate(b::FermionBasis) = iterate(values(b.dict))
Base.iterate(b::FermionBasis, state) = iterate(values(b.dict), state)
Base.length(::FermionBasis{M}) where {M} = M
Base.eltype(b::FermionBasis) = eltype(b.dict)
Base.keytype(b::FermionBasis) = keytype(b.dict)
symmetry(b::FermionBasis) = b.symmetry

handle_labels(iter, iters...) = Base.product(iter, iters...)
handle_labels(iter) = iter
nbr_of_modes(::FermionBasis{M}) where {M} = M

function Base.:(==)(b1::FermionBasis, b2::FermionBasis)
    if b1 === b2
        return true
    end
    if b1.jw != b2.jw
        return false
    end
    if b1.symmetry != b2.symmetry
        return false
    end
    if b1.dict != b2.dict
        return false
    end
    return true
end

"""
    struct AbelianFockSymmetry{IF,FI,QN,QNfunc} <: AbstractSymmetry

AbelianFockSymmetry represents a symmetry that is diagonal in fock space, i.e. particle number conservation, parity, spin consvervation.

## Fields
- `indtofockdict::IF`: A dictionary mapping indices to Fock states.
- `focktoinddict::FI`: A dictionary mapping Fock states to indices.
- `qntoblocksizes::Dictionary{QN,Int}`: A dictionary mapping quantum numbers to block sizes.
- `qntofockstates::Dictionary{QN,Vector{Int}}`: A dictionary mapping quantum numbers to Fock states.
- `qntoinds::Dictionary{QN,Vector{Int}}`: A dictionary mapping quantum numbers to indices.
- `conserved_quantity::QNfunc`: A function that computes the conserved quantity from a fock number.
"""
struct AbelianFockSymmetry{IF,FI,QN,QNfunc} <: AbstractSymmetry
    indtofockdict::IF
    focktoinddict::FI
    qntoblocksizes::Dictionary{QN,Int}
    qntofockstates::Dictionary{QN,Vector{FockNumber}}
    qntoinds::Dictionary{QN,Vector{Int}}
    conserved_quantity::QNfunc
end

Base.:(==)(sym1::AbelianFockSymmetry, sym2::AbelianFockSymmetry) = sym1.indtofockdict == sym2.indtofockdict && sym1.focktoinddict == sym2.focktoinddict && sym1.qntoblocksizes == sym2.qntoblocksizes && sym1.qntofockstates == sym2.qntofockstates && sym1.qntoinds == sym2.qntoinds && sym1.conserved_quantity == sym2.conserved_quantity

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
