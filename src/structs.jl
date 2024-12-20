abstract type AbstractBasis end
abstract type AbstractManyBodyBasis <: AbstractBasis end

abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry end

"""
    struct FermionBasis{M,S,T,Sym} <: AbstractManyBodyBasis

Fermion basis for representing many-body fermions.

## Fields
- `dict::OrderedDict`: A dictionary that maps fermion labels to a representation of the fermion.
- `symmetry::Sym`: The symmetry of the basis.
"""
struct FermionBasis{M,D,Sym} <: AbstractManyBodyBasis
    dict::D
    symmetry::Sym
end
function FermionBasis(iters...; qn=NoSymmetry(), kwargs...)
    labels = handle_labels(iters...)
    fockstates = get(kwargs, :fockstates, 0:2^length(labels)-1)
    M = length(labels)
    labelled_symmetry = instantiate(qn, labels)
    sym_concrete = focksymmetry(fockstates, labelled_symmetry)
    # sym_more_concrete = symmetry(fockstates, sym_concrete)
    reps = ntuple(n -> fermion_sparse_matrix(n, length(fockstates), sym_concrete), M)
    d = OrderedDict(zip(labels, reps))
    FermionBasis{M,typeof(d),typeof(sym_concrete)}(d, sym_concrete)
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
nbr_of_fermions(::FermionBasis{M}) where {M} = M


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
    qntofockstates::Dictionary{QN,Vector{Int}}
    qntoinds::Dictionary{QN,Vector{Int}}
    conserved_quantity::QNfunc
end

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
    N = nbr_of_fermions(b)
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
nbr_of_fermions(::FermionBdGBasis{M}) where {M} = M
Base.getindex(b::FermionBdGBasis, i) = BdGFermion(i, b)
Base.getindex(b::FermionBdGBasis, args...) = BdGFermion(args, b)
indexpos(f::BdGFermion, b::FermionBdGBasis) = b.position[f.id] + !f.hole * nbr_of_fermions(b)

Base.iterate(b::FermionBdGBasis) = ((result, state) = iterate(keys(b.position)); (b[result], state));
Base.iterate(b::FermionBdGBasis, state) = (res = iterate(keys(b.position), state); isnothing(res) ? nothing : (b[res[1]], res[2]))
Base.length(::FermionBdGBasis{M}) where {M} = M


abstract type AbstractDiagonalHamiltonian end
