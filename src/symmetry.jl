blockinds(i::Integer, sym::AbelianFockSymmetry) = blockinds(i, values(sym.qntoblocksizes))
blockinds(inds::Dims, sym::AbelianFockSymmetry) = map(i -> blockinds(i, sym), inds)
blockinds(sym::AbelianFockSymmetry) = sizestoinds(values(sym.qntoblocksizes))

qninds(qns::Tuple, sym::AbelianFockSymmetry) = map(qn -> qninds(qn, sym), qns)
qninds(qn, sym::AbelianFockSymmetry) = sym.qntoinds[qn]
blockinds(inds::Dims, sizes) = map(n -> blockinds(n, sizes), inds)
blockinds(i::Integer, sizes) = sizestoinds(sizes)[i]

sizestoinds(sizes) = accumulate((a, b) -> last(a) .+ (1:b), sizes, init=0:0)::Vector{UnitRange{Int}}
abstract type AbstractQuantumNumber end

"""
    focksymmetry(fockstates, qn)

Constructs an `AbelianFockSymmetry` object that represents the symmetry of a many-body fermionic system. 

# Arguments
- `fockstates`: The fockstates to iterate over
- `qn`: A function that takes an integer representing a fock state and returns corresponding quantum number.
"""
function focksymmetry(fockstates, qn)
    oldinds = eachindex(fockstates)
    qntooldinds = group(ind -> qn(fockstates[ind]), oldinds)
    sortkeys!(qntooldinds)
    oldindfromnew = vcat(qntooldinds...)
    blocksizes = map(length, qntooldinds)
    newindfromold = map(first, sort!(collect(enumerate(oldindfromnew)), by=last))
    indtofockdict = map(i -> fockstates[i], oldindfromnew)
    indtofock(ind) = indtofockdict[ind]
    focktoinddict = Dictionary(fockstates, newindfromold)
    qntoinds = map(oldinds -> map(oldind -> newindfromold[oldind], oldinds), qntooldinds)
    qntofockstates = map(oldinds -> fockstates[oldinds], qntooldinds)
    AbelianFockSymmetry(indtofockdict, focktoinddict, blocksizes, qntofockstates, qntoinds, qn)
end
focksymmetry(::AbstractVector, ::NoSymmetry) = NoSymmetry()
instantiate(::NoSymmetry, labels) = NoSymmetry()
indtofock(ind, sym::AbelianFockSymmetry) = FockNumber(sym.indtofockdict[ind])
focktoind(f, sym::AbelianFockSymmetry) = sym.focktoinddict[f]

"""
    fermion_sparse_matrix(fermion_number, totalsize, sym)

Constructs a sparse matrix of size `totalsize` representing a fermionic operator at bit position `fermion_number` in a many-body fermionic system with symmetry `sym`. 
"""
function fermion_sparse_matrix(fermion_number, totalsize, sym)
    ininds = 1:totalsize
    amps = Int[]
    ininds_final = Int[]
    outinds = Int[]
    sizehint!(amps, totalsize)
    sizehint!(ininds_final, totalsize)
    sizehint!(outinds, totalsize)
    for n in ininds
        f = indtofock(n, sym)
        newfockstate, amp = removefermion(fermion_number, f)
        if !iszero(amp)
            push!(amps, amp)
            push!(ininds_final, focktoind(f, sym))
            push!(outinds, focktoind(newfockstate, sym))
        end
    end
    return sparse(outinds, ininds_final, amps, totalsize, totalsize)
end


"""
    blockdiagonal(m::AbstractMatrix, basis::AbstractManyBodyBasis)

Construct a BlockDiagonal version of `m` using the symmetry of `basis`. No checking is done to ensure this is a faithful representation.
"""
blockdiagonal(m::AbstractMatrix, basis::AbstractManyBodyBasis) = blockdiagonal(m, basis.symmetry)
blockdiagonal(::Type{T}, m::AbstractMatrix, basis::AbstractManyBodyBasis) where {T} = blockdiagonal(T, m, basis.symmetry)

blockdiagonal(m::AbstractMatrix, ::NoSymmetry) = m
function blockdiagonal(m::AbstractMatrix, sym::AbelianFockSymmetry)
    blockinds = values(sym.qntoinds)
    BlockDiagonal([m[block, block] for block in blockinds])
end
function blockdiagonal(::Type{T}, m::AbstractMatrix, sym::AbelianFockSymmetry) where {T}
    blockinds = values(sym.qntoinds)
    BlockDiagonal([T(m[block, block]) for block in blockinds])
end
function blockdiagonal(m::Hermitian, sym::AbelianFockSymmetry)
    blockinds = values(sym.qntoinds)
    Hermitian(BlockDiagonal([m[block, block] for block in blockinds]))
end
function blockdiagonal(::Type{T}, m::Hermitian, sym::AbelianFockSymmetry) where {T}
    blockinds = values(sym.qntoinds)
    Hermitian(BlockDiagonal([T(m[block, block]) for block in blockinds]))
end

focktoind(fs::FockNumber, b::AbstractBasis) = focktoind(fs, symmetry(b))
indtofock(ind, b::AbstractBasis) = indtofock(ind, symmetry(b))

focktoind(fs::FockNumber, ::NoSymmetry) = fs.f + 1
indtofock(ind, ::NoSymmetry) = FockNumber(ind - 1)

function nextfockstate_with_same_number(v)
    #http://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    t = (v | (v - 1)) + 1
    t | (((div((t & -t), (v & -v))) >> 1) - 1)
end
"""
    fockstates(M, n)

Generate a list of Fock states with `n` occupied fermions in a system with `M` different fermions.
"""
function fockstates(M, n)
    v::Int = focknbr_from_bits(ntuple(i -> true, n)).f
    maxv = v * 2^(M - n)
    states = Vector{FockNumber}(undef, binomial(M, n))
    count = 1
    while v <= maxv
        states[count] = FockNumber(v)
        v = nextfockstate_with_same_number(v)
        count += 1
    end
    states
end

struct FermionConservation <: AbstractSymmetry end
struct FermionSubsetConservation{M} <: AbstractSymmetry
    mask::M
end
struct UninstantiatedFermionSubsetConservation{L} <: AbstractSymmetry
    labels::L
end
FermionSubsetConservation(::Nothing) = NoSymmetry()
FermionConservation(labels, jw::JordanWignerOrdering) = FermionSubsetConservation(focknbr_from_site_labels(labels, jw))
FermionConservation(labels) = UninstantiatedFermionSubsetConservation(labels)
instantiate(qn::UninstantiatedFermionSubsetConservation, jw::JordanWignerOrdering) = FermionConservation(qn.labels, jw)
instantiate(qn::FermionSubsetConservation, ::JordanWignerOrdering) = qn
instantiate(qn::FermionConservation, ::JordanWignerOrdering) = qn

(qn::FermionSubsetConservation)(fs) = fermionnumber(fs, qn.mask)
(qn::FermionConservation)(fs) = fermionnumber(fs)

@testitem "ConservedFermions" begin
    labels = 1:4
    conservedlabels = 1:4
    qn = FermionConservation(conservedlabels) 
    c1 = FermionBasis(labels; qn)
    c2 = FermionBasis(labels; qn=QuantumDots.fermionnumber)
    @test all(c1 == c2 for (c1, c2) in zip(c1, c2))
    c1 = FermionBasis(labels)
    c2 = FermionBasis(labels; qn=FermionConservation(()))
    @test all(c1 == c2 for (c1, c2) in zip(c1, c2))

    conservedlabels = 2:2
    qn = FermionConservation(conservedlabels)
    c1 = FermionBasis(labels; qn)
    @test all(c1.symmetry.qntoblocksizes .== 2^(length(labels) - length(conservedlabels)))
end

struct ProductSymmetry{T} <: AbstractSymmetry
    symmetries::T
end
instantiate(qn::ProductSymmetry, labels) = prod(instantiate(sym, labels) for sym in qn.symmetries)
(qn::ProductSymmetry)(fs) = map(sym -> sym(fs), qn.symmetries)
Base.:*(sym1::AbstractSymmetry, sym2::AbstractSymmetry) = ProductSymmetry((sym1, sym2))
Base.:*(sym1::AbstractSymmetry, sym2::ProductSymmetry) = ProductSymmetry((sym1, sym2.symmetries...))
Base.:*(sym1::ProductSymmetry, sym2::AbstractSymmetry) = ProductSymmetry((sym1.symmetries..., sym2))
Base.:*(sym1::ProductSymmetry, sym2::ProductSymmetry) = ProductSymmetry((sym1.symmetries..., sym2.symmetries...))

struct ParityConservation <: AbstractSymmetry end
(qn::ParityConservation)(fs) = parity(fs)
instantiate(qn::ParityConservation, labels) = qn

@testitem "ProductSymmetry" begin
    labels = 1:4
    qn = FermionConservation() * ParityConservation()
    c = FermionBasis(labels; qn)
    @test keys(c.symmetry.qntoinds).values == [(n, (-1)^n) for n in 0:4]
    qn = prod(FermionConservation([l], c.jw) for l in labels)
    @test all(FermionBasis(labels; qn).symmetry.qntoblocksizes .== 1)
end

struct IndexConservation{L} <: AbstractSymmetry
    labels::L
end
instantiate(qn::IndexConservation, jw::JordanWignerOrdering) = IndexConservation(qn.labels, jw)
IndexConservation(index, jw::JordanWignerOrdering) = FermionConservation(filter(label -> index in label || index == label, jw.labels), jw)
@testitem "IndexConservation" begin
    labels = 1:4
    qn = IndexConservation(1)
    qn2 = FermionConservation(1:1)
    c = FermionBasis(labels; qn)
    c2 = FermionBasis(labels; qn=qn2)
    @test all(c == c2 for (c, c2) in zip(c, c2))

    spatial_labels = 1:1
    spin_labels = (:↑, :↓)
    all_labels = collect(Base.product(spatial_labels, spin_labels))[:]
    qn = IndexConservation(:↑) * IndexConservation(:↓)
    c = FermionBasis(all_labels; qn)
    @test all(c.symmetry.qntoblocksizes .== 1)
end

instantiate(f::F, labels) where {F} = f