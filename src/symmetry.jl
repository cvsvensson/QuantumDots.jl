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
    symmetry(fockstates, qn)

Constructs an `AbelianFockSymmetry` object that represents the symmetry of a many-body fermionic system. 

# Arguments
- `fockstates`: The fockstates to iterate over
- `qn`: A function that takes an integer representing a fock state and returns corresponding quantum number.
"""
function symmetry(fockstates::AbstractVector, qn)
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
symmetry(fs::AbstractVector, ::NoSymmetry) = NoSymmetry()

indtofock(ind, sym::AbelianFockSymmetry) = sym.indtofockdict[ind]
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

focktoind(fs, b::AbstractBasis) = focktoind(fs, symmetry(b))
indtofock(ind, b::AbstractBasis) = indtofock(ind, symmetry(b))

focktoind(fs, ::NoSymmetry) = fs + 1
indtofock(ind, ::NoSymmetry) = ind - 1

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
    v::Int = focknbr_from_bits(ntuple(i -> true, n))
    maxv = v * 2^(M - n)
    states = Vector{Int}(undef, binomial(M, n))
    count = 1
    while v <= maxv
        states[count] = v
        v = nextfockstate_with_same_number(v)
        count += 1
    end
    states
end

struct FermionConservation <: AbstractSymmetry end
struct FermionSubsetConservation{M} <: AbstractSymmetry
    mask::M
end
FermionConservation(labels, all_labels) = FermionSubsetConservation(focknbr_from_site_indices(siteindices(labels, all_labels)))
(qn::FermionSubsetConservation)(fs) = fermionnumber(fs, qn.mask)
(qn::FermionConservation)(fs) = fermionnumber(fs)

@testitem "ConservedFermions" begin
    labels = 1:4
    conservedlabels = 1:4
    qn = FermionConservation(conservedlabels, labels)
    c1 = FermionBasis(labels; qn)
    c2 = FermionBasis(labels; qn=QuantumDots.fermionnumber)
    @test all(c1 == c2 for (c1, c2) in zip(c1, c2))
    c1 = FermionBasis(labels)
    c2 = FermionBasis(labels; qn=FermionConservation((), labels))
    @test all(c1 == c2 for (c1, c2) in zip(c1, c2))

    conservedlabels = 2:2
    qn = FermionConservation(conservedlabels, labels)
    c1 = FermionBasis(labels; qn)
    @test all(c1.symmetry.qntoblocksizes .== 2^(length(labels) - length(conservedlabels)))
end

struct ProductSymmetry{T} <: AbstractSymmetry
    symmetries::T
end
(qn::ProductSymmetry)(fs) = map(sym -> sym(fs), qn.symmetries)
Base.:*(sym1::AbstractSymmetry, sym2::AbstractSymmetry) = ProductSymmetry((sym1, sym2))
Base.:*(sym1::AbstractSymmetry, sym2::ProductSymmetry) = ProductSymmetry((sym1, sym2.symmetries...))
Base.:*(sym1::ProductSymmetry, sym2::AbstractSymmetry) = ProductSymmetry((sym1.symmetries..., sym2))
Base.:*(sym1::ProductSymmetry, sym2::ProductSymmetry) = ProductSymmetry((sym1.symmetries..., sym2.symmetries...))

struct ParityConservation <: AbstractSymmetry end
(qn::ParityConservation)(fs) = parity(fs)

@testitem "ProductSymmetry" begin
    labels = 1:4
    qn = FermionConservation() * ParityConservation()
    c = FermionBasis(labels; qn)
    @test keys(c.symmetry.qntoinds).values == [(n, (-1)^n) for n in 0:4]
    qn = prod(FermionConservation([l],labels) for l in labels)
    @test all(FermionBasis(labels; qn).symmetry.qntoblocksizes .== 1)
end
