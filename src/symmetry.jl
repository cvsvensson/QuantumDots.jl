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
    symmetry(M::Int, qn)

Constructs an `AbelianFockSymmetry` object that represents the symmetry of a many-body fermionic system. 

# Arguments
- `M::Int`: The number of fermions in the system.
- `qn`: A function that takes an integer representing a fock state and returns corresponding quantum number.
"""
function symmetry(M::Int, qn)
    qntooldinds = group(ind -> qn(ind - 1), 1:2^M)
    sortkeys!(qntooldinds)
    oldindfromnew = vcat(qntooldinds...)
    blocksizes = map(length, qntooldinds)
    newindfromold = map(first, sort(collect(enumerate(oldindfromnew)), by=last))
    indtofockdict = oldindfromnew .- 1
    indtofock(ind) = indtofockdict[ind]
    focktoinddict = Dictionary(0:(2^M-1), newindfromold)
    qntoinds = map(oldinds -> map(oldind -> newindfromold[oldind], oldinds), qntooldinds)
    qntofockstates = map(oldinds -> oldinds .- 1, qntooldinds)
    AbelianFockSymmetry(indtofockdict, focktoinddict, blocksizes, qntofockstates, qntoinds, qn)
end
symmetry(M::Int, ::NoSymmetry) = NoSymmetry()

indtofock(ind, sym::AbelianFockSymmetry) = sym.indtofockdict[ind]
focktoind(f, sym::AbelianFockSymmetry) = sym.focktoinddict[f]

"""
    fermion_sparse_matrix(fermion_number, totalsize, sym)

Constructs a sparse matrix of size `totalsize` representing a fermionic operator at bit position `fermion_number` in a many-body fermionic system with symmetry `sym`. 
"""
function fermion_sparse_matrix(fermion_number, totalsize, sym)
    mat = spzeros(Int, totalsize, totalsize)
    _fill!(mat, fs -> removefermion(fermion_number, fs), sym)
    mat
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