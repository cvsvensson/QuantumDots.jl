focknbr_from_bits(bits) = mapreduce(nb -> nb[2] * (1 << (nb[1] - 1)), +, enumerate(bits))
focknbr_from_site_index(site::Integer) = 1 << (site - 1)
focknbr_from_site_indices(sites) = mapreduce(focknbr_from_site_index, +, sites, init=0)
# focknbr(sites::NTuple{N,<:Integer}) where {N} = mapreduce(site -> 1 << (site - 1), +, sites)

bits(s::Integer, N) = digits(Bool, s, base=2, pad=N)
parity(fs::Int) = iseven(fermionnumber(fs)) ? 1 : -1
fermionnumber(fs::Int) = count_ones(fs)

fermionnumber(fs::Int, mask) = count_ones(fs & mask)
fermionnumber(sublabels, labels) = Base.Fix2(fermionnumber, focknbr_from_site_indices(siteindices(sublabels, labels)))

function insert_bits(x::Int, positions)
    result = 0
    bit_index = 1
    for pos in positions
        if x & (1 << (bit_index - 1)) != 0
            result |= (1 << (pos - 1))
        end
        bit_index += 1
    end
    return result
end

"""
    siteindex(id, labels)

Find the index of the first occurrence of `id` in `labels`.
"""
siteindex(id, labels)::Int = findfirst(x -> x == id, labels)
"""
    siteindex(id, b::AbstractBasis)

Find the index of the first occurrence of `id` in the labels in basis `b`.
"""
siteindex(id, b::AbstractBasis) = siteindex(id, collect(keys(b)))

"""
    siteindices(ids, b::AbstractBasis)

Return the site indices corresponding to the given `ids` in the labels of the basis `b`.
"""
siteindices(ids, b::AbstractBasis) = map(id -> siteindex(id, b), ids)
"""
    siteindices(ids, b::AbstractBasis)

Return the site indices corresponding to the given `ids` in the labels.
"""
siteindices(ids, labels) = map(id -> siteindex(id, labels), ids)

"""
    tensor(v::AbstractVector, b::AbstractBasis)

Return a tensor representation of the vector `v` in the basis `b`, with one index for each site.
"""
function tensor(v::AbstractVector{T}, b::AbstractBasis) where {T}
    M = length(b)
    @assert length(v) == 2^M
    t = Array{T,M}(undef, ntuple(i -> 2, M))
    for I in CartesianIndices(t)
        fs = focknbr_from_bits(Bool.(Tuple(I) .- 1))
        t[I] = v[focktoind(fs, b)] #* parity(fs)
    end
    return t
end
##https://iopscience.iop.org/article/10.1088/1751-8121/ac0646/pdf (10c)
_bit(f, k) = Bool((f >> (k - 1)) & 1)
function phase_factor(focknbr1, focknbr2, subinds::NTuple)::Int
    bitmask = focknbr_from_site_indices(subinds)
    prod(i -> (jwstring_left(i, bitmask & focknbr1) * jwstring_left(i, bitmask & focknbr2))^_bit(focknbr2, i), subinds, init=1)
end
function phase_factor(focknbr1, focknbr2, N)::Int
    prod(_phase_factor(focknbr1, focknbr2, i) for i in 1:N; init=1)
end

function _phase_factor(focknbr1, focknbr2, i)::Int
    _bit(focknbr2, i) ? (jwstring_left(i, focknbr1) * jwstring_left(i, focknbr2)) : 1
end

"""
    partial_trace(v::AbstractMatrix, bsub::AbstractBasis, bfull::AbstractBasis)

Compute the partial trace of a matrix `v`, leaving the subsystem defined by the basis `bsub`.
"""
partial_trace(v::AbstractMatrix, bsub::AbstractBasis, bfull::AbstractBasis) = partial_trace(v, Tuple(keys(bsub)), bfull, symmetry(bsub))

partial_trace(v::AbstractVector, args...) = partial_trace(v * v', args...)

"""
    partial_trace(m::AbstractMatrix, labels, b::FermionBasis, sym::AbstractSymmetry=NoSymmetry())

Compute the partial trace of a matrix `m` in basis `b`, leaving only the subsystems specified by `labels`. `sym` determines the ordering of the basis states.
"""
function partial_trace(m::AbstractMatrix{T}, labels, b::AbstractBasis, sym::AbstractSymmetry=NoSymmetry()) where {T}
    N = length(labels)
    mout = zeros(T, 2^N, 2^N)
    partial_trace!(mout, m, labels, b, sym)
end

"""
    partial_trace!(mout, m::AbstractMatrix, labels, b::FermionBasis, sym::AbstractSymmetry=NoSymmetry())

Compute the partial trace of a matrix `m` in basis `b`, leaving only the subsystems specified by `labels`. The result is stored in `mout`, and `sym` determines the ordering of the basis states.
"""
function partial_trace!(mout, m::AbstractMatrix{T}, labels, b::FermionBasis{M}, sym::AbstractSymmetry=NoSymmetry()) where {T,M}
    N = length(labels)
    fill!(mout, zero(eltype(mout)))
    outinds = siteindices(labels, b) #::NTuple{N,Int}
    @assert all(outinds[n] > outinds[n-1] for n in Iterators.drop(eachindex(outinds), 1)) "Subsystems must be ordered in the same way as the full system" #Is this true?
    bitmask = 2^M - 1 - focknbr_from_site_indices(outinds)
    outbits(f) = map(i -> _bit(f, i), outinds)
    for f1 in UnitRange{UInt64}(0, 2^M - 1), f2 in UnitRange{UInt64}(0, 2^M - 1)
        if (f1 & bitmask) != (f2 & bitmask)
            continue
        end
        newfocknbr1 = focknbr_from_bits(outbits(f1))
        newfocknbr2 = focknbr_from_bits(outbits(f2))
        s1 = phase_factor(f1, f2, M)
        s2 = phase_factor(newfocknbr1, newfocknbr2, N)
        s = s2 * s1
        mout[focktoind(newfocknbr1, sym), focktoind(newfocknbr2, sym)] += s * m[focktoind(f1, b), focktoind(f2, b)]
    end
    return mout
end

"""
    partial_trace!(mout, m::AbstractMatrix, labels, b::FermionBasis, sym::AbstractSymmetry=NoSymmetry())

Compute the partial trace of a matrix `m` in basis `b`, leaving only the subsystems specified by `labels`. The result is stored in `mout`, and `sym` determines the ordering of the basis states.
"""
function partial_transpose(m::AbstractMatrix{T}, labels, b::FermionBasis{M}) where {T,M}
    mout = zero(m)
    partial_transpose!(mout, m, labels, b)
end
function partial_transpose!(mout, m::AbstractMatrix{T}, labels, b::FermionBasis{M}) where {T,M}
    fill!(mout, zero(eltype(mout)))
    outinds = siteindices(labels, b)
    bitmask = 2^M - 1 - focknbr_from_site_indices(outinds)
    outbits(f) = map(i -> _bit(f, i), outinds)
    for f1 in UnitRange{UInt64}(0, 2^M - 1)
        f1R = (f1 & bitmask)
        f1L = (f1 & ~bitmask)
        for f2 in UnitRange{UInt64}(0, 2^M - 1)
            f2R = (f2 & bitmask)
            f2L = (f2 & ~bitmask)
            newfocknbr1 = f2L + f1R
            newfocknbr2 = f1L + f2R
            s1 = phase_factor(f1, f2, M)
            s2 = phase_factor(newfocknbr1, newfocknbr2, M)
            s = s2 * s1
            mout[focktoind(newfocknbr1, b), focktoind(newfocknbr2, b)] = s * m[focktoind(f1, b), focktoind(f2, b)]
        end
    end
    return mout
end

@testitem "Partial transpose" begin
    using LinearAlgebra
    import QuantumDots: partial_transpose
    qn = ParityConservation()
    c1 = FermionBasis(1:1; qn)
    c2 = FermionBasis(2:2; qn)
    c12 = FermionBasis(1:2; qn)

    A = rand(ComplexF64, 2, 2)
    B = rand(ComplexF64, 2, 2)
    C = wedge((A, B), (c1, c2), c12)
    Cpt = partial_transpose(C, (1,), c12)
    Cpt2 = wedge((transpose(A), B), (c1, c2), c12)
    @test Cpt ≈ Cpt2

    ## Larger system
    labels = 1:4
    N = length(labels)
    cN = FermionBasis(labels; qn)
    cs = [FermionBasis(i:i; qn) for i in labels]
    Ms = [rand(ComplexF64, 2, 2) for _ in labels]
    M = wedge(Ms, cs, cN)

    single_subsystems = [(i,) for i in 1:4]
    for (k,) in single_subsystems
        Mpt = partial_transpose(M, (k,), cN)
        Mpt2 = wedge([(n == k) ? transpose(M) : M for (n, M) in enumerate(Ms)], cs, cN)
        @test Mpt ≈ Mpt2
    end
    pair_iterator = [(i, j) for i in 1:4, j in 1:4 if i != j]
    triple_iterator = [(i, j, k) for i in 1:4, j in 1:4, k in 1:4 if length(unique((i, j, k))) == 3]
    for (i, j) in pair_iterator
        Mpt = partial_transpose(M, (i, j), cN)
        Mpt2 = wedge([(n == i || n == j) ? transpose(M) : M for (n, M) in enumerate(Ms)], cs, cN)
        @test Mpt ≈ Mpt2
    end
    for (i, j, k) in triple_iterator
        Mpt = partial_transpose(M, (i, j, k), cN)
        Mpt2 = wedge([(n == i || n == j || n == k) ? transpose(M) : M for (n, M) in enumerate(Ms)], cs, cN)
        @test Mpt ≈ Mpt2
    end
    Mpt = partial_transpose(M, labels, cN)
    Mpt2 = wedge([transpose(M) for M in Ms], cs, cN)
    @test Mpt ≈ Mpt2
    @test !(Mpt ≈ transpose(M)) # partial transpose is not the same as transpose even if the full system is transposed
    @test M ≈ partial_transpose(M, (), cN)

    M = rand(ComplexF64, 2^N, 2^N) 
    pt(l, M) = partial_transpose(M, l, cN)
    for (i, j) in pair_iterator
        @test pt((i, j), M) ≈ pt((j, i), M) ≈ pt(j, pt(i, M)) ≈ pt(i, pt(j, M))
    end
    for (i, j, k) in triple_iterator
        @test pt((i, j, k), M) ≈ pt((j, i, k), M) ≈ pt((j, k, i), M) ≈ pt((k, j, i), M) ≈ pt((k, i, j), M) ≈ pt((i, k, j), M) ≈ pt(i, pt(j, pt(k, M))) ≈ pt(j, pt(i, pt(k, M))) ≈ pt(k, pt(j, pt(i, M)))
    end

end


function reshape_to_matrix(t::AbstractArray{<:Any,N}, leftindices::NTuple{NL,Int}) where {N,NL}
    rightindices::NTuple{N - NL,Int} = Tuple(setdiff(ntuple(identity, N), leftindices))
    reshape_to_matrix(t, leftindices, rightindices)
end
function reshape_to_matrix(t::AbstractArray{<:Any,N}, leftindices::NTuple{NL,Int}, rightindices::NTuple{NR,Int}) where {N,NL,NR}
    @assert NL + NR == N
    tperm = permutedims(t, (leftindices..., rightindices...))
    lsize = prod(i -> size(tperm, i), leftindices, init=1)
    rsize = prod(i -> size(tperm, i), rightindices, init=1)
    reshape(tperm, lsize, rsize)
end

function LinearAlgebra.svd(v::AbstractVector, leftlabels::NTuple, b::AbstractBasis)
    linds = siteindices(leftlabels, b)
    t = tensor(v, b)
    svd(reshape_to_matrix(t, linds))
end