
siteindex(label, ordering::JordanWignerOrdering) = ordering.ordering[label]
siteindex(label, b::AbstractManyBodyBasis) = siteindex(label, b.jw)
siteindices(labels, jw::JordanWignerOrdering) = map(Base.Fix2(siteindex, jw), labels)
siteindices(labels, b::AbstractManyBodyBasis) = siteindices(labels, b.jw)

label_at_site(n, ordering::JordanWignerOrdering) = ordering.labels[n]
_label_type(::JordanWignerOrdering{L}) where {L} = L
focknbr_from_site_label(label, jw) = focknbr_from_site_index(siteindex(label, jw))
focknbr_from_site_labels(labels, jw) = mapreduce(Base.Fix2(focknbr_from_site_label, jw), +, labels, init=FockNumber(0))

Base.:+(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f + f2.f)
Base.:-(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f - f2.f)
Base.:⊻(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f ⊻ f2.f)
Base.:&(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f & f2.f)
Base.:&(f1::Integer, f2::FockNumber) = FockNumber(f1 & f2.f)
Base.:|(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f | f2.f)
Base.iszero(f::FockNumber) = iszero(f.f)
Base.:*(b::Bool, f::FockNumber) = FockNumber(b * f.f)
Base.:~(f::FockNumber) = FockNumber(~f.f)

focknbr_from_bits(bits) = mapreduce(nb -> FockNumber(nb[2] * (1 << (nb[1] - 1))), +, enumerate(bits))
focknbr_from_site_index(site::Integer) = FockNumber(1 << (site - 1))
focknbr_from_site_indices(sites) = mapreduce(focknbr_from_site_index, +, sites, init=FockNumber(0))

bits(f::FockNumber, N) = digits(Bool, f.f, base=2, pad=N)
parity(f::FockNumber) = iseven(fermionnumber(f)) ? 1 : -1
fermionnumber(f::FockNumber) = count_ones(f)
Base.count_ones(f::FockNumber) = count_ones(f.f)

fermionnumber(fs::FockNumber, mask) = count_ones(fs & mask)

"""
    jwstring(site, focknbr)
    
Parity of the number of fermions to the right of site.
"""
jwstring(site, focknbr) = jwstring_left(site, focknbr)
jwstring_anti(site, focknbr) = jwstring_right(site, focknbr)
jwstring_right(site, focknbr::FockNumber) = iseven(count_ones(focknbr.f >> site)) ? 1 : -1
jwstring_left(site, focknbr::FockNumber) = iseven(count_ones(focknbr.f) - count_ones(focknbr.f >> (site - 1))) ? 1 : -1

function insert_bits(_x::FockNumber, positions)
    x = _x.f
    result = 0
    bit_index = 1
    for pos in positions
        if x & (1 << (bit_index - 1)) != 0
            result |= (1 << (pos - 1))
        end
        bit_index += 1
    end
    return FockNumber(result)
end

@testitem "Fock" begin
    using Random
    Random.seed!(1234)

    N = 6
    focknumber = FockNumber(20) # = 16+4 = 00101
    fbits = bits(focknumber, N)
    @test fbits == [0, 0, 1, 0, 1, 0]

    @test QuantumDots.focknbr_from_bits(fbits) == focknumber
    @test QuantumDots.focknbr_from_bits(Tuple(fbits)) == focknumber
    @test !QuantumDots._bit(focknumber, 1)
    @test !QuantumDots._bit(focknumber, 2)
    @test QuantumDots._bit(focknumber, 3)
    @test !QuantumDots._bit(focknumber, 4)
    @test QuantumDots._bit(focknumber, 5)

    @test QuantumDots.focknbr_from_site_indices((3, 5)) == focknumber
    @test QuantumDots.focknbr_from_site_indices([3, 5]) == focknumber

    @testset "removefermion" begin
        focknbr = FockNumber(rand(1:2^N) - 1)
        fockbits = bits(focknbr, N)
        function test_remove(n)
            QuantumDots.removefermion(n, focknbr) == (fockbits[n] ? (focknbr - FockNumber(2^(n - 1)), (-1)^sum(fockbits[1:n-1])) : (FockNumber(0), 0))
        end
        @test all([test_remove(n) for n in 1:N])
    end

    @testset "ToggleFermions" begin
        focknbr = FockNumber(177) # = 1000 1101, msb to the right
        digitpositions = Vector([7, 8, 2, 3])
        daggers = BitVector([1, 0, 1, 1])
        newfocknbr, sign = QuantumDots.togglefermions(digitpositions, daggers, focknbr)
        @test newfocknbr == FockNumber(119) # = 1110 1110
        @test sign == 1
        # swap two operators
        digitpositions = Vector([7, 2, 8, 3])
        daggers = BitVector([1, 1, 0, 1])
        newfocknbr, sign = QuantumDots.togglefermions(digitpositions, daggers, focknbr)
        @test newfocknbr == FockNumber(119) # = 1110 1110
        @test sign == -1

        # annihilate twice
        digitpositions = Vector([5, 3, 5])
        daggers = BitVector([0, 1, 0])
        _, sign = QuantumDots.togglefermions(digitpositions, daggers, focknbr)
        @test sign == 0
    end

    fs = QuantumDots.fockstates(10, 5)
    @test length(fs) == binomial(10, 5)
    @test allunique(fs)
    @test all(QuantumDots.fermionnumber.(fs) .== 5)
end


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
_bit(f::FockNumber, k) = Bool((f.f >> (k - 1)) & 1)


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

Compute the fermionic partial trace of a matrix `m` in basis `b`, leaving only the subsystems specified by `labels`. The result is stored in `mout`, and `sym` determines the ordering of the basis states.
"""
function partial_trace!(mout, m::AbstractMatrix{T}, labels, b::FermionBasis{M}, sym::AbstractSymmetry=NoSymmetry()) where {T,M}
    consistent_ordering(labels, b.jw) || throw(ArgumentError("Subsystem must be ordered in the same way as the full system"))
    N = length(labels)
    fill!(mout, zero(eltype(mout)))
    outinds = siteindices(labels, b.jw)
    bitmask = FockNumber(2^M - 1) - focknbr_from_site_indices(outinds)
    outbits(f) = map(i -> _bit(f, i), outinds)
    for _f1 in UnitRange{UInt64}(0, 2^M - 1), _f2 in UnitRange{UInt64}(0, 2^M - 1)
        f1 = FockNumber(_f1)
        f2 = FockNumber(_f2)
        if (f1 & bitmask) != (f2 & bitmask)
            continue
        end
        newfocknbr1 = focknbr_from_bits(outbits(f1))
        newfocknbr2 = focknbr_from_bits(outbits(f2))
        s1 = phase_factor_f(f1, f2, M)
        s2 = phase_factor_f(newfocknbr1, newfocknbr2, N)
        s = s2 * s1
        mout[focktoind(newfocknbr1, sym), focktoind(newfocknbr2, sym)] += s * m[focktoind(f1, b), focktoind(f2, b)]
    end
    return mout
end

"""
    partial_transpose(m::AbstractMatrix, labels, b::FermionBasis{M})

Compute the fermionic partial transpose of a matrix `m` in subsystem denoted by `labels`.
"""
function partial_transpose(m::AbstractMatrix, labels, b::FermionBasis)
    mout = zero(m)
    partial_transpose!(mout, m, labels, b)
end
function partial_transpose!(mout, m::AbstractMatrix, labels, b::FermionBasis{M}) where {M}
    @warn "partial_transpose may not be physically meaningful" maxlog = 10
    fill!(mout, zero(eltype(mout)))
    outinds = siteindices(labels, b)
    bitmask = FockNumber(2^M - 1) - focknbr_from_site_indices(outinds)
    outbits(f) = map(i -> _bit(f, i), outinds)
    fockstates = get_fockstates(b)
    for f1 in fockstates#UnitRange{UInt64}(0, 2^M - 1)
        f1R = (f1 & bitmask)
        f1L = (f1 & ~bitmask)
        for f2 in fockstates#UnitRange{UInt64}(0, 2^M - 1)
            f2R = (f2 & bitmask)
            f2L = (f2 & ~bitmask)
            newfocknbr1 = f2L + f1R
            newfocknbr2 = f1L + f2R
            s1 = phase_factor_f(f1, f2, M)
            s2 = phase_factor_f(newfocknbr1, newfocknbr2, M)
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

    @test pt((1, 2), M) ≈ pt((1, 2, 3, 4), pt((3, 4), M))
    @test all(pt(l, pt(l, M)) == M for l in Iterators.flatten((single_subsystems, pair_iterator, triple_iterator)))
end

function FockSplitter(b::FermionBasis, bs)
    fermionpositions = Tuple(map(Base.Fix2(siteindices, b.jw) ∘ Tuple ∘ collect ∘ keys, bs))
    Base.Fix2(split_focknumber, fermionpositions)
end
function split_focknumber(f::FockNumber, fermionpositions)
    map(positions -> focknbr_from_bits(map(i -> _bit(f, i), positions)), fermionpositions)
end
function split_focknumber(f::FockNumber, fockmapper::FockMapper)
    split_focknumber(f, fockmapper.fockpositions)
end
@testitem "Split focknumber" begin
    import QuantumDots: focknbr_from_site_indices as fock
    b1 = FermionBasis((1, 3))
    b2 = FermionBasis((2, 4))
    b = FermionBasis(1:4)
    focksplitter = QuantumDots.FockSplitter(b, (b1, b2))
    @test focksplitter(fock((1, 2, 3, 4))) == (fock((1, 2)), fock((1, 2)))
    @test focksplitter(fock((1,))) == (fock((1,)), fock(()))
    @test focksplitter(fock(())) == (fock(()), fock(()))
    @test focksplitter(fock((1, 2, 3))) == (fock((1, 2)), fock((1,)))
    @test focksplitter(fock((1, 3))) == (fock((1, 2)), fock(()))
    @test focksplitter(fock((2, 4))) == (fock(()), fock((1, 2)))
    @test focksplitter(fock((3, 2))) == (fock((2,)), fock((1,)))
    @test focksplitter(fock((3, 4))) == (fock((2,)), fock((2,)))
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