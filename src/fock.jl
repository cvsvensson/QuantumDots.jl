
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

struct FockMapper{P}
    fermionpositions::P
end
FockMapper(bs, b) = FockMapper_tuple(bs, b)
FockMapper_collect(bs, b) = FockMapper(map(Base.Fix2(siteindices, b.jw) ∘ collect ∘ keys, bs)) #faster construction
FockMapper_tuple(bs, b) = FockMapper(map(Base.Fix2(siteindices, b.jw) ∘ Tuple ∘ keys, bs)) #faster application

struct FockShifter{M}
    shifts::M
end
(fm::FockMapper)(f::NTuple{N,FockNumber}) where {N} = mapreduce(insert_bits, +, f, fm.fermionpositions)
(fs::FockShifter)(f::NTuple{N,FockNumber}) where {N} = mapreduce((f, M) -> shift_right(f, M), +, f, fs.shifts)
shift_right(f::FockNumber, M) = FockNumber(f.f << M)

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


##https://iopscience.iop.org/article/10.1088/1751-8121/ac0646/pdf (10c)
_bit(f::FockNumber, k) = Bool((f.f >> (k - 1)) & 1)


partial_trace(v::AbstractVector, args...) = partial_trace(v * v', args...)

"""
    partial_trace(m::AbstractMatrix,  bfull::AbstractBasis, bsub::AbstractBasis)

Compute the partial trace of a matrix `m`, leaving the subsystem defined by the basis `bsub`.
"""
function partial_trace(m::AbstractMatrix{T}, b::AbstractBasis, bout::AbstractBasis, phase_factors=use_partial_trace_phase_factors(b, bout)) where {T}
    N = length(get_fockstates(bout))
    mout = zeros(T, N, N)
    partial_trace!(mout, m, b, bout, phase_factors)
end

use_partial_trace_phase_factors(b1::FermionBasis, b2::FermionBasis) = true

"""
    partial_trace!(mout, m::AbstractMatrix, b::AbstractManyBodyBasis, bout::AbstractManyBodyBasis, phase_factors)

Compute the fermionic partial trace of a matrix `m` in basis `b`, leaving only the subsystems specified by `labels`. The result is stored in `mout`, and `bout` determines the ordering of the basis states.
"""
function partial_trace!(mout, m::AbstractMatrix, b::AbstractManyBodyBasis, bout::AbstractManyBodyBasis, phase_factors=use_partial_trace_phase_factors(b, bout))
    M = nbr_of_modes(b)
    sym = symmetry(bout)
    labels = collect(keys(bout))
    if phase_factors
        consistent_ordering(labels, b.jw) || throw(ArgumentError("Subsystem must be ordered in the same way as the full system"))
    end
    N = length(labels)
    fill!(mout, zero(eltype(mout)))
    outinds = siteindices(labels, b.jw)
    bitmask = FockNumber(2^M - 1) - focknbr_from_site_indices(outinds)
    outbits(f) = map(i -> _bit(f, i), outinds)
    fockstates = get_fockstates(b)
    for f1 in fockstates, f2 in fockstates
        if (f1 & bitmask) != (f2 & bitmask)
            continue
        end
        newfocknbr1 = focknbr_from_bits(outbits(f1))
        newfocknbr2 = focknbr_from_bits(outbits(f2))
        s1 = phase_factors ? phase_factor_f(f1, f2, M) : 1
        s2 = phase_factors ? phase_factor_f(newfocknbr1, newfocknbr2, N) : 1
        s = s2 * s1
        mout[focktoind(newfocknbr1, sym), focktoind(newfocknbr2, sym)] += s * m[focktoind(f1, b), focktoind(f2, b)]
    end
    return mout
end

"""
    partial_transpose(m::AbstractMatrix, b::AbstractManyBodyBasis, labels)

Compute the fermionic partial transpose of a matrix `m` in subsystem denoted by `labels`.
"""
function partial_transpose(m::AbstractMatrix, b::AbstractManyBodyBasis, labels, phase_factors=use_partial_transpose_phase_factors(b))
    mout = zero(m)
    partial_transpose!(mout, m, b, labels, phase_factors)
end
function partial_transpose!(mout, m::AbstractMatrix, b::AbstractManyBodyBasis, labels, phase_factors=use_partial_transpose_phase_factors(b))
    @warn "partial_transpose may not be physically meaningful" maxlog = 10
    M = nbr_of_modes(b)
    fill!(mout, zero(eltype(mout)))
    outinds = siteindices(labels, b)
    bitmask = FockNumber(2^M - 1) - focknbr_from_site_indices(outinds)
    outbits(f) = map(i -> _bit(f, i), outinds)
    fockstates = get_fockstates(b)
    for f1 in fockstates
        f1R = (f1 & bitmask)
        f1L = (f1 & ~bitmask)
        for f2 in fockstates
            f2R = (f2 & bitmask)
            f2L = (f2 & ~bitmask)
            newfocknbr1 = f2L + f1R
            newfocknbr2 = f1L + f2R
            s1 = phase_factors ? phase_factor_f(f1, f2, M) : 1
            s2 = phase_factors ? phase_factor_f(newfocknbr1, newfocknbr2, M) : 1
            s = s2 * s1
            mout[focktoind(newfocknbr1, b), focktoind(newfocknbr2, b)] = s * m[focktoind(f1, b), focktoind(f2, b)]
        end
    end
    return mout
end
use_partial_transpose_phase_factors(::FermionBasis) = true

@testitem "Partial transpose" begin
    using LinearAlgebra
    import QuantumDots: partial_transpose
    qn = ParityConservation()
    c1 = FermionBasis(1:1; qn)
    c2 = FermionBasis(2:2; qn)
    c12 = FermionBasis(1:2; qn)

    A = rand(ComplexF64, 2, 2)
    B = rand(ComplexF64, 2, 2)
    C = fermionic_kron((A, B), (c1, c2), c12)
    Cpt = partial_transpose(C, c12, (1,))
    Cpt2 = fermionic_kron((transpose(A), B), (c1, c2), c12)
    @test Cpt ≈ Cpt2

    ## Larger system
    labels = 1:4
    N = length(labels)
    cN = FermionBasis(labels; qn)
    cs = [FermionBasis(i:i; qn) for i in labels]
    Ms = [rand(ComplexF64, 2, 2) for _ in labels]
    M = fermionic_kron(Ms, cs, cN)

    single_subsystems = [(i,) for i in 1:4]
    for (k,) in single_subsystems
        Mpt = partial_transpose(M, cN, (k,))
        Mpt2 = fermionic_kron([(n == k) ? transpose(M) : M for (n, M) in enumerate(Ms)], cs, cN)
        @test Mpt ≈ Mpt2
    end
    pair_iterator = [(i, j) for i in 1:4, j in 1:4 if i != j]
    triple_iterator = [(i, j, k) for i in 1:4, j in 1:4, k in 1:4 if length(unique((i, j, k))) == 3]
    for (i, j) in pair_iterator
        Mpt = partial_transpose(M, cN, (i, j))
        Mpt2 = fermionic_kron([(n == i || n == j) ? transpose(M) : M for (n, M) in enumerate(Ms)], cs, cN)
        @test Mpt ≈ Mpt2
    end
    for (i, j, k) in triple_iterator
        Mpt = partial_transpose(M, cN, (i, j, k))
        Mpt2 = fermionic_kron([(n == i || n == j || n == k) ? transpose(M) : M for (n, M) in enumerate(Ms)], cs, cN)
        @test Mpt ≈ Mpt2
    end
    Mpt = partial_transpose(M, cN, labels)
    Mpt2 = fermionic_kron([transpose(M) for M in Ms], cs, cN)
    @test Mpt ≈ Mpt2
    @test !(Mpt ≈ transpose(M)) # partial transpose is not the same as transpose even if the full system is transposed
    @test M ≈ partial_transpose(M, cN, ())

    M = rand(ComplexF64, 2^N, 2^N)
    pt(l, M) = partial_transpose(M, cN, l)
    for (i, j) in pair_iterator
        @test pt((i, j), M) ≈ pt((j, i), M) ≈ pt(j, pt(i, M)) ≈ pt(i, pt(j, M))
    end
    for (i, j, k) in triple_iterator
        @test pt((i, j, k), M) ≈ pt((j, i, k), M) ≈ pt((j, k, i), M) ≈ pt((k, j, i), M) ≈ pt((k, i, j), M) ≈ pt((i, k, j), M) ≈ pt(i, pt(j, pt(k, M))) ≈ pt(j, pt(i, pt(k, M))) ≈ pt(k, pt(j, pt(i, M)))
    end

    @test pt((1, 2), M) ≈ pt((1, 2, 3, 4), pt((3, 4), M))
    @test all(pt(l, pt(l, M)) == M for l in Iterators.flatten((single_subsystems, pair_iterator, triple_iterator)))
end

function FockSplitter(b::AbstractManyBodyBasis, bs)
    fermionpositions = Tuple(map(Base.Fix2(siteindices, b.jw) ∘ Tuple ∘ collect ∘ keys, bs))
    Base.Fix2(split_focknumber, fermionpositions)
end
function split_focknumber(f::FockNumber, fermionpositions)
    map(positions -> focknbr_from_bits(map(i -> _bit(f, i), positions)), fermionpositions)
end
function split_focknumber(f::FockNumber, fockmapper::FockMapper)
    split_focknumber(f, fockmapper.fermionpositions)
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

    fockmapper = QuantumDots.FockMapper((b1, b2), b)
    @test QuantumDots.split_focknumber(fock((1, 2, 4)), fockmapper) == focksplitter(fock((1, 2, 4)))

    b1 = FermionBasis((1, 2))
    b2 = FermionBasis((3,))
    b = FermionBasis(1:3)
    focksplitter = QuantumDots.FockSplitter(b, (b1, b2))
    @test focksplitter(fock((1, 2, 3))) == (fock((1, 2)), fock((1,)))
    @test focksplitter(fock((1, 3))) == (fock((1,)), fock((1,)))
    @test focksplitter(fock((1, 2))) == (fock((1, 2)), fock(()))
    @test focksplitter(fock((2,))) == (fock((2,)), fock(()))
    @test focksplitter(fock((2, 3))) == (fock((2,)), fock((1,)))
    @test focksplitter(fock((3,))) == (fock(()), fock((1)))
end

use_reshape_phase_factors(b::FermionBasis, bs) = true
function Base.reshape(m::AbstractMatrix, b::AbstractManyBodyBasis, bs, phase_factors=use_reshape_phase_factors(b, bs))
    _reshape_mat_to_tensor(m, b, bs, FockSplitter(b, bs), phase_factors)
end
function Base.reshape(m::AbstractVector, b::AbstractManyBodyBasis, bs, phase_factors=use_reshape_phase_factors(b, bs))
    _reshape_vec_to_tensor(m, b, bs, FockSplitter(b, bs), phase_factors)
end

function Base.reshape(t::AbstractArray, bs, b::AbstractManyBodyBasis, phase_factors=use_reshape_phase_factors(b, bs))
    if ndims(t) == 2 * length(bs)
        return _reshape_tensor_to_mat(t, bs, b, FockMapper(bs, b), phase_factors)
    elseif ndims(t) == length(bs)
        return _reshape_tensor_to_vec(t, bs, b, FockMapper(bs, b), phase_factors)
    else
        throw(ArgumentError("The number of dimensions in the tensor must match the number of subsystems"))
    end
end

function _reshape_vec_to_tensor(v, b::AbstractManyBodyBasis, bs, fock_splitter, phase_factors)
    if phase_factors
        isorderedpartition(bs, b) || throw(ArgumentError("The partition must be ordered according to jw"))
    end
    dims = length.(get_fockstates.(bs))
    fs = get_fockstates(b)
    Is = map(f -> focktoind(f, b), fs)
    Iouts = map(f -> focktoind.(fock_splitter(f), bs), fs)
    t = Array{eltype(v),length(bs)}(undef, dims...)
    for (I, Iout) in zip(Is, Iouts)
        t[Iout...] = v[I...]
    end
    return t
end

function _reshape_mat_to_tensor(m::AbstractMatrix, b::AbstractManyBodyBasis, bs, fock_splitter, phase_factors)
    #reshape the matrix m in basis b into a tensor where each index pair has a basis in bs
    if phase_factors
        isorderedpartition(bs, b) || throw(ArgumentError("The partition must be ordered according to jw"))
    end
    dims = length.(get_fockstates.(bs))
    fs = get_fockstates(b)
    Is = map(f -> focktoind(f, b), fs)
    Iouts = map(f -> focktoind.(fock_splitter(f), bs), fs)
    t = Array{eltype(m),2 * length(bs)}(undef, dims..., dims...)
    partition = map(collect ∘ keys, bs)
    for (I1, Iout1, f1) in zip(Is, Iouts, fs)
        for (I2, Iout2, f2) in zip(Is, Iouts, fs)
            s = phase_factors ? phase_factor_h(f1, f2, partition, b.jw) : 1
            t[Iout1..., Iout2...] = m[I1, I2] * s
        end
    end
    return t
end

function _reshape_tensor_to_mat(t, bs, b::AbstractManyBodyBasis, fockmapper, phase_factors)
    if phase_factors
        isorderedpartition(bs, b) || throw(ArgumentError("The partition must be ordered according to jw"))
    end
    fs = Base.product(get_fockstates.(bs)...)
    fsb = map(fockmapper, fs)
    Is = map(f -> focktoind.(f, bs), fs)
    Iouts = map(f -> focktoind(f, b), fsb)
    m = Matrix{eltype(t)}(undef, length(fsb), length(fsb))
    partition = map(collect ∘ keys, bs)

    for (I1, Iout1, f1) in zip(Is, Iouts, fsb)
        for (I2, Iout2, f2) in zip(Is, Iouts, fsb)
            s = phase_factors ? phase_factor_h(f1, f2, partition, b.jw) : 1
            m[Iout1, Iout2] = t[I1..., I2...] * s
        end
    end
    return m
end

function _reshape_tensor_to_vec(t, bs, b::AbstractManyBodyBasis, fockmapper, phase_factors)
    isorderedpartition(bs, b) || throw(ArgumentError("The partition must be ordered according to jw"))
    fs = Base.product(get_fockstates.(bs)...)
    v = Vector{eltype(t)}(undef, length(fs))
    for fs in fs
        Is = focktoind.(fs, bs)
        fb = fockmapper(fs)
        Iout = focktoind(fb, b)
        v[Iout] = t[Is...]
    end
    return v
end

@testitem "Reshape" begin
    using LinearAlgebra
    function majorana_basis(b)
        majoranas = Dict((l, s) => (s == :- ? 1im : 1) * b[l] + hc for (l, s) in Base.product(keys(b), [:+, :-]))
        labels = collect(keys(majoranas))
        basisops = mapreduce(vec, vcat, [[prod(l -> majoranas[l], ls) for ls in Base.product([labels for _ in 1:n]...) if (issorted(ls) && allunique(ls))] for n in 1:length(labels)])
        pushfirst!(basisops, I + 0 * first(basisops))
        map(Hermitian ∘ (x -> x / sqrt(complex(tr(x * x)))), basisops)
    end

    qns = [NoSymmetry(), ParityConservation(), FermionConservation()]
    for qn in qns
        b = FermionBasis(1:2; qn)
        majbasis = majorana_basis(b)
        @test all(map(ishermitian, majbasis))
        overlaps = [tr(Γ1' * Γ2) for (Γ1, Γ2) in Base.product(majbasis, majbasis)]
        @test overlaps ≈ I
        @test rank(mapreduce(vec, hcat, majbasis)) == length(majbasis)
    end

    for (qn1, qn2, qn3) in Base.product(qns, qns, qns)
        b1 = FermionBasis((1, 3); qn=qn1)
        b2 = FermionBasis((2, 4); qn=qn2)
        d1 = 2^QuantumDots.nbr_of_modes(b1)
        d2 = 2^QuantumDots.nbr_of_modes(b2)
        bs = (b1, b2)
        b = FermionBasis(sort(vcat(keys(b1)..., keys(b2)...)); qn=qn3)
        m = b[1]
        t = reshape(m, b, bs)
        m12 = QuantumDots.reshape_to_matrix(t, (1, 3))
        @test rank(m12) == 1
        @test abs(dot(reshape(svd(m12).U, d1, d1, d2^2)[:, :, 1], b1[1])) ≈ norm(b1[1])

        m = b[1] + b[2]
        t = reshape(m, b, bs)
        m12 = QuantumDots.reshape_to_matrix(t, (1, 3))
        @test rank(m12) == 2

        m = rand(ComplexF64, d1 * d2, d1 * d2)
        t = reshape(m, b, bs)
        m2 = reshape(t, bs, b)
        @test m ≈ m2
        t = reshape(m, b, bs, false) #without phase factors (standard decomposition)
        m2 = reshape(t, bs, b, false)
        @test m ≈ m2

        v = rand(ComplexF64, d1 * d2)
        tv = reshape(v, b, bs)
        v2 = reshape(tv, bs, b)
        @test v ≈ v2
        # Note the how reshaping without phase factors is used in a contraction
        @test sum(reshape(m, b, bs, false)[:, :, i, j] * tv[i, j] for i in 1:d1, j in 1:d2) ≈ reshape(m * v, b, bs)

        m1 = rand(ComplexF64, d1 * d2, d1 * d2)
        m2 = rand(ComplexF64, d1 * d2, d1 * d2)
        t1 = reshape(m1, b, bs, false)
        t2 = reshape(m2, b, bs, false)
        t3 = zeros(ComplexF64, d1, d2, d1, d2)
        for i in 1:d1, j in 1:d2, k in 1:d1, l in 1:d2, k1 in 1:d1, k2 in 1:d2
            t3[i, j, k, l] += t1[i, j, k1, k2] * t2[k1, k2, k, l]
        end
        @test reshape(m1 * m2, b, bs, false) ≈ t3
        @test m1 * m2 ≈ reshape(t3, bs, b, false)

        basis1 = majorana_basis(b1)
        basis2 = majorana_basis(b2)
        basis12all = [fermionic_kron((Γ1, Γ2), bs, b) for (Γ1, Γ2) in Base.product(basis1, basis2)]
        basis12oddodd = [project_on_parities(Γ, b, bs, (-1, -1)) for Γ in basis12all]
        basis12oddeven = [project_on_parities(Γ, b, bs, (-1, 1)) for Γ in basis12all]
        basis12evenodd = [project_on_parities(Γ, b, bs, (1, -1)) for Γ in basis12all]
        basis12eveneven = [project_on_parities(Γ, b, bs, (1, 1)) for Γ in basis12all]
        basis12normalized = map(x -> x / sqrt(tr(x^2) + 0im), basis12all)

        @test all(map(tr, map(adjoint, basis12all) .* basis12all) .≈ 1)
        overlaps = [tr(Γ1' * Γ2) for (Γ1, Γ2) in Base.product(vec(basis12all), vec(basis12all))]
        @test overlaps ≈ I
        @test all(ishermitian, basis12normalized)
        @test all(map(tr, basis12normalized .* basis12normalized) .≈ 1)
        @test all(filter(x -> abs(x) > 0.01, map(tr, basis12oddodd .* basis12oddodd)) .≈ -1)
        @test all(filter(x -> abs(x) > 0.01, map(tr, basis12eveneven .* basis12eveneven)) .≈ 1)
        @test all(filter(x -> abs(x) > 0.01, map(tr, basis12evenodd .* basis12evenodd)) .≈ 1)
        @test all(filter(x -> abs(x) > 0.01, map(tr, basis12oddeven .* basis12oddeven)) .≈ 1)

        Hvirtual = rand(ComplexF64, length(basis1), length(basis2))
        Hoddoddvirtual = [Hvirtual[I] * norm(basis12oddodd[I]) for I in CartesianIndices(Hvirtual)]
        Hvirtual_no_oddodd = Hvirtual - Hoddoddvirtual
        H = sum(Hvirtual[I] * basis12all[I] for I in CartesianIndices(Hvirtual))
        H_no_oddodd = sum(Hvirtual_no_oddodd[I] * basis12all[I] for I in CartesianIndices(Hvirtual))
        Hotherbasis = sum(Hvirtual[I] * basis12normalized[I] for I in CartesianIndices(Hvirtual))
        H_no_oddodd_otherbasis = sum(Hvirtual_no_oddodd[I] * basis12normalized[I] for I in CartesianIndices(Hvirtual))
        @test H_no_oddodd_otherbasis ≈ H_no_oddodd

        t = reshape(H, b, bs)
        Hvirtual2 = QuantumDots.reshape_to_matrix(t, (1, 3))
        @test svdvals(Hvirtual) ≈ svdvals(Hvirtual2)
        Hvirtual3 = [tr(Γ' * H) / sqrt(tr(Γ' * Γ) + 0im) for Γ in basis12all]
        @test Hvirtual3 ≈ Hvirtual
        Hvirtual4 = [tr(Γ' * Hotherbasis) for Γ in basis12normalized]
        @test Hvirtual4 ≈ Hvirtual
        # @test svdvals(Hvirtual) ≈ svdvals(Hvirtual4)

        t_no_oddodd = reshape(H_no_oddodd, b, bs, true)
        Hvirtual_no_oddodd2 = QuantumDots.reshape_to_matrix(t_no_oddodd, (1, 3))
        @test svdvals(Hvirtual_no_oddodd) ≈ svdvals(Hvirtual_no_oddodd2)
        Hvirtual_no_oddodd3 = [tr(Γ' * H_no_oddodd) / sqrt(tr(Γ' * Γ) + 0im) for Γ in basis12all]
        @test svdvals(Hvirtual_no_oddodd) ≈ svdvals(Hvirtual_no_oddodd3)
        Hvirtual_no_oddodd4 = [tr(Γ' * H_no_oddodd_otherbasis) for Γ in basis12normalized]
        @test svdvals(Hvirtual_no_oddodd) ≈ svdvals(Hvirtual_no_oddodd4)



        ## Test consistency with partial trace
        m = rand(ComplexF64, d1 * d2, d1 * d2)
        m2 = partial_trace(m, b, b2, true)
        t = reshape(m, b, bs, true)
        tpt = sum(t[k, :, k, :] for k in axes(t, 1))
        @test m2 ≈ tpt

        m2 = partial_trace(m, b, b2, false)
        t = reshape(m, b, bs, false)
        tpt = sum(t[k, :, k, :] for k in axes(t, 1))
        @test m2 ≈ tpt

        mE = project_on_parity(m, b, 1)
        mO = project_on_parity(m, b, -1)
        m1 = rand(ComplexF64, d1, d1)
        m2 = rand(ComplexF64, d2, d2)
        m2O = project_on_parity(m2, b2, -1)
        m2E = project_on_parity(m2, b2, 1)
        m1O = project_on_parity(m1, b1, -1)
        m1E = project_on_parity(m1, b1, 1)
        mEE = project_on_parities(m, b, bs, (1, 1))
        mOO = project_on_parities(m, b, bs, (-1, -1))

        F = partial_trace(m * fermionic_kron((m1, I), bs, b), b, b2)
        @test tr(F * m2) ≈ tr(m * fermionic_kron((m1, I), bs, b) * fermionic_kron((I, m2), bs, b))

        t = reshape(m, b, bs, false)
        tpt = sum(t[k1, :, k2, :] * m1[k2, k1] for k1 in axes(t, 1), k2 in axes(t, 3))
        @test partial_trace(m * kron((m1, I), bs, b), b, b2, false) ≈ tpt

        ## More bases
        b3 = FermionBasis(5:5; qn3)
        d3 = 2^QuantumDots.nbr_of_modes(b3)
        bs = (b1, b2, b3)
        b = wedge(bs)
        m = rand(ComplexF64, d1 * d2 * d3, d1 * d2 * d3)
        t = reshape(m, b, (b1, b2, b3))
        @test ndims(t) == 6
        @test m ≈ reshape(t, bs, b)
    end
end

function reshape_to_matrix(t::AbstractArray{<:Any,N}, leftindices::NTuple{NL,Int}) where {N,NL}
    rightindices::NTuple{N - NL,Int} = Tuple(setdiff(ntuple(identity, N), leftindices))
    reshape_to_matrix(t, leftindices, rightindices)
end
function reshape_to_matrix(t::AbstractArray{<:Any,N}, leftindices::NTuple{NL,Int}, rightindices::NTuple{NR,Int}) where {N,NL,NR}
    @assert NL + NR == N
    tperm = permutedims(t, (leftindices..., rightindices...))
    lsize = prod(i -> size(t, i), leftindices, init=1)
    rsize = prod(i -> size(t, i), rightindices, init=1)
    reshape(tperm, lsize, rsize)
end

function project_on_parities(op::AbstractMatrix, b, bs, parities)
    length(bs) == length(parities) || throw(ArgumentError("The number of parities must match the number of subsystems"))
    for (bsub, parity) in zip(bs, parities)
        op = project_on_subparity(op, b, bsub, parity)
    end
    return op
end

function project_on_subparity(op::AbstractMatrix, b::FermionBasis, bsub::FermionBasis, parity)
    P = fermionic_embedding(parityoperator(bsub), bsub, b)
    return project_on_parity(op, P, parity)
end

project_on_parity(op::AbstractMatrix, b::FermionBasis, parity) = project_on_parity(op, parityoperator(b), parity)

function project_on_parity(op::AbstractMatrix, P::AbstractMatrix, parity)
    Peven = (I + P) / 2
    Podd = (I - P) / 2
    if parity == 1
        return Peven * op * Peven + Podd * op * Podd
    elseif parity == -1
        return Podd * op * Peven + Peven * op * Podd
    else
        throw(ArgumentError("Parity must be either 1 or -1"))
    end
end

@testitem "Parity projection" begin
    bs = [FermionBasis(2k-1:2k) for k in 1:3]
    b = wedge(bs)
    op = rand(ComplexF64, size(first(b)))
    local_parity_iter = (1, -1)
    all_parities = Base.product([local_parity_iter for _ in 1:length(bs)]...)
    @test sum(project_on_parities(op, b, bs, parities) for parities in all_parities) ≈ op

    ops = [rand(ComplexF64, size(first(b))) for b in bs]
    for parities in all_parities
        projected_ops = [project_on_parity(op, bsub, parity) for (op, bsub, parity) in zip(ops, bs, parities)]
        op = wedge(projected_ops, bs, b)
        @test op ≈ project_on_parities(op, b, bs, parities)
    end
end
