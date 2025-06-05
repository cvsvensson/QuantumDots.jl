
function Base.reshape(m::AbstractMatrix, H::AbstractFockHilbertSpace, Hs, phase_factors=use_reshape_phase_factors(H, Hs))
    _reshape_mat_to_tensor(m, H, Hs, FockSplitter(H, Hs), phase_factors)
end
function Base.reshape(m::AbstractVector, H::AbstractFockHilbertSpace, Hs, phase_factors=use_reshape_phase_factors(H, Hs))
    _reshape_vec_to_tensor(m, H, Hs, FockSplitter(H, Hs), phase_factors)
end

function Base.reshape(t::AbstractArray, Hs, H::AbstractFockHilbertSpace, phase_factors=use_reshape_phase_factors(H, Hs))
    if ndims(t) == 2 * length(Hs)
        return _reshape_tensor_to_mat(t, Hs, H, FockMapper(Hs, H), phase_factors)
    elseif ndims(t) == length(Hs)
        return _reshape_tensor_to_vec(t, Hs, H, FockMapper(Hs, H), phase_factors)
    else
        throw(ArgumentError("The number of dimensions in the tensor must match the number of subsystems"))
    end
end

function _reshape_vec_to_tensor(v, H::AbstractFockHilbertSpace, Hs, fock_splitter, phase_factors)
    if phase_factors
        isorderedpartition(Hs, H) || throw(ArgumentError("The partition must be ordered according to jw"))
    end
    dims = length.(focknumbers.(Hs))
    fs = focknumbers(H)
    Is = map(f -> focktoind(f, H), fs)
    Iouts = map(f -> focktoind.(fock_splitter(f), Hs), fs)
    t = Array{eltype(v),length(Hs)}(undef, dims...)
    for (I, Iout) in zip(Is, Iouts)
        t[Iout...] = v[I...]
    end
    return t
end

function _reshape_mat_to_tensor(m::AbstractMatrix, H::AbstractFockHilbertSpace, Hs, fock_splitter, phase_factors)
    #reshape the matrix m in basis b into a tensor where each index pair has a basis in bs
    if phase_factors
        isorderedpartition(Hs, H) || throw(ArgumentError("The partition must be ordered according to jw"))
    end
    dims = length.(focknumbers.(Hs))
    fs = focknumbers(H)
    Is = map(f -> focktoind(f, H), fs)
    Iouts = map(f -> focktoind.(fock_splitter(f), Hs), fs)
    t = Array{eltype(m),2 * length(Hs)}(undef, dims..., dims...)
    partition = map(collect ∘ keys, Hs)
    for (I1, Iout1, f1) in zip(Is, Iouts, fs)
        for (I2, Iout2, f2) in zip(Is, Iouts, fs)
            s = phase_factors ? phase_factor_h(f1, f2, partition, H.jw) : 1
            t[Iout1..., Iout2...] = m[I1, I2] * s
        end
    end
    return t
end

function _reshape_tensor_to_mat(t, Hs, H::AbstractFockHilbertSpace, fockmapper, phase_factors)
    if phase_factors
        isorderedpartition(Hs, H) || throw(ArgumentError("The partition must be ordered according to jw"))
    end
    fs = Base.product(focknumbers.(Hs)...)
    fsb = map(fockmapper, fs)
    Is = map(f -> focktoind.(f, Hs), fs)
    Iouts = map(f -> focktoind(f, H), fsb)
    m = Matrix{eltype(t)}(undef, length(fsb), length(fsb))
    partition = map(collect ∘ keys, Hs)

    for (I1, Iout1, f1) in zip(Is, Iouts, fsb)
        for (I2, Iout2, f2) in zip(Is, Iouts, fsb)
            s = phase_factors ? phase_factor_h(f1, f2, partition, H.jw) : 1
            m[Iout1, Iout2] = t[I1..., I2...] * s
        end
    end
    return m
end

function _reshape_tensor_to_vec(t, Hs, H::AbstractFockHilbertSpace, fockmapper, phase_factors)
    isorderedpartition(Hs, H) || throw(ArgumentError("The partition must be ordered according to jw"))
    fs = Base.product(focknumbers.(Hs)...)
    v = Vector{eltype(t)}(undef, length(fs))
    for fs in fs
        Is = focktoind.(fs, Hs)
        fb = fockmapper(fs)
        Iout = focktoind(fb, H)
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
