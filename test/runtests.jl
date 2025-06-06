#using QuantumDots
#using Test, LinearAlgebra, Random, BlockDiagonals

using TestItemRunner
@run_package_tests

@testitem "Parameters" begin
    using Random
    Random.seed!(1234)
    N = 4
    ph = parameter(1)
    ph2 = parameter(1; closed=true)
    @test ph isa QuantumDots.HomogeneousChainParameter
    @test ph2 isa QuantumDots.HomogeneousChainParameter
    @test ph == QuantumDots.parameter(1, :homogeneous)
    @test ph2 == QuantumDots.parameter(1, :homogeneous; closed=true)
    pih = parameter(rand(N), :inhomogeneous)
    @test pih isa QuantumDots.InHomogeneousChainParameter
    pd = parameter(1, :diff)
    @test pd isa QuantumDots.DiffChainParameter
    pr = parameter(rand(Int(ceil(N / 2))), :reflected)
    @test pr isa QuantumDots.ReflectedChainParameter
    @test_throws ErrorException parameter(1, :not_a_valid_option)
    @test Vector(ph, N; size=1) == fill(1, N)
    @test Vector(ph, N; size=2) == [fill(1, N - 1)..., 0]
    @test Vector(ph2, N) == fill(1, N)
    @test Vector(ph2, N; size=2) == fill(1, N)
    @test Vector(pih, N) == pih.values
    @test Vector(pr, N)[1:Int(ceil(N / 2))] == pr.values
    @test Vector(pr, N)[1:Int(ceil(N / 2))] == reverse(Vector(pr, N)[Int(ceil((N + 1) / 2)):end])
    @test Vector(pd, N) == 0:N-1

    for p in (ph, ph2, pih, pr, pd)
        @test [QuantumDots.getvalue(p, i, N) for i in 1:N] == Vector(p, N; size=1)
    end
end

@testitem "CAR" begin
    using LinearAlgebra
    for qn in [QuantumDots.NoSymmetry(), QuantumDots.parity, QuantumDots.fermionnumber]
        c = FermionBasis(1:2; qn)
        @test c[1] * c[1] == 0I
        @test c[1]' * c[1] + c[1] * c[1]' == I
        @test c[1]' * c[2] + c[2] * c[1]' == 0I
        @test c[1] * c[2] + c[2] * c[1] == 0I
    end
end

@testitem "Basis" begin
    using SparseArrays, LinearAlgebra, Random
    Random.seed!(1234)
    N = 2
    B = FermionBasis(1:N)
    @test QuantumDots.nbr_of_modes(B) == N
    Bspin = FermionBasis(1:N, (:↑, :↓); qn=QuantumDots.fermionnumber)
    @test QuantumDots.nbr_of_modes(Bspin) == 2N
    @test B[1] isa SparseMatrixCSC
    @test Bspin[1, :↑] isa SparseMatrixCSC
    @test parityoperator(B) isa SparseMatrixCSC
    @test parityoperator(Bspin) isa SparseMatrixCSC
    @test pretty_print(B[1], B) |> isnothing
    @test pretty_print(pi * B[1][:, 1], B) |> isnothing
    @test pretty_print(rand() * Bspin[1, :↑], Bspin) |> isnothing
    @test pretty_print(rand(ComplexF64) * Bspin[1, :↑][:, 1], Bspin) |> isnothing

    # fn = QuantumDots.fermionnumber((1,), B.jw)
    # @test fn.(0:3) == [0, 1, 0, 1]
    # fn = QuantumDots.fermionnumber((2,), B)
    # @test fn.(0:3) == [0, 0, 1, 1]
    # fn = QuantumDots.fermionnumber((1, 2), B)
    # @test fn.(0:3) == [0, 1, 1, 2]

    (c,) = QuantumDots.cell(1, B)
    @test c == B[1]
    (c1, c2) = QuantumDots.cell(1, Bspin)
    @test c1 == Bspin[1, :↑]
    @test c2 == Bspin[1, :↓]

    a = FermionBasis(1:3)
    as = (FermionBasis(1:1), FermionBasis(2:2), FermionBasis(3:3))
    @test all(f == a[n] for (n, f) in enumerate(a))
    v = [QuantumDots.indtofock(i, a) for i in 1:8]
    t1 = reshape(v, a, as)
    t2 = [i1 + 2i2 + 4i3 for i1 in (0, 1), i2 in (0, 1), i3 in (0, 1)]
    @test t1 == FockNumber.(t2)

    a = FermionBasis(1:3; qn=QuantumDots.parity)
    v = [QuantumDots.indtofock(i, a) for i in 1:8]
    t1 = reshape(v, a, as)
    t2 = [i1 + 2i2 + 4i3 for i1 in (0, 1), i2 in (0, 1), i3 in (0, 1)]
    @test t1 == FockNumber.(t2)


    c = FermionBasis(1:2, (:a, :b))
    cparity = FermionBasis(1:2, (:a, :b); qn=QuantumDots.parity)
    ρ = Matrix(Hermitian(rand(2^4, 2^4) .- 0.5))
    ρ = ρ / tr(ρ)
    function bilinears(c, labels)
        ops = reduce(vcat, [[c[l], c[l]'] for l in labels])
        return [op1 * op2 for (op1, op2) in Base.product(ops, ops)]
    end
    function bilinear_equality(c, csub, ρ)
        subsystem = Tuple(keys(csub))
        ρsub = partial_trace(ρ, c, csub)
        @test tr(ρsub) ≈ 1
        all((tr(op1 * ρ) ≈ tr(op2 * ρsub)) for (op1, op2) in zip(bilinears(c, subsystem), bilinears(csub, subsystem)))
    end
    function get_subsystems(c, N)
        t = collect(Base.product(ntuple(i -> keys(c), N)...))
        (t[I] for I in CartesianIndices(t) if issorted(Tuple(I)) && allunique(Tuple(I)))
    end
    for N in 1:4
        @test all(bilinear_equality(c, FermionBasis(subsystem), ρ) for subsystem in get_subsystems(c, N))
        @test all(bilinear_equality(c, FermionBasis(subsystem; qn=QuantumDots.parity), ρ) for subsystem in get_subsystems(c, N))
        @test all(bilinear_equality(c, FermionBasis(subsystem; qn=QuantumDots.parity), ρ) for subsystem in get_subsystems(cparity, N))
        @test all(bilinear_equality(c, FermionBasis(subsystem), ρ) for subsystem in get_subsystems(cparity, N))
    end

    ## Single particle density matrix
    N = 3
    c = FermionBasis(1:N)
    cbdg = FermionBdGBasis(1:N)
    rho = zero(first(c) * first(c))
    rho[1] = 1
    rho = rho / tr(rho)
    @test one_particle_density_matrix(rho, c) ≈ Diagonal([0, 0, 0, 1, 1, 1])
    @test one_particle_density_matrix(rho, c, [1]) ≈ Diagonal([0, 1])
    @test one_particle_density_matrix(rho, c, [2]) ≈ Diagonal([0, 1])
    @test one_particle_density_matrix(rho, c, [1, 2]) ≈ Diagonal([0, 0, 1, 1])

    get_ham(c) = (0.5c[1]' * c[1] + 0.3c[2]' * c[2] + 0.01c[3]' * c[3] + (c[1]' * c[2]' + 0.5 * c[2]' * c[3]' + hc))
    H = Matrix(get_ham(c))
    Hbdg = BdGMatrix(get_ham(cbdg))
    gs = first(eachcol(diagonalize(H).vectors))
    opdm_bdg = one_particle_density_matrix(diagonalize(Hbdg)[2])
    opdm = one_particle_density_matrix(gs * gs', c)
    @test opdm ≈ opdm_bdg

    @test (H - tr(H) * I / size(H, 1)) ≈ QuantumDots.many_body_hamiltonian(Hbdg, c)
    G1 = opdm_bdg[[1, 1 + N], [1, 1 + N]]
    G2 = (opdm_bdg[[1, 2, 1 + N, 2 + N], [1, 2, 1 + N, 2 + N]])
    G3 = (opdm_bdg[[1, 2, 3, 1 + N, 2 + N, 3 + N], [1, 2, 3, 1 + N, 2 + N, 3 + N]])
    G13 = (opdm_bdg[[1, 3, 1 + N, 3 + N], [1, 3, 1 + N, 3 + N]])
    c1 = FermionBasis(1:1)
    @test norm(partial_trace(gs, c, c1)) ≈ norm(one_particle_density_matrix(gs * gs', c, (1,))) ≈ norm(G1)
    @test one_particle_density_matrix(gs * gs', c, (1, 2)) ≈ G2
    @test one_particle_density_matrix(gs * gs', c, (1, 2, 3)) == one_particle_density_matrix(gs * gs', c) ≈ G3

    reduced_density_matrix = partial_trace(gs, c, FermionBasis((1,)))
    reduced_density_matrix2 = partial_trace(gs, c, FermionBasis((1, 2)))
    reduced_density_matrix3 = partial_trace(gs, c, FermionBasis((1, 2, 3)))
    reduced_density_matrix13 = partial_trace(gs, c, FermionBasis((1, 3)))
    c1 = FermionBasis(1:1)
    c12 = FermionBasis(1:2)
    @test reduced_density_matrix ≈ many_body_density_matrix(G1, c1)
    @test many_body_density_matrix(G1, c1) ≈ reverse(many_body_density_matrix(G1, FermionBasis(1:1; qn=QuantumDots.parity)))
    @test reduced_density_matrix2 ≈ many_body_density_matrix(G2, c12) ≈
          QuantumDots.many_body_density_matrix_exp(G2, c12)
    @test reduced_density_matrix13 ≈ many_body_density_matrix(G13, c12) ≈
          QuantumDots.many_body_density_matrix_exp(G13, c12)
    @test reduced_density_matrix13 ≈ many_body_density_matrix(G13, c12)
    @test reduced_density_matrix3 ≈ many_body_density_matrix(G3, c)

end


@testitem "Fermionic trace" begin
    using LinearAlgebra
    N = 4
    cs = [FermionBasis(n:n) for n in 1:N]
    c = FermionBasis(1:4)
    ops = [rand(ComplexF64, 2, 2) for _ in 1:N]
    op = fermionic_kron(ops, cs, c)
    @test tr(op) ≈ prod(tr, ops)

    op = fermionic_kron(ops, cs[[3, 2, 1, 4]], c)
    @test tr(op) ≈ prod(tr, ops)
end


@testitem "Fermionic partial trace" begin
    using LinearAlgebra
    qns = [NoSymmetry(), ParityConservation(), FermionConservation()]
    for qn in qns
        c = FermionBasis(1:3; qn)
        c1 = FermionBasis(1:1; qn)
        c2 = FermionBasis(2:2; qn)
        c12 = FermionBasis(1:2; qn)
        c13 = FermionBasis(1:3; qn)
        c23 = FermionBasis(2:3; qn)
        γ = Hermitian([0I rand(ComplexF64, 4, 4); rand(ComplexF64, 4, 4) 0I])
        f = c[1]
        @test tr(c1[1] * partial_trace(γ, c, c1,)) ≈ tr(f * γ)
        @test tr(c12[1] * partial_trace(γ, c, c12,)) ≈ tr(f * γ)
        @test tr(c13[1] * partial_trace(γ, c, c13,)) ≈ tr(f * γ)

        f = c[2]
        @test tr(c2[2] * partial_trace(γ, c, c2)) ≈ tr(f * γ)
        @test tr(c12[2] * partial_trace(γ, c, c12)) ≈ tr(f * γ)
        @test tr(c23[2] * partial_trace(γ, c, c23)) ≈ tr(f * γ)
    end
end


@testitem "BdG" begin
    using QuantumDots.SkewLinearAlgebra, LinearAlgebra, Random, SparseArrays
    Random.seed!(1234)

    N = 2
    labels = 1:N
    μ1 = rand()
    μ2 = rand()
    b = QuantumDots.FermionBdGBasis(labels)
    length(QuantumDots.FermionBdGBasis(1:2, (:a, :b))) == 4

    @test all(f == b[n] for (n, f) in enumerate(b))

    @test iszero(b[1] * b[1])
    @test iszero(b[1]' * b[1]')
    @test iszero(*(b[1], b[1]; symmetrize=false))
    @test iszero(*(b[1]', b[1]'; symmetrize=false))
    @test QuantumDots.cell(1, b)[1] == b[1]
    @test length(QuantumDots.cell(1, b)) == 1

    @test b[1] isa QuantumDots.BdGFermion
    @test b[1]' isa QuantumDots.BdGFermion
    @test b[1] * b[1] isa SparseMatrixCSC
    @test b[1].hole
    @test !b[1]'.hole
    @test b[1] + b[1] isa QuantumDots.QuasiParticle
    @test b[1] - b[1] isa QuantumDots.QuasiParticle
    @test norm(QuantumDots.rep(b[1] - b[1])) == 0
    @test QuantumDots.rep(b[1] + b[1]) == 2QuantumDots.rep(b[1])

    qp = b[1] + 1im * b[2]
    @test keys(qp.weights).values == [(1, :h), (2, :h)]
    @test keys(qp'.weights).values == [(1, :p), (2, :p)]
    @test qp.weights.values == [1, 1im]
    @test qp'.weights.values == [1, -1im]
    @test qp[1, :h] == 1
    @test qp'[1, :p] == 1
    @test qp[2, :p] == 0

    A = Matrix(μ1 * b[1]' * b[1] + μ2 * b[2]' * b[2])
    Abdg = QuantumDots.BdGMatrix(A)
    vals, vecs = QuantumDots.enforce_ph_symmetry(eigen(A))
    vals2, vecs2 = diagonalize(Abdg, QuantumDots.NormalEigenAlg())
    @test norm(vals - sort([-μ1, -μ2, μ1, μ2])) < 1e-14
    @test QuantumDots.ground_state_parity(vals, vecs) == 1
    vals_skew, vecs_skew = diagonalize(A, QuantumDots.SkewEigenAlg())
    vals_skew ≈ vals
    (vecs_skew' * vecs)^2 ≈ I
    @test QuantumDots.ground_state_parity(vals_skew, vecs_skew) == 1

    vals, vecs = QuantumDots.enforce_ph_symmetry(eigen(Matrix(μ1 * b[1]' * b[1] - μ2 * b[2]' * b[2])))
    @test QuantumDots.ground_state_parity(vals, vecs) == -1

    t = Δ = 1
    ham(b) = Matrix(QuantumDots.kitaev_hamiltonian(b; μ=0, t, Δ=exp(1im), V=0))
    poor_mans_ham = ham(b)
    vals, vecs = eigen(poor_mans_ham)
    es0, ops0 = QuantumDots.enforce_ph_symmetry(vals, vecs)
    es, ops = diagonalize(BdGMatrix(poor_mans_ham), QuantumDots.NormalEigenAlg())
    es2, ops2 = diagonalize(BdGMatrix(poor_mans_ham), QuantumDots.SkewEigenAlg())
    @test es0 ≈ es
    @test es0 ≈ es2
    @test I ≈ ops0' * ops0
    @test I ≈ ops' * ops
    @test I ≈ ops2' * ops2
    @test poor_mans_ham ≈ ops0 * Diagonal(es0) * ops0'
    @test poor_mans_ham ≈ ops * Diagonal(es) * ops'
    @test poor_mans_ham ≈ ops2 * Diagonal(es2) * ops2'

    ham(b) = Matrix(QuantumDots.kitaev_hamiltonian(b; μ=0, t, Δ, V=0))
    poor_mans_ham = ham(b)
    es, ops = diagonalize(BdGMatrix(poor_mans_ham), QuantumDots.NormalEigenAlg())

    @test QuantumDots.check_ph_symmetry(es, ops)
    @test norm(sort(es, by=abs)[1:2]) < 1e-12
    qps = map(op -> QuantumDots.QuasiParticle(op, b), eachcol(ops))
    @test all(map(qp -> iszero(qp * qp), qps))

    b_mb = QuantumDots.FermionBasis(labels)
    poor_mans_ham_mb = ham(b_mb)
    es_mb, states = eigen(poor_mans_ham_mb)
    P = QuantumDots.parityoperator(b_mb)

    parity(v) = v' * P * v
    gs_odd = parity(states[:, 1]) ≈ -1 ? states[:, 1] : states[:, 2]
    gs_even = parity(states[:, 1]) ≈ 1 ? states[:, 1] : states[:, 2]

    majcoeffs = QuantumDots.majorana_coefficients(gs_odd, gs_even, b_mb)
    majcoeffsbdg = QuantumDots.majorana_coefficients(qps[2])
    @test norm(map((m1, m2) -> abs2(m1) - abs2(m2), majcoeffs[1], majcoeffsbdg[1])) < 1e-12
    @test norm(map((m1, m2) -> abs2(m1) - abs2(m2), majcoeffs[2], majcoeffsbdg[2])) < 1e-12

    gs_parity = QuantumDots.ground_state_parity(es, ops)
    @test gs_parity ≈ parity(states[:, 1])
    ρeven, ρodd = if gs_parity == 1
        one_particle_density_matrix(qps[1:2]),
        one_particle_density_matrix(qps[[1, 3]])
    else
        one_particle_density_matrix(qps[[1, 3]]),
        one_particle_density_matrix(qps[1:2])
    end
    ρeven_mb = one_particle_density_matrix(gs_even * gs_even', b_mb)
    ρodd_mb = one_particle_density_matrix(gs_odd * gs_odd', b_mb)
    qps_mb = map(qp -> QuantumDots.many_body_fermion(qp, b_mb), qps)

    @test ρeven ≈ ρeven_mb
    @test ρodd ≈ ρodd_mb
    @test ρodd[[1, 3], [1, 3]] ≈ ρeven[[1, 3], [1, 3]]
    @test ρodd[[2, 4], [2, 4]] ≈ ρeven[[2, 4], [2, 4]]

    @test (gs_parity == 1 ? ρeven : ρodd) ≈ QuantumDots.one_particle_density_matrix(ops)

    @test poor_mans_ham ≈ mapreduce((e, qp) -> e * qp' * qp / 2, +, es, qps)

    qp = qps[2]
    @test qp isa QuantumDots.QuasiParticle
    @test 2 * qp isa QuantumDots.QuasiParticle
    @test qp * 2 isa QuantumDots.QuasiParticle
    @test qp / 2 isa QuantumDots.QuasiParticle
    @test qp + qp isa QuantumDots.QuasiParticle
    @test qp - qp isa QuantumDots.QuasiParticle
    @test qp + b[1] isa QuantumDots.QuasiParticle
    @test b[1] + qp isa QuantumDots.QuasiParticle
    @test qp - b[1] isa QuantumDots.QuasiParticle
    @test b[1] - qp isa QuantumDots.QuasiParticle
    @test abs(QuantumDots.majorana_polarization(qp)) ≈ 1

    @test qp[(1, :h)] == qp[1, :h]
    @test typeof(qp * b[1]) <: AbstractMatrix
    @test typeof(b[1] * qp) <: AbstractMatrix

    us, vs = (rand(length(labels)), rand(length(labels)))
    normalize!(us)
    normalize!(vs)
    vs = vs - dot(us, vs) * us
    vs = normalize!(vs) / sqrt(2)
    us = normalize!(us) / sqrt(2)
    χ = sum(us .* [b[i] for i in keys(b)]) + sum(vs .* [b[i]' for i in keys(b)])
    @test iszero(χ * χ)
    @test iszero(χ' * χ')
    χ_mb = sum(us .* [b_mb[i] for i in keys(b_mb)]) + sum(vs .* [b_mb[i]' for i in keys(b)])
    @test χ_mb ≈ QuantumDots.many_body_fermion(χ, b_mb)
    @test χ_mb' ≈ QuantumDots.many_body_fermion(χ', b_mb)

    @test all(b_mb[k] ≈ QuantumDots.many_body_fermion(b[k], b_mb) for k in 1:N)
    @test all(b_mb[k]' ≈ QuantumDots.many_body_fermion(b[k]', b_mb) for k in 1:N)


    # Longer kitaev 
    b = QuantumDots.FermionBdGBasis(1:5)
    b_mb = QuantumDots.FermionBasis(1:5; qn=QuantumDots.parity)
    ham2(b) = Matrix(QuantumDots.kitaev_hamiltonian(b; μ=0.1, t=1.1, Δ=1.0, V=0))
    pmmbdgham = ham2(b)
    pmmham = blockdiagonal(ham2(b_mb), b_mb)
    es, ops = diagonalize(BdGMatrix(pmmbdgham), QuantumDots.SkewEigenAlg(1e-10))
    es2, ops2 = diagonalize(BdGMatrix(pmmbdgham), QuantumDots.NormalEigenAlg(1e-10))
    @test QuantumDots.check_ph_symmetry(es, ops)
    qps = map(op -> QuantumDots.QuasiParticle(op, b), eachcol(ops))
    @test all(map(qp -> iszero(qp * qp), qps))

    eig = diagonalize(pmmham)
    fullsectors = QuantumDots.blocks(eig; full=true)
    oddvals = fullsectors[1].values
    evenvals = fullsectors[2].values
    oddvecs = fullsectors[1].vectors
    evenvecs = fullsectors[2].vectors

    majcoeffs = QuantumDots.majorana_coefficients(oddvecs[:, 1], evenvecs[:, 1], b_mb)
    majcoeffsbdg = QuantumDots.majorana_coefficients(qps[5])
    @test norm(map((m1, m2) -> abs2(m1) - abs2(m2), majcoeffs[1], majcoeffsbdg[1])) < 1e-12
    @test norm(map((m1, m2) -> abs2(m1) - abs2(m2), majcoeffs[2], majcoeffsbdg[2])) < 1e-12

    m = rand(10, 10)
    @test !QuantumDots.isantisymmetric(m)
    @test !QuantumDots.isbdgmatrix(m, m, m, m)
    H = Matrix(Hermitian(rand(ComplexF64, 2, 2)))
    Δ = rand(ComplexF64, 2, 2)
    Δ = Δ - transpose(Δ)

    @test QuantumDots.isantisymmetric(Δ)
    @test QuantumDots.isbdgmatrix(H, Δ, -conj(H), -conj(Δ))

    bdgm = BdGMatrix(H, Δ)
    @test size(bdgm) == (4, 4)
    @test Matrix(bdgm) == [bdgm[i, j] for i in axes(bdgm, 1), j in axes(bdgm, 2)] == collect(bdgm)
    @test Matrix(bdgm) ≈ hvcat(bdgm)

    @test QuantumDots.bdg_to_skew(bdgm) == QuantumDots.bdg_to_skew(Matrix(bdgm))
    @test QuantumDots.skew_to_bdg(QuantumDots.bdg_to_skew(bdgm)) ≈ bdgm

    @test 2 * bdgm ≈ bdgm + bdgm ≈ bdgm * 2
    @test iszero(bdgm - bdgm)
    if VERSION ≥ v"1.10-"
        hpbdgm = hermitianpart(bdgm)
        @test Matrix(hpbdgm) ≈ hermitianpart(Matrix(bdgm))
        @test hpbdgm ≈ hermitianpart!(bdgm)
    end

    m = sprand(10, 10, 0.1)
    @test !QuantumDots.isantisymmetric(m)
    @test !QuantumDots.isbdgmatrix(m, m, m, m)
    H = Matrix(Hermitian(sprand(ComplexF64, 10, 10, 0.1)))
    Δ = sprand(ComplexF64, 10, 10, 0.1)
    Δ = Δ - transpose(Δ)

    @test QuantumDots.isantisymmetric(Δ)
    @test QuantumDots.isbdgmatrix(H, Δ, -conj(H), -conj(Δ))

    bdgm = BdGMatrix(H, Δ)
    @test size(bdgm) == (20, 20)
    @test Matrix(bdgm) == [bdgm[i, j] for i in axes(bdgm, 1), j in axes(bdgm, 2)] == collect(bdgm) == bdgm[:, :]
    @test Matrix(bdgm) ≈ hvcat(bdgm)

    @test QuantumDots.bdg_to_skew(bdgm) == QuantumDots.bdg_to_skew(Matrix(bdgm))

    @test 2 * bdgm ≈ bdgm + bdgm ≈ bdgm * 2
    @test iszero(bdgm - bdgm)
    if VERSION ≥ v"1.10-"
        hpbdgm = hermitianpart(bdgm)
        @test Matrix(hpbdgm) ≈ hermitianpart(Matrix(bdgm))
        @test hpbdgm ≈ hermitianpart!(bdgm)
    end

end

@testitem "Kitaev" begin
    using Random, LinearAlgebra, BlockDiagonals
    Random.seed!(1234)

    N = 4
    c = FermionBasis(1:N)
    ham = Hermitian(QuantumDots.kitaev_hamiltonian(c; μ=0.0, t=1.0, Δ=1.0))
    vals, vecs = diagonalize(ham)
    @test abs(vals[1] - vals[2]) < 1e-12
    p = parityoperator(c)
    v1, v2 = eachcol(vecs[:, 1:2])
    @test dot(v1, p, v1) * dot(v2, p, v2) ≈ -1
    w = [dot(v1, f + f', v2) for f in c]
    z = [dot(v1, (f' - f), v2) for f in c]
    @test abs.(w .^ 2 - z .^ 2) ≈ [1, 0, 0, 1]
    w, z = QuantumDots.majorana_coefficients(v1, v2, c)
    mps = QuantumDots.majorana_polarization(w, z, 1:2)
    @test mps.mp ≈ 1 && mps.mpu ≈ 1

    ϕ = rand() * 2pi
    wϕ, zϕ = QuantumDots.majorana_coefficients(v1, exp(1im * ϕ) * v2, c)
    mpsϕ = QuantumDots.majorana_polarization(wϕ, zϕ, 1:2)
    @test mpsϕ.mp ≈ 1 && mpsϕ.mpu ≈ 1
    wϕ2, zϕ2 = QuantumDots.rotate_majorana_coefficients(wϕ, zϕ, -mpsϕ.phase)

    function test_angle(w)
        a = mod(angle(w[findmax(abs, w)[2]]), pi / 4)
        a < 1e-12 || a > pi / 4 - 1e-12
    end
    @test test_angle(wϕ2) && test_angle(zϕ2)
    @test !(test_angle(wϕ) && test_angle(zϕ))

    eig = diagonalize(ham)
    eigsectors = blocks(eig)
    @test v1 ≈ eigsectors[1].vectors[:, 1]
    gs = QuantumDots.ground_state.(eigsectors)
    @test gs[1].vector ≈ v1
    @test gs[1].value ≈ eigsectors[1].values[1]

    N = 5
    c = FermionBasis(1:N; qn=QuantumDots.parity)
    ham = QuantumDots.blockdiagonal(QuantumDots.kitaev_hamiltonian(c; μ=0.0, t=1.0, Δ=1.0), c)
    vals, vecs = diagonalize(ham)
    @test abs(vals[1] - vals[1+size(vecs.blocks[1], 1)]) < 1e-12
    p = parityoperator(c)
    v1 = vecs[:, 1]
    v2 = vecs[:, 1+size(vecs.blocks[1], 1)]
    @test dot(v1, p, v1) * dot(v2, p, v2) ≈ -1
    w = [dot(v1, f + f', v2) for f in c]
    z = [dot(v1, (f' - f), v2) for f in c]
    @test abs.(w .^ 2 - z .^ 2) ≈ [1, 0, 0, 0, 1]
    w, z = QuantumDots.majorana_coefficients(v1, v2, c)
    mps = QuantumDots.majorana_polarization(w, z, 1:2)
    @test mps.mp ≈ 1 && mps.mpu ≈ 1

    eig = diagonalize(ham)
    eigsectors = blocks(eig; full=true)
    @test v1 ≈ eigsectors[1].vectors[:, 1]
    @test v2 ≈ eigsectors[2].vectors[:, 1]

    eigsectors = blocks(eig; full=false)
    @test v1 ≈ vcat(eigsectors[1].vectors[:, 1], zero(eigsectors[1].vectors[:, 1]))
    @test v2 ≈ vcat(zero(eigsectors[1].vectors[:, 1]), eigsectors[2].vectors[:, 1])
end

@testitem "Parity and number operator" begin
    using SparseArrays, LinearAlgebra
    function get_ops(qn)
        N = 2
        a = FermionBasis(1:N; qn)
        ham = a[1]' * a[1] + π * a[2]' * a[2]
        vals, vecs = eigen(Matrix(ham))
        @test vals ≈ [0, 1, π, π + 1]
        parityop = parityoperator(a)
        numberop = numberoperator(a)
        @test all([v' * parityop * v for v in eachcol(vecs)] .≈ [1, -1, -1, 1])
        @test Diagonal(diag(parityop)) == parityop
        @test Diagonal(diag(numberop)) == numberop
        @test sum(a[i]'a[i] for i in 1:N) == numberop
        @test prod(2(a[i]'a[i] - 1 / 2 * sparse(I, 2^N, 2^N)) for i in 1:N) == parityop
        return parityop, numberop
    end
    parityop, numberop = get_ops(QuantumDots.NoSymmetry())
    @test diag(parityop) == [1, -1, -1, 1]
    @test diag(numberop) == [0, 1, 1, 2]

    parityop, numberop = get_ops(QuantumDots.parity)
    @test diag(parityop) == [-1, -1, 1, 1]
    @test diag(numberop) == [1, 1, 0, 2]

    parityop, numberop = get_ops(QuantumDots.fermionnumber)
    @test diag(parityop) == [1, -1, -1, 1]
    @test diag(numberop) == [0, 1, 1, 2]

end


@testitem "BlockDiagonal" begin
    using SparseArrays, LinearAlgebra, BlockDiagonals
    N = 2
    a = FermionBasis(1:N; qn=QuantumDots.parity)
    ham0 = a[1]' * a[1] + π * a[2]' * a[2]
    ham = blockdiagonal(ham0, a)
    @test ham isa BlockDiagonal{Float64,SparseMatrixCSC{Float64,Int}}
    ham = blockdiagonal(Matrix, ham0, a)
    @test ham isa BlockDiagonal{Float64,Matrix{Float64}}
    vals, vecs = eigen(ham)
    @test vals ≈ [0, 1, π, π + 1]
    parityop = blockdiagonal(parityoperator(a), a)
    numberop = blockdiagonal(numberoperator(a), a)
end


@testitem "build_function" begin
    using Symbolics, BlockDiagonals
    N = 2
    bases = [FermionBasis(1:N), FermionBasis(1:N; qn=QuantumDots.parity), FermionBdGBasis(1:N)]
    @variables x
    ham(c) = x * sum(f -> 1.0 * f'f, c)
    converts = [Matrix, x -> blockdiagonal(x, bases[2]), x -> BdGMatrix(x; check=false)]
    hams = map((f, c) -> (f ∘ ham)(c), converts, bases)
    fs = [build_function(H, x; expression=Val{false}) for H in hams]
    newhams = map(f -> f[1](1.0), fs)
    @test newhams[1] isa Matrix
    @test newhams[2] isa BlockDiagonal
    @test newhams[3] isa BdGMatrix
    @test all(isnothing(pretty_print(ham(b), b)) for b in bases[[1, 2]])
    cache = 0.1 .* newhams
    newhams = map(f -> f[1](0.3), fs)
    map((m, f) -> f[2](m, 0.3), cache, fs)
    @test all(newhams .≈ cache)
end

@testitem "Fast generated hamiltonians" begin
    using Random, LinearAlgebra, Symbolics
    Random.seed!(1234)

    N = 5
    params = rand(3)
    _hamiltonian(a, μ, t, Δ) = μ * sum(a[i]'a[i] for i in 1:QuantumDots.nbr_of_modes(a)) + t * (a[1]'a[2] + a[2]'a[1]) + Δ * (a[1]'a[2]' + a[2]a[1])

    a = FermionBasis(1:N)
    hamiltonian(params...) = _hamiltonian(a, params...)
    fastham! = QuantumDots.fastgenerator(hamiltonian, 3)
    mat = hamiltonian((2 .* params)...)
    fastham!(mat, params...)
    @test mat ≈ hamiltonian(params...)

    #parity conservation
    a = FermionBasis(1:N; qn=QuantumDots.parity)
    hamiltonian(params...) = _hamiltonian(a, params...)
    parityham! = QuantumDots.fastgenerator(hamiltonian, 3)
    mat = hamiltonian((2 .* params)...)
    parityham!(mat, params...)
    @test mat ≈ hamiltonian(params...)

    _bd(m) = blockdiagonal(m, a).blocks
    bdham = _bd ∘ hamiltonian

    oddham! = QuantumDots.fastgenerator(first ∘ bdham, 3)
    oddmat = bdham((2 .* params)...) |> first
    oddham!(oddmat, params...)
    @test oddmat ≈ bdham(params...) |> first

    evenham! = QuantumDots.fastgenerator(last ∘ bdham, 3)
    evenmat = bdham((2 .* params)...) |> last
    evenham!(evenmat, params...)
    @test evenmat ≈ bdham(params...) |> last

    _bd2(xs...) = blockdiagonal(hamiltonian(xs...), a)
    paritybd! = QuantumDots.fastblockdiagonal(_bd2, 3)
    bdham = _bd2(2params...)
    paritybd!(bdham, params...)
    @test bdham ≈ _bd2(params...)
    @test bdham.blocks |> first ≈ oddmat
    @test bdham.blocks |> last ≈ evenmat

    #number conservation
    a = FermionBasis(1:N; qn=QuantumDots.fermionnumber)
    hamiltonian(params...) = _hamiltonian(a, params...)

    numberham! = QuantumDots.fastgenerator(hamiltonian, 3)
    mat = hamiltonian((2 .* params)...)
    numberham!(mat, params...)
    @test mat ≈ hamiltonian(params...)

    numberbdham(params...) = blockdiagonal(hamiltonian(params...), a)
    numberbd! = QuantumDots.fastblockdiagonal(numberbdham, 3)
    bdham = numberbdham(2params...)
    numberbd!(bdham, params...)
    @test bdham ≈ numberbdham(params...)
    @test bdham ≈ hamiltonian(params[1:end-1]..., 0.0)

    b = FermionBasis(1:2, (:a, :b); qn=QuantumDots.parity)
    nparams = 8
    params = rand(nparams)
    ham = (t, Δ, V, θ, h, U, Δ1, μ) -> Matrix(QuantumDots.BD1_hamiltonian(b; μ, t, Δ, V, θ, h, U, Δ1, ϕ=0))
    hammat = ham(params...)
    fastgen! = QuantumDots.fastgenerator(ham, nparams)
    hammat2 = ham(rand(Float64, nparams)...)
    fastgen!(hammat2, params...)
    @test hammat2 ≈ hammat

    hambd(p...) = QuantumDots.blockdiagonal(ham(p...), b)
    @test sort!(abs.(eigvals(hambd(params...)))) ≈ sort!(abs.(eigvals(hammat)))

    fastgen! = QuantumDots.fastblockdiagonal(hambd, nparams)
    bdhammat2 = hambd(rand(nparams)...)
    fastgen!(bdhammat2, params...)
    @test hambd(params...) ≈ bdhammat2

end

@testitem "rotations" begin
    using Random
    Random.seed!(1234)

    N = 2
    b = FermionBasis(1:N, (:↑, :↓))
    standard_hopping = QuantumDots.hopping(1, b[1, :↑], b[2, :↑]) + QuantumDots.hopping(1, b[1, :↓], b[2, :↓])
    standard_pairing = QuantumDots.pairing(1, b[1, :↑], b[2, :↓]) - QuantumDots.pairing(1, b[1, :↓], b[2, :↑])
    local_pairing = sum(QuantumDots.pairing(1, QuantumDots.cell(j, b)...) for j in 1:N)
    θ = rand()
    ϕ = rand()
    @test QuantumDots.hopping_rotated(1, QuantumDots.cell(1, b), QuantumDots.cell(2, b), (0, 0), (0, 0)) ≈ standard_hopping
    @test QuantumDots.hopping_rotated(1, QuantumDots.cell(1, b), QuantumDots.cell(2, b), (θ, ϕ), (θ, ϕ)) ≈ standard_hopping
    @test QuantumDots.pairing_rotated(1, QuantumDots.cell(1, b), QuantumDots.cell(2, b), (0, 0), (0, 0)) ≈ standard_pairing
    @test QuantumDots.pairing_rotated(1, QuantumDots.cell(1, b), QuantumDots.cell(2, b), (θ, ϕ), (θ, ϕ)) ≈ standard_pairing

    soc = QuantumDots.hopping(exp(1im * ϕ), b[1, :↓], b[2, :↑]) - QuantumDots.hopping(exp(-1im * ϕ), b[1, :↑], b[2, :↓])
    @test QuantumDots.hopping_rotated(1, QuantumDots.cell(1, b), QuantumDots.cell(2, b), (0, 0), (θ, ϕ)) ≈ standard_hopping * cos(θ / 2) + sin(θ / 2) * soc

    Δk = QuantumDots.pairing(exp(1im * ϕ), b[1, :↑], b[2, :↑]) + QuantumDots.pairing(exp(-1im * ϕ), b[1, :↓], b[2, :↓])
    @test QuantumDots.pairing_rotated(1, QuantumDots.cell(1, b), QuantumDots.cell(2, b), (0, 0), (θ, ϕ)) ≈ standard_pairing * cos(θ / 2) + sin(θ / 2) * Δk

    θp = parameter(θ, :homogeneous)
    ϕp = parameter(ϕ, :homogeneous)
    @test standard_hopping ≈ QuantumDots.BD1_hamiltonian(b; t=1, μ=0, V=0, U=0, h=0, θ=θp, ϕ=ϕp, Δ=0, Δ1=0)
    @test standard_pairing ≈ QuantumDots.BD1_hamiltonian(b; t=0, μ=0, V=0, U=0, h=0, θ=θp, ϕ=ϕp, Δ=0, Δ1=1)
    @test QuantumDots.BD1_hamiltonian(b; t=0, μ=0, V=0, U=0, h=0, θ=θp, ϕ=ϕp, Δ=1, Δ1=0) ≈ local_pairing

    @test QuantumDots.BD1_hamiltonian(b; t=0, μ=1, V=0, U=0, h=0, θ=θp, ϕ=ϕp, Δ=0, Δ1=0) ≈ -QuantumDots.numberoperator(b)

    @test QuantumDots.BD1_hamiltonian(b; t=0, μ=1, V=0, U=0, h=0, θ=θp, ϕ=ϕp, Δ=0, Δ1=0) == QuantumDots.BD1_hamiltonian(b; t=0, μ=1, V=0, U=0, h=0, θ=θ .* [0, 1], ϕ=ϕ .* [0, 1], Δ=0, Δ1=0)
end


@testitem "transport" begin
    using OrdinaryDiffEqTsit5, LinearSolve, Random, LinearAlgebra
    import DifferentiationInterface as AD
    using ForwardDiff, FiniteDifferences
    Random.seed!(1234)

    function test_qd_transport(qn)
        # using QuantumDots, Test, Pkg
        # Pkg.activate("./test")
        # using LinearSolve, LinearAlgebra
        # import DifferentiationInterface as AD, ForwardDiff, FiniteDifferences
        # qn = QuantumDots.NoSymmetry()
        # qn = QuantumDots.parity
        N = 1
        a = FermionBasis(1:N; qn)
        bd(m) = QuantumDots.blockdiagonal(m, a)
        get_hamiltonian(μ) = bd(μ * sum(a[i]'a[i] for i in 1:N))
        T = rand()
        μL, μR, μH = rand(3)
        leftlead = CombinedLead((a[1]',); T, μ=μL)
        rightlead = NormalLead(a[N]'; T, μ=μR)
        leads = Dict(:left => leftlead, :right => rightlead)

        particle_number = bd(numberoperator(a))
        ham = get_hamiltonian(μH)
        diagham = diagonalize(ham)
        diagham2 = QuantumDots.remove_high_energy_states(diagham, μH / 2)
        @test diagham.original ≈ ham
        @test diagham.values ≈ (qn == QuantumDots.parity ? [μH, 0] : [0, μH])
        @test diagham2.values ≈ [0]
        ls = LindbladSystem(ham, leads)
        ls_cache = LindbladSystem(ham, leads; usecache=true)
        mo = QuantumDots.LinearOperator(ls)
        @test mo isa MatrixOperator
        @test collect(mo) ≈ collect(QuantumDots.LinearOperator(ls_cache))

        vr = rand(ComplexF64, size(ls.total, 1))
        vl = rand(ComplexF64, size(ls.total, 1))
        @test all(dot(vl, d * vr) ≈ dot(d' * vl, vr) for d in values(ls.dissipators))

        lazyls = LazyLindbladSystem(ham, leads)
        @test eltype(lazyls) == ComplexF64
        @test eltype(lazyls.dissipators[:left]) == ComplexF64
        mr = rand(ComplexF64, size(ham))
        ml = rand(ComplexF64, size(ham))
        @test all(dot(ml, d * mr) ≈ dot(d' * ml, mr) for d in values(lazyls.dissipators))

        prob = StationaryStateProblem(ls)
        prob2 = StationaryStateProblem(lazyls)
        ρinternal = solve(prob; abstol=1e-12)
        ρinternal2 = solve(prob2, LinearSolve.KrylovJL_LSMR(); abstol=1e-12)
        @test tomatrix(ρinternal, ls) ≈ reshape(ρinternal2, size(tomatrix(ρinternal, ls))...)
        ρ = tomatrix(ρinternal, ls)
        linsolve = init(prob)
        @test solve!(linsolve) ≈ ρinternal
        rhod = diag(ρ)
        @test ρ ≈ ρ'
        @test tr(ρ) ≈ 1
        p2 = (QuantumDots.fermidirac(μH, T, μL) + QuantumDots.fermidirac(μH, T, μR)) / 2
        p1 = 1 - p2
        analytic_current = -1 / 2 * (QuantumDots.fermidirac(μH, T, μL) - QuantumDots.fermidirac(μH, T, μR))
        @test rhod ≈ (qn == QuantumDots.parity ? [p2, p1] : [p1, p2])

        numeric_current = QuantumDots.measure(ρ, particle_number, ls)
        cm = conductance_matrix(AD.AutoForwardDiff(), ls, ρinternal, particle_number)
        cm2 = conductance_matrix(AD.AutoFiniteDifferences(FiniteDifferences.central_fdm(3, 1)), ls, particle_number)
        cm3 = conductance_matrix(1e-5, ls, particle_number)
        @test norm(cm - cm2) < 1e-3
        @test norm(cm - cm3) < 1e-3

        @test numeric_current ≈ QuantumDots.measure(ρinternal, particle_number, ls)
        @test abs(sum(values(numeric_current))) < 1e-10
        @test numeric_current ≈ [-analytic_current, analytic_current]

        pauli = PauliSystem(ham, leads)
        pauli_prob = StationaryStateProblem(pauli)
        ρ_pauli_internal = solve(pauli_prob)
        ρ_pauli = tomatrix(ρ_pauli_internal, pauli)
        linsolve_pauli = init(pauli_prob)
        @test solve!(linsolve_pauli) ≈ ρ_pauli_internal

        @test diag(ρ_pauli) ≈ rhod
        @test tr(ρ_pauli) ≈ 1
        rate_current = QuantumDots.get_currents(ρ_pauli, pauli)
        # @test rate_current[:left] ≈ QuantumDots.get_currents(ρ_pauli_internal, pauli)[:left]
        @test rate_current ≈ QuantumDots.get_currents(ρ_pauli_internal, pauli)
        @test numeric_current(:left) / numeric_current(:right) ≈ rate_current(:left) / rate_current(:right)

        cmpauli = conductance_matrix(AD.AutoForwardDiff(), pauli, ρ_pauli_internal)
        cmpauli2 = conductance_matrix(AD.AutoFiniteDifferences(FiniteDifferences.central_fdm(3, 1)), pauli)
        cmpauli3 = conductance_matrix(1e-5, pauli)

        @test norm(cmpauli - cmpauli2) < 1e-3
        @test norm(cmpauli - cmpauli3) < 1e-3
        eigen_particle_number = QuantumDots.changebasis(particle_number, diagham)
        @test vec(sum(eigen_particle_number * pauli.dissipators[:left].total_master_matrix, dims=1)) ≈ pauli.dissipators[:left].Iin + pauli.dissipators[:left].Iout

        prob = ODEProblem(ls, I / 2^N, (0, 100))
        sol = solve(prob, Tsit5())
        @test all(diff([tr(tomatrix(sol(t), ls)^2) for t in 0:0.1:1]) .> 0)
        @test norm(ρinternal - sol(100)) < 1e-3

        prob = ODEProblem((du, u, p, t) -> ls_cache(du, u, p), QuantumDots.internal_rep(I / 2^N, ls), (0, 100))
        sol_cache = solve(prob, Tsit5())
        @test norm(sol(100) - sol_cache(100)) < 1e-8

        prob = ODEProblem(pauli, I / 2^N, (0, 100))
        sol = solve(prob, Tsit5())
        @test norm(ρ_pauli_internal - sol(100)) < 1e-3

        @test QuantumDots.internal_rep(ρ, ls) ≈
              QuantumDots.internal_rep(ρinternal, ls) ≈
              QuantumDots.internal_rep(Matrix(ρ), ls)
        @test QuantumDots.internal_rep(ρ_pauli, pauli) ≈
              QuantumDots.internal_rep(ρ_pauli_internal, pauli) ≈
              QuantumDots.internal_rep(Matrix(ρ_pauli), pauli)

        @test islinear(pauli)
        @test all(map(islinear, values(pauli.dissipators)))
        @test islinear(ls)
        @test all(map(islinear, values(ls.dissipators)))
        @test eltype(ls) == eltype(ls.total)
        @test eltype(pauli) == eltype(pauli.total_master_matrix)
        @test Matrix(ls) == ls.total
        @test Matrix(pauli) == pauli.total_master_matrix
        @test eltype(pauli.dissipators[:left]) == eltype(Matrix(pauli.dissipators[:left]))
        @test size(pauli) == size(Matrix(pauli)) == size(pauli.dissipators[:left])
        @test size(ls) == size(Matrix(ls)) == size(ls.dissipators[:left])

        A = QuantumDots.LinearOperator(ls)
        vr = rand(ComplexF64, size(A, 2))
        vl = rand(ComplexF64, size(A, 2))
        solr = solve(ODEProblem(A, vr, (0, 10)), Tsit5(); abstol=1e-5)
        soll = solve(ODEProblem(A', vl, (0, 10)), Tsit5(); abstol=1e-5)
        @test isapprox(dot(soll(10), vr), dot(vl, solr(10)); atol=1e-4)
    end
    test_qd_transport(QuantumDots.NoSymmetry())
    test_qd_transport(QuantumDots.parity)
    test_qd_transport(QuantumDots.fermionnumber)


    N = 2
    qn = QuantumDots.NoSymmetry()
    a = FermionBasis(1:N; qn)
    bd(m) = blockdiagonal(m, a)
    hamiltonian = bd(sum(a[n]'a[n] for n in 1:N) + 0.2 * (sum(a[n]a[n+1] + hc for n in 1:N-1)))
    T = 0.1
    μL = 0.5
    μR = 0.0
    leftlead = CombinedLead((a[1]',); T, μ=μL)
    rightlead = NormalLead(a[N]'; T, μ=μR)
    leads = Dict(:left => leftlead, :right => rightlead)

    particle_number = bd(numberoperator(a))
    ls = LindbladSystem(hamiltonian, leads)
    mo = QuantumDots.LinearOperator(ls)
    mo2 = QuantumDots.LinearOperator(ls; normalizer=true)

    lazyls = LazyLindbladSystem(hamiltonian, leads)
    fo = QuantumDots.LinearOperator(lazyls)
    fo2 = QuantumDots.LinearOperator(lazyls; normalizer=true)
    v1 = rand(2^(2N))
    v2 = rand(2^(2N))
    v2n = rand(2^(2N) + 1)
    vc1 = deepcopy(complex(v1))
    vc2 = deepcopy(complex(v1))
    @test fo * v1 ≈ mo * v1
    @test mul!(vc2, fo, v1) ≈ mul!(vc1, mo, v1)
    @test dot(fo' * v2, v1) ≈ dot(v2, fo * v1)
    @test fo2 * v1 ≈ mo2 * v1
    @test dot(fo2' * v2n, v1) ≈ dot(v2n, fo2 * v1)

    m = rand(ComplexF64, 2^N, 2^N)
    mout = deepcopy(m)
    @test lazyls * m ≈ reshape(mo * vec(m), size(m)...)
    @test mul!(mout, lazyls, m) ≈ reshape(mo * vec(m), size(m)...)
    lazyd = first(values(lazyls.dissipators))
    d = first(values(ls.dissipators))
    @test lazyd * m ≈ reshape(d * vec(m), size(m)...)
    mul!(mout, lazyd, m)
    @test mout ≈ lazyd * m
    @test mout ≈ reshape(mul!(vc1, d, vec(m)), size(m)...)

    prob1 = StationaryStateProblem(ls)
    prob2 = StationaryStateProblem(lazyls)
    ρinternal1 = solve(prob1, LinearSolve.KrylovJL_LSMR(); abstol=1e-12)
    ρinternal2 = solve(prob2, LinearSolve.KrylovJL_LSMR(); abstol=1e-12)
    @test ρinternal1 ≈ ρinternal2

    prob = ODEProblem(lazyls, Matrix{ComplexF64}(I, 2^N, 2^N) / 2^N, (0, 2))
    sol = solve(prob, Tsit5())
    @test tr(sol(1)) ≈ 1

    u = sol(1)
    diss = lazyls.dissipators[:left]
    out = diss * u
    @test abs(tr(out)) < 1e-10
    @test diss(u) ≈ out
    @test diss(u, nothing, nothing) ≈ out
    @test !(diss(u, (; μ=1), nothing) ≈ out)
    @test diss(u, (; μ=diss.lead.μ), nothing) ≈ out
    @test diss(similar(out), u, nothing, nothing) ≈ out

    out = lazyls * u
    @test abs(tr(out)) < 1e-10
    @test lazyls(u) ≈ out
    @test lazyls(u, nothing, nothing) ≈ out
    @test !(lazyls(u, (; left=(; μ=1)), nothing) ≈ out)
    @test lazyls(u, (; left=(; μ=lazyls.dissipators[:left].lead.μ)), nothing) ≈ out
    @test lazyls(similar(out), u, nothing, nothing) ≈ out
    @test lazyls(similar(out), u, Dict(:left => (; μ=1.0)), nothing) ≈ lazyls(u, Dict(:left => (; μ=1)), nothing)

    out = QuantumDots.internal_rep(out, ls)
    um = QuantumDots.internal_rep(u, ls)
    @test ls(um, nothing, nothing) ≈ out
    ls2 = LindbladSystem(hamiltonian, leads; usecache=true)
    @test ls2(similar(out), um, Dict(:left => (; μ=1.0)), nothing) ≈ ls(um, Dict(:left => (; μ=1)), nothing)

    # cm0 = conductance_matrix(AD.FiniteDifferencesBackend(), ls, ρinternal1, particle_number)
    @test_broken conductance_matrix(AD.AutoFiniteDifferences(central_fdm(3,1)), lazyls, ρinternal2, particle_number) #Needs AD of LazyLindbladDissipator, which is not a matrix
    @test_broken cm2 = conductance_matrix(AD.AutoForwardDiff(), lazyls, ρinternal2, particle_number) #Same as above

    ls2 = QuantumDots.__update_coefficients(ls, (; left=(; μ=0.1)))
    @test ls2.dissipators[:left].lead.μ ≈ 0.1
    @test_throws ArgumentError QuantumDots.__update_coefficients!(ls, (; left=(; μ=0.1)))

    ls1 = LindbladSystem(hamiltonian, leads; usecache=true)
    lazyls1 = LazyLindbladSystem(hamiltonian, leads)
    ls2 = QuantumDots.__update_coefficients!(ls1, (; left=(; μ=0.1)))
    lazyls2 = QuantumDots.__update_coefficients!(lazyls1, (; left=(; μ=0.1)))
    @test ls2.dissipators[:left].lead.μ ≈ 0.1
    @test lazyls2.dissipators[:left].lead.μ ≈ 0.1
    @test ls1.dissipators[:left].lead.μ ≈ 0.1
    @test lazyls1.dissipators[:left].lead.μ ≈ 0.1
end

@testitem "Khatri-Rao" begin
    using Random, BlockDiagonals, LinearAlgebra
    Random.seed!(1234)

    bdm = BlockDiagonal([rand(2, 2), rand(3, 3), rand(5, 5)])
    bz = size.(blocks(bdm), 1)
    kv = QuantumDots.KhatriRaoVectorizer(bz)
    m = Matrix(bdm)
    @test Matrix(QuantumDots.khatri_rao_lazy_dissipator(bdm, kv)) ≈ Matrix(QuantumDots.khatri_rao_dissipator(bdm, kv))
    @test Matrix(QuantumDots.khatri_rao_lazy_dissipator(bdm, kv)) ≈ Matrix(QuantumDots.khatri_rao_lazy_dissipator(m, kv))
    @test Matrix(QuantumDots.khatri_rao_lazy_dissipator(bdm, kv)) ≈ QuantumDots.khatri_rao_dissipator(m, kv)
    @test QuantumDots.khatri_rao_dissipator(m, kv) isa Matrix
    @test QuantumDots.khatri_rao(m, m, kv) ≈ cat(map(kron, bdm.blocks, bdm.blocks)...; dims=(1, 2))
    @test QuantumDots.khatri_rao(m, m, kv) ≈
          QuantumDots.khatri_rao(bdm, bdm, kv) ≈
          QuantumDots.khatri_rao(m, bdm, kv) ≈
          QuantumDots.khatri_rao(m, bdm, kv) ≈
          QuantumDots.khatri_rao(bdm, m, kv) ≈
          Matrix(QuantumDots.khatri_rao_lazy(m, m, kv)) ≈
          Matrix(QuantumDots.khatri_rao_lazy(m, bdm, kv)) ≈
          Matrix(QuantumDots.khatri_rao_lazy(bdm, bdm, kv)) ≈
          Matrix(QuantumDots.khatri_rao_lazy(bdm, m, kv))

    d = Diagonal(m)
    dkv = QuantumDots.KhatriRaoVectorizer(fill(1, size(d, 1)))
    @test QuantumDots.khatri_rao(d, d, dkv) ≈ d^2 ≈
          Matrix(QuantumDots.khatri_rao_lazy(d, d, dkv))

    @test QuantumDots.khatri_rao_commutator(m, kv) ≈ QuantumDots.khatri_rao_commutator(bdm, kv)
    id = I(size(m, 1))
    @test QuantumDots.khatri_rao_commutator(m, kv) ≈
          QuantumDots.khatri_rao(id, m, kv) - QuantumDots.khatri_rao(transpose(m), id, kv) ≈
          QuantumDots.khatri_rao_commutator(bdm, kv) ≈
          QuantumDots.khatri_rao(id, bdm, kv) - QuantumDots.khatri_rao(transpose(bdm), id, kv) ≈
          Matrix(QuantumDots.khatri_rao_lazy_commutator(m, kv)) ≈
          Matrix(QuantumDots.khatri_rao_lazy_commutator(bdm, kv))

end

@testitem "TSL" begin
    using Random, Symbolics, BlockDiagonals
    Random.seed!(1234)

    p = (; zip([:μL, :μC, :μR, :h, :t, :Δ, :tsoc, :U], rand(8))...)
    tsl, tsl!, m, c = QuantumDots.TSL_generator()
    @test c isa FermionBasis{6}
    m2 = tsl(; p...)
    tsl!(m; p...)
    @test m ≈ m2

    tsl, tsl!, m, c = QuantumDots.TSL_generator(QuantumDots.parity)
    @test c isa FermionBasis{6}
    @test m isa BlockDiagonal
    m2 = tsl(; p...)
    tsl!(m; p...)
    @test m ≈ m2

    tsl, tsl!, m, c = QuantumDots.TSL_generator(; dense=true)
    @test m isa Matrix{Float64}
    m2 = tsl(; p...)
    tsl!(m; p...)
    @test m ≈ reshape(m2, size(m))

    tsl, tsl!, m, c = QuantumDots.TSL_generator(; bdg=true, dense=true)
    @test c isa QuantumDots.FermionBdGBasis{6}
    @test m isa Matrix{Float64}
    m2 = tsl(; p...)
    tsl!(m; p...)
    @test m ≈ reshape(m2, size(m))
end
