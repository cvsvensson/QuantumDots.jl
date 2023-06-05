using QuantumDots
using Test, LinearAlgebra, SparseArrays, Random, Krylov, BlockDiagonals
Random.seed!(1234)

@testset "QuantumDots.jl" begin

end

@testset "Fock" begin
    N = 6
    focknumber = 20 # = 16+4 = 00101
    fbits = bits(focknumber,N)
    @test fbits == [0,0,1,0,1,0]

    @test QuantumDots.focknbr(fbits) == 20
    @test QuantumDots.focknbr(Tuple(fbits)) == 20
    @test !QuantumDots._bit(focknumber,1)
    @test !QuantumDots._bit(focknumber,2)
    @test QuantumDots._bit(focknumber,3)
    @test !QuantumDots._bit(focknumber,4)
    @test QuantumDots._bit(focknumber,5)

    @test QuantumDots.focknbr((3,5)) == 20
    @test QuantumDots.focknbr([3,5]) == 20

    @testset "removefermion" begin
        focknbr = rand(1:2^N) - 1
        fockbits = bits(focknbr,N)
        function test_remove(n)
            QuantumDots.removefermion(n,focknbr) == (fockbits[n] ? (focknbr - 2^(n-1), (-1)^sum(fockbits[1:n-1])) : (0,0))
        end
        all([test_remove(n) for n in 1:N])
    end

    @testset "ToggleFermions" begin
        focknbr = 177 # = 1000 1101, msb to the right
        digitpositions = Vector([7, 8, 2, 3])
        daggers = BitVector([1,0,1,1])
        newfocknbr, sign = QuantumDots.togglefermions(digitpositions, daggers, focknbr)
        @test newfocknbr == 119 # = 1110 1110
        @test sign == -1
        # swap two operators
        digitpositions = Vector([7, 2, 8, 3])
        daggers = BitVector([1,1,0,1])
        newfocknbr, sign = QuantumDots.togglefermions(digitpositions, daggers, focknbr)
        @test newfocknbr == 119 # = 1110 1110
        @test sign == 1
    
        # annihilate twice
        digitpositions = Vector([5, 3, 5])
        daggers = BitVector([0, 1, 0])
        _, sign = QuantumDots.togglefermions(digitpositions, daggers, focknbr)
        @test sign == 0
    end

    fs = QuantumDots.fockstates(10,5)
    @test length(fs) == binomial(10,5)
    @test allunique(fs)
    @test all(QuantumDots.fermionnumber.(fs) .== 5)
    
end

@testset "Basis" begin
    N = 6
    B = FermionBasis(1:N)
    Bdense = FermionBasis(1:N; sparse = false)
    @test QuantumDots.nbr_of_fermions(B) == N
    Bspin = FermionBasis(1:N,(:↑,:↓))
    @test QuantumDots.nbr_of_fermions(Bspin) == 2N
    @test B[1] isa SparseMatrixCSC
    @test Bspin[1,:↑] isa SparseMatrixCSC
    @test parityoperator(B) isa SparseMatrixCSC
    @test parityoperator(Bspin) isa SparseMatrixCSC
    @test Bdense[1] isa Matrix
    @test parityoperator(Bdense) isa Matrix

    (c,) = QuantumDots.cell(1,B)
    @test c == B[1]
    (c1,c2) = QuantumDots.cell(1,Bspin)
    @test c1 == Bspin[1,:↑]
    @test c2 == Bspin[1,:↓]
    
    a = FermionBasis(1:3)
    v = [QuantumDots.indtofock(i,a) for i in 1:8]
    t1 = QuantumDots.tensor(v,a)
    t2 = [i1 + 2i2 + 4i3 for i1 in (0,1), i2 in (0,1), i3 in (0,1)]
    @test t1 == t2

    a = FermionBasis(1:3; qn = QuantumDots.parity)
    v = [QuantumDots.indtofock(i,a) for i in 1:8]
    t1 = QuantumDots.tensor(v,a)
    t2 = [i1 + 2i2 + 4i3 for i1 in (0,1), i2 in (0,1), i3 in (0,1)]
    @test t1 == t2

    @test sort(QuantumDots.svd(v,(1,),a).S .^2) ≈ eigvals(QuantumDots.reduced_density_matrix(v,(1,),a))
    
    c = FermionBasis(1:2,(:a,:b))
    cparity = FermionBasis(1:2,(:a,:b); qn = QuantumDots.parity)
    ρ = Matrix(Hermitian(rand(2^4,2^4) .- .5))
    ρ = ρ/tr(ρ)
    function bilinears(c,labels)
        ops = reduce(vcat,[[c[l], c[l]'] for l in labels])
        return [op1*op2 for (op1,op2) in Base.product(ops,ops)]
    end
    function bilinear_equality(c,csub,ρ)
        subsystem = Tuple(keys(csub))
        ρsub = QuantumDots.reduced_density_matrix(ρ,csub,c)
        @test tr(ρsub) ≈ 1
        all((tr(op1*ρ) ≈ tr(op2*ρsub)) for (op1,op2) in zip(bilinears(c,subsystem), bilinears(csub,subsystem)))
    end
    function get_subsystems(c,N)
        t = collect(Base.product(ntuple(i->keys(c),N)...))
        (t[I] for I in CartesianIndices(t) if issorted(Tuple(I)) && allunique(Tuple(I)))
    end
    for N in 1:4
        @test all(bilinear_equality(c,FermionBasis(subsystem),ρ) for subsystem in get_subsystems(c,N))
        @test all(bilinear_equality(c,FermionBasis(subsystem; qn = QuantumDots.parity),ρ) for subsystem in get_subsystems(c,N))
        @test all(bilinear_equality(c,FermionBasis(subsystem; qn = QuantumDots.parity),ρ) for subsystem in get_subsystems(cparity,N))
        @test all(bilinear_equality(c,FermionBasis(subsystem),ρ) for subsystem in get_subsystems(cparity,N))
    end
    @test_throws AssertionError bilinear_equality(c,FermionBasis(((1,:b),(1,:a))),ρ) 
end

@testset "BdG" begin
    N=2
    labels = Tuple(1:N)
    μ1 = rand()
    μ2 = rand()
    b = QuantumDots.FermionBdGBasis(labels)
    
    @test iszero(b[1]*b[1])
    @test iszero(b[1]'*b[1]')
    @test iszero(*(b[1],b[1]; symmetrize=false))
    @test iszero(*(b[1]',b[1]'; symmetrize=false))
    @test QuantumDots.cell(1, b)[1] == b[1]
    @test length(QuantumDots.cell(1, b)) == 1

    @test b[1] isa QuantumDots.BdGFermion
    @test b[1]' isa QuantumDots.BdGFermion
    @test b[1]*b[1] isa SparseMatrixCSC
    @test b[1].hole
    @test !b[1]'.hole
    @test b[1] + b[1] isa QuantumDots.QuasiParticle
    @test b[1] - b[1] isa QuantumDots.QuasiParticle
    @test norm(QuantumDots.rep(b[1] - b[1])) == 0
    @test QuantumDots.rep(b[1] + b[1]) == 2QuantumDots.rep(b[1])

    vals, vecs = QuantumDots.enforce_ph_symmetry(eigen(Matrix(μ1*b[1]'*b[1] + μ2*b[2]'*b[2])))
    @test norm(vals - sort([-μ1,-μ2,μ1,μ2])) < 1e-14
    @test QuantumDots.ground_state_parity(vals,vecs) == 1
    vals, vecs = QuantumDots.enforce_ph_symmetry(eigen(Matrix(μ1*b[1]'*b[1] - μ2*b[2]'*b[2])))
    @test QuantumDots.ground_state_parity(vals,vecs) == -1
    
    t = Δ = 1
    ham(b) =  Matrix(QuantumDots.kitaev_hamiltonian(b; μ= 0,t,Δ,V=0))
    poor_mans_ham = ham(b) #Matrix(QuantumDots.kitaev_hamiltonian(b; μ= 0,t,Δ,V=0))
    es, ops = QuantumDots.enforce_ph_symmetry(eigen(poor_mans_ham))
    @test QuantumDots.check_ph_symmetry(es, ops)
    @test norm(sort(es,by=abs)[1:2]) < 1e-12
    qps = map(op -> QuantumDots.QuasiParticle(op,b), eachcol(ops))
    @test all(map(qp->iszero(qp*qp), qps))

    b_mb = QuantumDots.FermionBasis(labels)
    poor_mans_ham_mb = ham(b_mb)#Matrix(QuantumDots.kitaev_hamiltonian(b_mb; μ= 0,t,Δ,V=0))
    es_mb, states = eigen(poor_mans_ham_mb)
    P = QuantumDots.parityoperator(b_mb)
    parity(v) = v'*P*v
    gs_odd = parity(states[:,1]) ≈ -1 ? states[:,1] : states[:,2]
    gs_even = parity(states[:,1]) ≈ 1 ? states[:,1] : states[:,2]
    
    gs_parity = QuantumDots.ground_state_parity(es, ops)
    ρeven, ρodd = if gs_parity == 1
        one_particle_density_matrix(qps[1:2]),
        one_particle_density_matrix(qps[[1,3]])
    else
        one_particle_density_matrix(qps[[1,3]]),
        one_particle_density_matrix(qps[1:2])
    end
    ρeven_mb = one_particle_density_matrix(gs_even*gs_even',b_mb)
    ρodd_mb = one_particle_density_matrix(gs_odd*gs_odd',b_mb)
    qps_mb = map(qp->QuantumDots.many_body_fermion(qp,b_mb), qps)
   
    @test ρeven ≈ ρeven_mb
    @test ρodd ≈ ρodd_mb
    @test ρodd[[1,3],[1,3]] ≈ ρeven[[1,3],[1,3]]
    @test ρodd[[2,4],[2,4]] ≈ ρeven[[2,4],[2,4]]

    @test (gs_parity == 1 ? ρeven : ρodd) ≈ QuantumDots.one_particle_density_matrix(ops)

    @test poor_mans_ham ≈ mapreduce((e,qp)->e*qp'*qp/2, +,es,qps)

    qp = qps[2]
    @test qp isa QuantumDots.QuasiParticle
    @test 2*qp isa QuantumDots.QuasiParticle
    @test qp*2 isa QuantumDots.QuasiParticle
    @test qp/2 isa QuantumDots.QuasiParticle
    @test qp + qp isa QuantumDots.QuasiParticle
    @test qp - qp isa QuantumDots.QuasiParticle
    @test qp + b[1] isa QuantumDots.QuasiParticle
    @test b[1] + qp isa QuantumDots.QuasiParticle
    @test qp - b[1] isa QuantumDots.QuasiParticle
    @test b[1] - qp isa QuantumDots.QuasiParticle
    @test abs(QuantumDots.majorana_polarization(qp)) ≈ 1

    @test_nowarn QuantumDots.visualize(qp)
    @test_nowarn QuantumDots.majvisualize(qp)
    @test qp[(1,:h)] == qp[1,:h]

    us, vs = (rand(length(labels)), rand(length(labels)))
    normalize!(us)
    normalize!(vs)
    vs = vs - dot(us,vs)*us
    vs = normalize!(vs)/sqrt(2)
    us = normalize!(us)/sqrt(2)
    χ = sum(us .* [b[i] for i in keys(b)]) + sum(vs .* [b[i]' for i in keys(b)])
    @test iszero(χ*χ)
    @test iszero(χ'*χ')
    χ_mb = sum(us .* [b_mb[i] for i in keys(b_mb)]) + sum(vs .* [b_mb[i]' for i in keys(b)])
    @test χ_mb ≈ QuantumDots.many_body_fermion(χ,b_mb)
    @test χ_mb' ≈ QuantumDots.many_body_fermion(χ',b_mb)
    
    @test all(b_mb[k] ≈ QuantumDots.many_body_fermion(b[k],b_mb) for k in 1:N)
    @test all(b_mb[k]' ≈ QuantumDots.many_body_fermion(b[k]',b_mb) for k in 1:N)
end

@testset "QN" begin
    function testsym(sym)
        qnsv = [(qn,) for qn in qns(sym)]
        blocksv = [rand(QuantumDots.blocksize(qn,sym)) .- 1/2 for qn in qns(sym)]
        v = QArray(qnsv,blocksv,(sym,))
        
        qnmat = collect(Base.product(qns(sym),qns(sym)))
        dind = diagind(qnmat)
        qnsm = vec(qnmat)
        blocksm = [rand(QuantumDots.blocksize(qn1,sym),QuantumDots.blocksize(qn2,sym)) .- 1/2 for (qn1,qn2) in qnsm]
        m = QuantumDots.QArray(qnsm,blocksm, (sym,sym))
        md = QuantumDots.QArray(qnsm[dind],blocksm[dind], (sym,sym))
        
        ma = Array(m);
        va = Array(v);
        mda = Array(md);

        @test size(v) == size(va)
        @test size(m) == size(ma)
        @test Array(m*v) ≈ ma*va
        @test Array(md*v) ≈ mda*va

        @test v'*v ≈ va'*va
        @test v'*(m*v) ≈ (v'*m)*v ≈ va'*ma*va

        @test all([ind == QuantumDots.qnindtoind(QuantumDots.indtoqnind(ind,sym),sym) for ind in eachindex(va)])
    end
    testsym(Z2Symmetry{1}())
    testsym(Z2Symmetry{4}())
    testsym(QuantumDots.U1Symmetry{1}())
    testsym(QuantumDots.U1Symmetry{5}())
end

@testset "Kitaev" begin
    N = 4
    c = FermionBasis(1:N)
    ham = Matrix(QuantumDots.kitaev_hamiltonian(c; μ = 0, t = 1, Δ = 1))
    vals, vecs = eigen(ham)
    @test abs(vals[1] - vals[2]) < 1e-12
    p = parityoperator(c)
    v1,v2 = eachcol(vecs[:,1:2])
    @test dot(v1,p,v1)*dot(v2,p,v2) ≈ -1
    w = [dot(v1,f+f',v2) for f in c.dict]
    z = [dot(v1,(f'-f),v2) for f in c.dict]
    @test abs.(w.^2 - z.^2) ≈ [1,0,0,1]
    
    N = 4
    c = FermionBasis(1:N; qn = QuantumDots.parity)
    ham = QuantumDots.blockdiagonal(Matrix(QuantumDots.kitaev_hamiltonian(c; μ = 0, t = 1, Δ = 1)), c)
    vals, vecs = BlockDiagonals.eigen_blockwise(ham)
    @test abs(vals[1] - vals[1+size(vecs.blocks[1],1)]) < 1e-12
    p = parityoperator(c)
    v1 = vecs[:,1]
    v2 = vecs[:,1+size(vecs.blocks[1],1)]
    @test dot(v1,p,v1)*dot(v2,p,v2) ≈ -1
    w = [dot(v1,f+f',v2) for f in c.dict]
    z = [dot(v1,(f'-f),v2) for f in c.dict]
    @test abs.(w.^2 - z.^2) ≈ [1,0,0,1]
end

@testset "Parity and number operator" begin
    function get_ops(qn)
        N = 2
        a = FermionBasis(1:N; qn)
        ham = a[1]'*a[1] + π*a[2]'*a[2]
        vals,vecs = eigen(Matrix(ham))
        @test vals ≈ [0,1,π,π+1]
        parityop = parityoperator(a)
        numberop = numberoperator(a)
        @test all([v'* parityop * v for v in eachcol(vecs)] .≈ [1,-1,-1,1])
        @test Diagonal(diag(parityop)) == parityop
        @test Diagonal(diag(numberop)) == numberop
        @test sum(a[i]'a[i] for i in 1:N) == numberop
        @test prod(2(a[i]'a[i] - 1/2*sparse(I,2^N,2^N)) for i in 1:N) == parityop
        return parityop,numberop
    end
    parityop, numberop = get_ops(QuantumDots.NoSymmetry())
    @test diag(parityop) == [1,-1,-1,1]
    @test diag(numberop) == [0,1,1,2]
    
    parityop, numberop = get_ops(QuantumDots.parity)
    @test diag(parityop) == [-1,-1,1,1]
    @test diag(numberop) == [1,1,0,2]
    
    parityop, numberop = get_ops(QuantumDots.fermionnumber)
    @test diag(parityop) == [1,-1,-1,1]
    @test diag(numberop) == [0,1,1,2]

end

@testset "BlockDiagonal" begin
    N = 2
    a = FermionBasis(1:N; qn = QuantumDots.parity)
    ham0 = a[1]'*a[1] + π*a[2]'*a[2]
    ham = blockdiagonal(ham0, a)
    @test ham isa BlockDiagonal{Float64,SparseMatrixCSC{Float64,Int}}
    ham = blockdiagonal(Matrix,ham0, a)
    @test ham isa BlockDiagonal{Float64,Matrix{Float64}}
    vals,vecs = eigen(ham)
    @test vals ≈ [0,1,π,π+1]
    parityop = blockdiagonal(parityoperator(a),a)
    numberop = blockdiagonal(numberoperator(a),a)
end

@testset "Fast generated hamiltonians" begin
    N = 5
    params = rand(3)
    hamiltonian(a,μ,t,Δ) = μ*sum(a[i]'a[i] for i in 1:QuantumDots.nbr_of_fermions(a)) + t*(a[1]'a[2] + a[2]'a[1]) + Δ*(a[1]'a[2]' + a[2]a[1])
    
    a = FermionBasis(1:N) 
    hamiltonian(params...) = hamiltonian(a,params...)
    fastham! = QuantumDots.fastgenerator(hamiltonian,3);
    mat = hamiltonian((2 .* params)...)
    fastham!(mat,params...)
    @test mat ≈ hamiltonian(params...)

    #parity conservation
    a = FermionBasis(1:N; qn= QuantumDots.parity) 
    hamiltonian(params...) = hamiltonian(a,params...)
    parityham! = QuantumDots.fastgenerator(hamiltonian,3);
    mat = hamiltonian((2 .* params)...)
    parityham!(mat,params...)
    @test mat ≈ hamiltonian(params...)

    _bd(m) = blockdiagonal(m,a).blocks
    bdham = _bd ∘ hamiltonian

    oddham! = QuantumDots.fastgenerator(first ∘ bdham,3);
    oddmat = bdham((2 .* params)...) |> first
    oddham!(oddmat,params...)
    @test oddmat ≈ bdham(params...) |> first

    evenham! = QuantumDots.fastgenerator(last ∘ bdham,3);
    evenmat = bdham((2 .* params)...) |> last
    evenham!(evenmat,params...)
    @test evenmat ≈ bdham(params...) |> last
    
    _bd2(xs...) = blockdiagonal(hamiltonian(xs...), a)
    paritybd! = QuantumDots.fastblockdiagonal(_bd2,3);
    bdham = _bd2(2params...)
    paritybd!(bdham,params...)
    @test bdham ≈ _bd2(params...)
    @test bdham.blocks |> first ≈ oddmat
    @test bdham.blocks |> last ≈ evenmat

    #number conservation
    a = FermionBasis(1:N; qn= QuantumDots.fermionnumber) 
    hamiltonian(params...) = hamiltonian(a,params...)

    numberham! = QuantumDots.fastgenerator(hamiltonian,3);
    mat = hamiltonian((2 .* params)...)
    numberham!(mat,params...)
    @test mat ≈ hamiltonian(params...)

    numberbdham(params...) = blockdiagonal(hamiltonian(params...), a)
    numberbd! = QuantumDots.fastblockdiagonal(numberbdham,3);
    bdham = numberbdham(2params...)
    numberbd!(bdham,params...)
    @test bdham ≈ numberbdham(params...)
    @test bdham ≈ hamiltonian(params[1:end-1]...,0.0)

    b = FermionBasis(1:2,(:a,:b); qn = QuantumDots.parity)
    nparams = 8
    params = rand(nparams)
    ham = (t, Δ, V, θ, h, U, Δ1, μ) -> Matrix(QuantumDots.BD1_hamiltonian(b; μ, t, Δ, V, θ, h, U, Δ1,ϕ = 0))
    hammat = ham(params...)
    fastgen! = QuantumDots.fastgenerator(ham, nparams)
    hammat2 = ham(rand(Float64,nparams)...)
    fastgen!(hammat2,params...) 
    @test hammat2 ≈ hammat

    hambd(p...) = QuantumDots.blockdiagonal(ham(p...),b)
    @test sort!(abs.(eigvals(hambd(params...)))) ≈ sort!(abs.(eigvals(hammat)))

    fastgen! = QuantumDots.fastblockdiagonal(hambd, nparams)
    bdhammat2 = hambd(rand(nparams)...)
    fastgen!(bdhammat2,params...) 
    @test hambd(params...) ≈ bdhammat2

end

@testset "rotations" begin
    N=2
    b = FermionBasis(1:N,(:↑,:↓))
    b2 = FermionBasis(1:N,(:↑,:↓); sparse = false)
    standard_hopping = QuantumDots.hopping(1,b[1,:↑],b[2,:↑]) +  QuantumDots.hopping(1,b[1,:↓],b[2,:↓])
    standard_pairing = QuantumDots.pairing(1,b[1,:↑],b[2,:↓]) - QuantumDots.pairing(1,b[1,:↓],b[2,:↑])
    local_pairing = sum(QuantumDots.pairing(1,QuantumDots.cell(j,b)...) for j in 1:N)
    θ = rand()
    ϕ = rand()
    @test QuantumDots.hopping_rotated(1,QuantumDots.cell(1,b), QuantumDots.cell(2,b),(0,0),(0,0)) ≈ standard_hopping
    @test QuantumDots.hopping_rotated(1,QuantumDots.cell(1,b), QuantumDots.cell(2,b),(θ,ϕ),(θ,ϕ)) ≈ standard_hopping
    @test QuantumDots.pairing_rotated(1,QuantumDots.cell(1,b), QuantumDots.cell(2,b),(0,0),(0,0)) ≈ standard_pairing
    @test QuantumDots.pairing_rotated(1,QuantumDots.cell(1,b), QuantumDots.cell(2,b),(θ,ϕ),(θ,ϕ)) ≈ standard_pairing

    soc = QuantumDots.hopping(exp(1im*ϕ),b[1,:↓],b[2,:↑]) - QuantumDots.hopping(exp(-1im*ϕ),b[1,:↑],b[2,:↓])
    @test QuantumDots.hopping_rotated(1,QuantumDots.cell(1,b), QuantumDots.cell(2,b),(0,0),(θ,ϕ)) ≈ standard_hopping*cos(θ/2) + sin(θ/2)*soc

    Δk = QuantumDots.pairing(exp(1im*ϕ),b[1,:↑],b[2,:↑]) + QuantumDots.pairing(exp(-1im*ϕ),b[1,:↓],b[2,:↓])
    @test QuantumDots.pairing_rotated(1,QuantumDots.cell(1,b), QuantumDots.cell(2,b),(0,0),(θ,ϕ)) ≈ standard_pairing*cos(θ/2) + sin(θ/2)*Δk

    @test standard_hopping ≈ QuantumDots.BD1_hamiltonian(b; t=1,μ=0,V=0,U=0,h=0,θ = (0,:diff),ϕ=(0,:diff),Δ = 0,Δ1 = 0)
    @test standard_pairing ≈ QuantumDots.BD1_hamiltonian(b; t=0,μ=0,V=0,U=0,h=0,θ=(0,:diff),ϕ=(0,:diff),Δ = 0,Δ1 = 1)
    @test QuantumDots.BD1_hamiltonian(b; t=0,μ=0,V=0,U=0,h=0,θ=(θ,:diff),ϕ=(ϕ,:diff),Δ = 1,Δ1 = 0) ≈ local_pairing
    
    @test QuantumDots.BD1_hamiltonian(b; t=0,μ=1,V=0,U=0,h=0,θ=(θ,:diff),ϕ=(ϕ,:diff),Δ = 0,Δ1 = 0) ≈ -QuantumDots.numberoperator(b)

    @test QuantumDots.BD1_hamiltonian(b; t=0,μ=1,V=0,U=0,h=0,θ=(θ,:diff),ϕ=(ϕ,:diff),Δ = 0,Δ1 = 0) == QuantumDots.BD1_hamiltonian(b; t=0,μ=1,V=0,U=0,h=0,θ=θ.*[0,1],ϕ=ϕ.*[0,1],Δ = 0,Δ1 = 0)
    
    
    H0 = Matrix(QuantumDots.BD1_hamiltonian(b; t=1,μ=1,V=2,U=3,h=4,θ=(1/2,:diff),ϕ=(1/3,:diff),Δ = -2,Δ1 = -10))
    H1 = zero(H0)
    @test H0 ≈ QuantumDots.BD1_hamiltonian!(H1,b; t=1,μ=1,V=2,U=3,h=4,θ=(1/2,:diff),ϕ=(1/3,:diff),Δ = -2,Δ1 = -10)
end

@testset "transport" begin
    N = 1
    a = FermionBasis(1:N)
    hamiltonian(μ) = μ*sum(a[i]'a[i] for i in 1:N)
    T = rand()
    μL,μR,μH = rand(3)
    leftlead = QuantumDots.NormalLead(T,μL; in = a[1]', out = a[1])
    rightlead = QuantumDots.NormalLead(T,μR; in = a[N]', out = a[N])
    particle_number = numberoperator(a)
    system = QuantumDots.OpenSystem(hamiltonian(μH),[leftlead, rightlead])
    measurements = [particle_number]
    lindbladsystem, transformed_measurements = QuantumDots.prepare_lindblad(system, measurements)
    @test diag(lindbladsystem.system.hamiltonian.eigenvalues) ≈ [0.0, μH]
    lindbladsystem2, _ = QuantumDots.prepare_lindblad(system, []; dE=μH/2)
    @test diag(lindbladsystem2.system.hamiltonian.eigenvalues) ≈ [0.0]

    ρ = QuantumDots.stationary_state(lindbladsystem)
    rhod = diag(ρ)
    p2 = (QuantumDots.fermidirac(μH,T,μL) + QuantumDots.fermidirac(μH,T,μR))/2
    p1 = 1 - p2
    analytic_current = -1/2*(QuantumDots.fermidirac(μH,T,μL) - QuantumDots.fermidirac(μH,T,μR))
    @test rhod ≈ [p1, p2]

    numeric_current = QuantumDots.measure(ρ,transformed_measurements[1],lindbladsystem)#real.(QuantumDots.conductance(system,[particle_number])[1])
    @test abs(sum(numeric_current)) < 1e-10
    @test sum(numeric_current; dims = 2) ≈ analytic_current .* [-1, 1] #Why not flip the signs?


    N = 1
    a = FermionBasis(1:N; qn = QuantumDots.parity)
    hamiltonian(μ) = QuantumDots.blockdiagonal(μ*sum(a[i]'a[i] for i in 1:N),a)
    particle_number = QuantumDots.blockdiagonal(numberoperator(a),a)
    leftlead = QuantumDots.NormalLead(T,μL; in = a[1]', out = a[1])
    rightlead = QuantumDots.NormalLead(T,μR; in = a[N]', out = a[N])
    system = QuantumDots.OpenSystem(hamiltonian(μH),[leftlead, rightlead])
    lindbladsystem, transformed_measurements = QuantumDots.prepare_lindblad(system, [particle_number]);
    @test diag(lindbladsystem.system.hamiltonian.eigenvalues) ≈ [μH, 0.0]
    lindbladsystem2, _ = QuantumDots.prepare_lindblad(system, []; dE=μH/2)
    @test diag(lindbladsystem2.system.hamiltonian.eigenvalues) ≈ [0.0]

    ρ = QuantumDots.stationary_state(lindbladsystem)
    rhod = diag(ρ)
    p2 = (QuantumDots.fermidirac(μH,T,μL) + QuantumDots.fermidirac(μH,T,μR))/2
    p1 = 1 - p2
    analytic_current = -1/2*(QuantumDots.fermidirac(μH,T,μL) - QuantumDots.fermidirac(μH,T,μR))
    @test rhod ≈ [p2, p1]
    numeric_current = QuantumDots.measure(ρ,transformed_measurements[1],lindbladsystem) #real.(QuantumDots.conductance(system,[particle_number])[1])
    @test abs(sum(numeric_current)) < 1e-10
    @test sum(numeric_current; dims = 2) ≈ analytic_current .* [-1, 1] #Why not flip the signs?

    N = 1
    a = FermionBasis(1:N; qn = QuantumDots.fermionnumber)
    hamiltonian(μ) = QuantumDots.blockdiagonal(μ*sum(a[i]'a[i] for i in 1:N),a)
    particle_number = QuantumDots.blockdiagonal(numberoperator(a),a)
    leftlead = QuantumDots.NormalLead(T,μL; in = a[1]', out = a[1])
    rightlead = QuantumDots.NormalLead(T,μR; in = a[N]', out = a[N])
    system = QuantumDots.OpenSystem(hamiltonian(μH),[leftlead, rightlead])
    lindbladsystem, transformed_measurements = QuantumDots.prepare_lindblad(system, [particle_number])
    @test diag(lindbladsystem.system.hamiltonian.eigenvalues) ≈ [0.0, μH]
    lindbladsystem2, _ = QuantumDots.prepare_lindblad(system, []; dE=μH/2)
    @test diag(lindbladsystem2.system.hamiltonian.eigenvalues) ≈ [0.0]

    ρ = QuantumDots.stationary_state(lindbladsystem)
    rhod = diag(ρ)
    p2 = (QuantumDots.fermidirac(μH,T,μL) + QuantumDots.fermidirac(μH,T,μR))/2
    p1 = 1 - p2
    analytic_current = -1/2*(QuantumDots.fermidirac(μH,T,μL) - QuantumDots.fermidirac(μH,T,μR))
    @test rhod ≈ [p1, p2]
    numeric_current = QuantumDots.measure(ρ,transformed_measurements[1],lindbladsystem) #real.(QuantumDots.conductance(system,[particle_number])[1])
    @test abs(sum(numeric_current)) < 1e-10
    @test sum(numeric_current; dims = 2) ≈ analytic_current .* [-1, 1] #Why not flip the signs?
end
