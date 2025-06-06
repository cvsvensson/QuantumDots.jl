
"""
    fermionic_kron(ms::AbstractVector, Hs::AbstractVector{<:AbstractHilbertSpace}, H::AbstractHilbertSpace=wedge(bs))

Compute the fermionic tensor product of matrices or vectors in `ms` with respect to the fermion bases `bs`, respectively. Return a matrix in the fermion basis `b`, which defaults to the wedge product of `bs`.
"""
function fermionic_kron(ms, Hs, H::AbstractHilbertSpace=wedge(Hs), phase_factors=use_wedge_phase_factors(Hs, H); match_labels=true)
    N = ndims(first(ms))
    mout = allocate_wedge_result(ms, Hs)

    fockmapper = if match_labels
        fermionpositions = map(Base.Fix2(siteindices, H.jw) ∘ collect ∘ keys, Hs)
        FockMapper(fermionpositions)
    else
        Ms = map(nbr_of_modes, Hs)
        shifts = (0, cumsum(Ms)...)
        FockShifter(shifts)
    end

    if N == 1
        return fermionic_kron_vec!(mout, Tuple(ms), Tuple(Hs), H, fockmapper)
    elseif N == 2
        return fermionic_kron_mat!(mout, Tuple(ms), Tuple(Hs), H, fockmapper, phase_factors)
    end
    throw(ArgumentError("Only 1D or 2D arrays are supported"))
end

fermionic_kron(Hs::Pair, phase_factors=use_wedge_phase_factors(Hs...); match_labels=true) = (ms...) -> fermionic_kron(ms, Hs, phase_factors; match_labels)
fermionic_kron(ms, Hs::Pair, phase_factors=use_wedge_phase_factors(Hs...); match_labels=true) = fermionic_kron(ms, first(Hs), last(Hs), phase_factors; match_labels)


uniform_to_sparse_type(::Type{UniformScaling{T}}) where {T} = SparseMatrixCSC{T,Int}
uniform_to_sparse_type(::Type{T}) where {T} = T
function allocate_wedge_result(ms, bs)
    T = Base.promote_eltype(ms...)
    N = ndims(first(ms))
    types = map(uniform_to_sparse_type ∘ typeof, ms)
    MT = Base.promote_op(kron, types...)
    dimlengths = map(length ∘ focknumbers, bs)
    Nout = prod(dimlengths)
    _mout = Zeros(T, ntuple(j -> Nout, N))
    try
        convert(MT, _mout)
    catch
        Array(_mout)
    end
end

wedge_iterator(m, ::AbstractFockHilbertSpace) = findall(!iszero, m)
wedge_iterator(::UniformScaling, H::AbstractFockHilbertSpace) = diagind(I(length(focknumbers(H))), IndexCartesian())
# wedge_iterator(::UniformScaling, b::FermionBasisTemplate) = diagind(I(length(focknumbers(b))), IndexCartesian())

function use_wedge_phase_factors(Hs, H::AbstractHilbertSpace)
    #check if all Hs and H have the same fermionic property, otherwise throw error
    f = isfermionic(H)
    all(H -> isfermionic(H) == f, Hs) || throw(ArgumentError("All Hilbert spaces should have the same fermionicity"))
    return f
end

function fermionic_kron_mat!(mout, ms::Tuple, Hs::Tuple, H::AbstractFockHilbertSpace, fockmapper, phase_factors=use_wedge_phase_factors(Hs, H))
    fill!(mout, zero(eltype(mout)))
    jw = H.jw
    partition = map(collect ∘ keys, Hs) # using collect here turns out to be a bit faster
    isorderedpartition(partition, jw) || throw(ArgumentError("The partition must be ordered according to jw"))

    inds = Base.product(map(wedge_iterator, ms, Hs)...)
    for I in inds
        I1 = map(i -> i[1], I)
        I2 = map(i -> i[2], I)
        fock1 = map(indtofock, I1, Hs)
        fullfock1 = fockmapper(fock1)
        outind1 = focktoind(fullfock1, H)
        fock2 = map(indtofock, I2, Hs)
        fullfock2 = fockmapper(fock2)
        outind2 = focktoind(fullfock2, H)
        s = phase_factors ? phase_factor_h(fullfock1, fullfock2, partition, jw) : 1
        v = mapreduce((m, i1, i2) -> m[i1, i2], *, ms, I1, I2)
        mout[outind1, outind2] += v * s
    end
    return mout
end

function fermionic_kron_vec!(mout, ms::Tuple, Hs::Tuple, H::AbstractFockHilbertSpace, fockmapper)
    fill!(mout, zero(eltype(mout)))
    U = embedding_unitary(Hs, H)
    dimlengths = map(length ∘ focknumbers, Hs)
    inds = CartesianIndices(Tuple(dimlengths))
    for I in inds
        TI = Tuple(I)
        fock = map(indtofock, TI, Hs)
        fullfock = fockmapper(fock)
        outind = focktoind(fullfock, H)
        mout[outind] += mapreduce((i1, m) -> m[i1], *, TI, ms)
    end
    return U * mout
end

"""
    embedding(m, b, bnew)

Compute the fermionic embedding of a matrix `m` in the basis `b` into the basis `bnew`.
"""
function embedding(m, H::AbstractFockHilbertSpace, Hnew, phase_factors=isfermionic(H))
    # See eq. 20 in J. Phys. A: Math. Theor. 54 (2021) 393001
    isorderedsubsystem(H, Hnew) || throw(ArgumentError("Can't embed $H into $Hnew"))
    bbar_labs = setdiff(collect(keys(Hnew)), collect(keys(H))) # arrays to keep order
    bbar = SimpleFockHilbertSpace(bbar_labs)
    Hs = (H, bbar)
    return fermionic_kron((m, I), Hs, Hnew, phase_factors)
end
function extension(m, H::AbstractFockHilbertSpace, Hbar, phase_factors=isfermionic(H))
    isdisjoint(keys(H), keys(Hbar)) || throw(ArgumentError("The bases of the two Hilbert spaces must be disjoint"))
    Hs = (H, Hnew)
    Hout = wedge(Hs)
    return fermionic_kron((m, I), Hs, Hout, phase_factors)
end
embedding(Hs::Pair{<:AbstractFockHilbertSpace,<:AbstractFockHilbertSpace}, phase_factors=isfermionic(first(Hs))) = m -> embedding(m, first(Hs), last(Hs), phase_factors)

"""
    wedge(ms, bs, b)

Compute the ordered product of the fermionic embeddings of the matrices `ms` in the bases `bs` into the basis `b`.
"""
function wedge(ms, Hs, H)
    # See eq. 26 in J. Phys. A: Math. Theor. 54 (2021) 393001
    isorderedpartition(Hs, H) || throw(ArgumentError("The subsystems must be a partition consistent with the jordan-wigner ordering of the full system"))
    return mapreduce(((m, fine_basis),) -> embedding(m, fine_basis, H), *, zip(ms, Hs))
end
wedge(ms, HsH::Pair{<:Any,<:AbstractFockHilbertSpace}) = wedge(ms, first(HsH), last(HsH))
wedge(HsH::Pair{<:Any,<:AbstractFockHilbertSpace}) = (ms...) -> wedge(ms, first(HsH), last(HsH))

@testitem "Fermionic tensor product properties" begin
    # Properties from J. Phys. A: Math. Theor. 54 (2021) 393001
    # Eq. 16
    using Random, Base.Iterators, LinearAlgebra
    import QuantumDots: embedding, wedge, embedding_unitary, canonical_embedding

    Random.seed!(1)
    N = 7
    rough_size = 5
    fine_size = 3
    rough_partitions = sort.(collect(partition(randperm(N), rough_size)))
    # divide each part of rough partition into finer partitions
    fine_partitions = map(rough_partition -> sort.(collect(partition(shuffle(rough_partition), fine_size))), rough_partitions)
    H = hilbert_space(1:N)
    c = fermions(H)
    Hs_rough = [hilbert_space(r_p) for r_p in rough_partitions]
    Hs_fine = map(f_p_list -> hilbert_space.(f_p_list), fine_partitions)

    ops_rough = map(r_p -> rand(ComplexF64, 2^length(r_p), 2^length(r_p)), rough_partitions)
    ops_fine = map(f_p_list -> [rand(ComplexF64, 2^length(f_p), 2^length(f_p)) for f_p in f_p_list], fine_partitions)

    # Associativity (Eq. 16)
    rhs = fermionic_kron(reduce(vcat, ops_fine), reduce(vcat, Hs_fine), H)
    finewedges = [fermionic_kron(ops_vec, cs_vec, c_rough) for (ops_vec, cs_vec, c_rough) in zip(ops_fine, Hs_fine, Hs_rough)]
    lhs = fermionic_kron(finewedges, Hs_rough, H)
    @test lhs ≈ rhs

    rhs = kron(reduce(vcat, ops_fine), reduce(vcat, Hs_fine), H)
    lhs = kron([kron(ops_vec, cs_vec, c_rough) for (ops_vec, cs_vec, c_rough) in zip(ops_fine, Hs_fine, Hs_rough)], Hs_rough, H)
    @test lhs ≈ rhs

    physical_ops_rough = [project_on_parity(op, H, 1) for (op, H) in zip(ops_rough, Hs_rough)]

    # Eq. 18
    As = ops_rough
    Bs = map(r_p -> rand(ComplexF64, 2^length(r_p), 2^length(r_p)), rough_partitions)
    lhs = tr(fermionic_kron(As, Hs_rough, H)' * fermionic_kron(Bs, Hs_rough, H))
    rhs = mapreduce((A, B) -> tr(A' * B), *, As, Bs)
    @test lhs ≈ rhs

    # Fermionic embedding

    # Eq. 19 
    As_modes = [rand(ComplexF64, 2, 2) for _ in 1:N]
    ξ = vcat(fine_partitions...)
    ξbases = vcat(Hs_fine...)
    modebases = [hilbert_space(j:j) for j in 1:N]
    lhs = prod(j -> embedding(As_modes[j], modebases[j], H), 1:N)
    rhs_ordered_prod(X, basis) = mapreduce(j -> embedding(As_modes[j], modebases[j], basis), *, X)
    rhs = fermionic_kron([rhs_ordered_prod(X, b) for (X, b) in zip(ξ, ξbases)], ξbases, H)
    @test lhs ≈ rhs

    # Associativity (Eq. 21)
    @test embedding(embedding(ops_fine[1][1], Hs_fine[1][1], Hs_rough[1]), Hs_rough[1], H) ≈ embedding(ops_fine[1][1], Hs_fine[1][1], H)
    @test all(map(Hs_rough, Hs_fine, ops_fine) do cr, cfs, ofs
        all(map(cfs, ofs) do cf, of
            embedding(embedding(of, cf, cr), cr, H) ≈ embedding(of, cf, H)
        end)
    end)

    ## Eq. 22
    HX = Hs_rough[1]
    Ux = embedding_unitary(rough_partitions, H)
    A = ops_rough[1]
    @test Ux !== I
    @test embedding(A, HX, H) ≈ Ux * canonical_embedding(A, HX, H) * Ux'
    # Eq. 93
    @test wedge(physical_ops_rough, Hs_rough, H) ≈ Ux * kron(physical_ops_rough, Hs_rough, H) * Ux'

    # Eq. 23
    X = rough_partitions[1]
    HX = Hs_rough[1]
    A = ops_rough[1]
    B = rand(ComplexF64, 2^length(X), 2^length(X))
    #Eq 5a and 5br are satisfied also when embedding matrices in larger subsystems
    @test embedding(A, HX, H)' ≈ embedding(A', HX, H)
    @test canonical_embedding(A, HX, H) * canonical_embedding(B, HX, H) ≈ canonical_embedding(A * B, HX, H)
    for cmode in modebases
        #Eq 5bl
        local A = rand(ComplexF64, 2, 2)
        local B = rand(ComplexF64, 2, 2)
        @test embedding(A, cmode, H) * embedding(B, cmode, H) ≈ embedding(A * B, cmode, H)
    end

    # Ordered product of embeddings

    # Eq. 31
    A = ops_rough[1]
    X = rough_partitions[1]
    Xbar = setdiff(1:N, X)
    HX = Hs_rough[1]
    HXbar = hilbert_space(Xbar)
    corr = embedding(A, HX, H)
    @test corr ≈ fermionic_kron([A, I], [HX, HXbar], H) ≈ wedge([A, I], [HX, HXbar], H) ≈ wedge([I, A], [HXbar, HX], H)

    # Eq. 32
    @test wedge(As_modes, modebases, H) ≈ fermionic_kron(As_modes, modebases, H)

    ## Fermionic partial trace

    # Eq. 36
    X = rough_partitions[1]
    A = ops_rough[1]
    B = rand(ComplexF64, 2^N, 2^N)
    HX = Hs_rough[1]
    lhs = tr(embedding(A, HX, H)' * B)
    rhs = tr(A' * partial_trace(B, H, HX))
    @test lhs ≈ rhs

    # Eq. 38 (using A, X, HX, HXbar from above)
    B = rand(ComplexF64, 2^length(Xbar), 2^length(Xbar))
    Hs = [HX, HXbar]
    ops = [A, B]
    @test partial_trace(fermionic_kron(ops, Hs, H), H, HX) ≈ partial_trace(wedge(ops, Hs, H), H, HX) ≈ partial_trace(wedge(reverse(ops), reverse(Hs), H), H, HX) ≈ A * tr(B)

    # Eq. 39
    A = rand(ComplexF64, 2^N, 2^N)
    X = fine_partitions[1][1]
    Y = rough_partitions[1]
    HX = Hs_fine[1][1]
    HY = Hs_rough[1]
    HZ = H
    Z = 1:N
    rhs = partial_trace(A, HZ, HX)
    lhs = partial_trace(partial_trace(A, HZ, HY), HY, HX)
    @test lhs ≈ rhs

    # Eq. 41
    HY = H
    @test partial_trace(A', HY, HX) ≈ partial_trace(A, HY, HX)'

    # Eq. 95
    ξ = rough_partitions
    Asphys = physical_ops_rough
    Bs = map(X -> rand(ComplexF64, 2^length(X), 2^length(X)), ξ)
    Bsphys = [project_on_parity(B, H, 1) for (B, H) in zip(Bs, Hs_rough)]
    lhs1 = wedge(Asphys, Hs_rough, H) * wedge(Bsphys, Hs_rough, H)
    rhs1 = wedge(Asphys .* Bsphys, Hs_rough, H)
    @test lhs1 ≈ rhs1
    @test wedge(Asphys, Hs_rough, H)' ≈ wedge(adjoint.(Asphys), Hs_rough, H)

    ## Unitary equivalence between wedge and kron
    ops = reduce(vcat, ops_fine)
    Hs = reduce(vcat, Hs_fine)
    physical_ops = [project_on_parity(op, H, 1) for (op, H) in zip(ops, Hs)]
    # Eq. 93 implies that the unitary equivalence holds for the physical operators
    @test svdvals(Matrix(wedge(physical_ops, Hs, H))) ≈ svdvals(Matrix(kron(physical_ops, Hs, H)))
    # However, it is more general. The unitary equivalence holds as long as all except at most one of the operators has a definite parity:

    numberops = map(numberoperator, Hs)
    Uemb = embedding_unitary(Hs, H)
    fine_partition = reduce(vcat, fine_partitions)
    for parities in Base.product([[-1, 1] for _ in 1:length(Hs)]...)
        projected_ops = [project_on_parity(op, H, p) for (op, H, p) in zip(ops, Hs, parities)] # project on local parity
        opsk = [[projected_ops[1:k-1]..., ops[k], projected_ops[k+1:end]...] for k in 1:length(ops)] # switch out one operator of definite parity for an operator of indefinite parity
        embedding_prods = [wedge(ops, Hs, H) for ops in opsk]
        kron_prods = [kron(ops, Hs, H) for ops in opsk]

        @test all(svdvals(Matrix(op1)) ≈ svdvals(Matrix(op2)) for (op1, op2) in zip(embedding_prods, kron_prods))
    end

    # Explicit construction of unitary equivalence in case of all even (except one) 
    function phase(k, f)
        Xkmask = QuantumDots.focknbr_from_site_labels(fine_partition[k], H.jw)
        iseven(count_ones(f & Xkmask)) && return 1
        phase = 1
        for r in 1:k-1
            Xrmask = QuantumDots.focknbr_from_site_labels(fine_partition[r], H.jw)
            phase *= (-1)^(count_ones(f & Xrmask))
        end
        return phase
    end
    opsk = [[physical_ops[1:k-1]..., ops[k], physical_ops[k+1:end]...] for k in 1:length(ops)]
    unitaries = [Diagonal([phase(k, f) for f in focknumbers(H)]) * Uemb for k in 1:length(opsk)]
    embedding_prods = [wedge(ops, Hs, H) for ops in opsk]
    kron_prods = [kron(ops, Hs, H) for ops in opsk]
    @test all(op1 ≈ U * op2 * U for (op1, op2, U) in zip(embedding_prods, kron_prods, unitaries))

end


@testitem "Wedge" begin
    using Random, LinearAlgebra, BlockDiagonals
    import SparseArrays: SparseMatrixCSC
    Random.seed!(1234)

    for qn in [NoSymmetry(), ParityConservation(), FermionConservation()]
        H1 = hilbert_space(1:1, qn)
        H2 = hilbert_space(1:3, qn)
        @test_throws ArgumentError wedge(H1, H2)
        H2 = hilbert_space(2:3, qn)
        H3 = hilbert_space(1:3, qn)
        H3w = wedge(H1, H2)
        @test H3w == wedge((H1, H2)) == wedge([H1, H2])
        Hs = [H1, H2]
        b1 = fermions(H1)
        b2 = fermions(H2)
        b3 = fermions(H3w)

        #test that they keep sparsity
        @test typeof(wedge((b1[1], b2[2]), Hs => H3)) == typeof(b1[1])
        @test typeof(kron((b1[1], b2[2]), Hs => H3)) == typeof(b1[1])
        @test typeof(wedge((b1[1], I), Hs => H3)) == typeof(b1[1])
        @test typeof(kron((b1[1], I), Hs => H3)) == typeof(b1[1])
        @test wedge((I, I), Hs => H3) isa SparseMatrixCSC
        @test kron((I, I), Hs => H3) isa SparseMatrixCSC

        O1 = isodd.(numberoperator(H1))
        O2 = isodd.(numberoperator(H2))
        for P1 in [O1, I - O1], P2 in [O2, I - O2] #Loop over different parity sectors because of superselection. Otherwise, minus signs come into play
            v1 = P1 * rand(2)
            v2 = P2 * rand(4)
            v3 = fermionic_kron([v1, v2], Hs => H3)
            for k1 in keys(b1), k2 in keys(b2)
                b1f = b1[k1]
                b2f = b2[k2]
                b3f = b3[k2] * b3[k1]
                b3fw = wedge([b1f, b2f], Hs => H3)
                v3w = fermionic_kron([b1f * v1, b2f * v2], Hs => H3)
                v3f = b3f * v3
                @test v3f == v3w || v3f == -v3w #Vectors are the same up to a sign
            end
        end

        # Test wedge of matrices
        P1 = parityoperator(H1)
        P2 = parityoperator(H2)
        P3 = parityoperator(H3)
        wedge([P1, P2], Hs => H3) ≈ P3


        rho1 = rand(2, 2)
        rho2 = rand(4, 4)
        rho3 = wedge([rho1, rho2], Hs => H3)
        for P1 in [P1 + I, I - P1], P2 in [P2 + I, I - P2] #Loop over different parity sectors because of superselection. Otherwise, minus signs come into play
            m1 = P1 * rho1 * P1
            m2 = P2 * rho2 * P2
            P3 = wedge([P1, P2], Hs => H3)
            m3 = P3 * rho3 * P3
            @test wedge([m1, m2], Hs => H3) == m3
        end

        h1 = Matrix(0.5b1[1]' * b1[1])
        h2 = Matrix(-0.1b2[2]' * b2[2] + 0.3b2[3]' * b2[3] + (b2[2]' * b2[3] + hc))
        vals1, vecs1 = eigen(h1)
        vals2, vecs2 = eigen(h2)
        h3 = Matrix(0.5b3[1]' * b3[1] - 0.1b3[2]' * b3[2] + 0.3b3[3]' * b3[3] + (b3[2]' * b3[3] + hc))
        vals3, vecs3 = eigen(h3)

        # test wedging with I (UniformScaling)
        H3w = wedge([h1, I], Hs => H3) + wedge([I, h2], Hs => H3)
        @test H3w == H3
        @test wedge([I, I], Hs => H3) == one(H3)

        vals3w = map(sum, Base.product(vals1, vals2)) |> vec
        p = sortperm(vals3w)
        vals3w[p] ≈ vals3

        vecs3w = vec(map(v12 -> fermionic_kron([v12[1], v12[2]], Hs => H3), Base.product(eachcol(vecs1), eachcol(vecs2))))[p]
        @test all(map((v3, v3w) -> abs(dot(v3, v3w)) ≈ norm(v3) * norm(v3w), eachcol(vecs3), vecs3w))

        β = 0.7
        rho1 = exp(-β * h1)
        rmul!(rho1, 1 / tr(rho1))
        rho2 = exp(-β * h2)
        rmul!(rho2, 1 / tr(rho2))
        rho3 = exp(-β * h3)
        rmul!(rho3, 1 / tr(rho3))
        rho3w = wedge([rho1, rho2], Hs => H3)
        @test rho3w ≈ rho3
        @test partial_trace(rho3, H3 => H1) ≈ rho1
        @test partial_trace(rho3, H3 => H2) ≈ rho2
        @test wedge([blockdiagonal(rho1, H1), blockdiagonal(rho2, H2)], Hs => H3) ≈ wedge([blockdiagonal(rho1, H1), rho2], Hs => H3)
        @test wedge([blockdiagonal(rho1, H1), blockdiagonal(rho2, H2)], Hs => H3) ≈ rho3

        # Test BD1_hamiltonian
        H1 = hilbert_space(Base.product(1:2, (:↑, :↓)), qn)
        H2 = hilbert_space(Base.product(3:4, (:↑, :↓)), qn)
        H12 = hilbert_space(Base.product(1:4, (:↑, :↓)), qn)
        H12w = wedge(H1, H2)
        Hs = [H1, H2]
        b1 = fermions(H1)
        b2 = fermions(H2)
        b12 = fermions(H12)
        θ1 = 0.5
        θ2 = 0.2
        params1 = (; μ=1, t=0.5, Δ=2.0, V=0, θ=1:4 .* θ1, ϕ=1.0, h=4.0, U=2.0, Δ1=0.1)
        params2 = (; μ=1, t=0.1, Δ=1.0, V=0, θ=1:4 .* θ2, ϕ=5.0, h=1.0, U=10.0, Δ1=-1.0)
        params12 = (; μ=[params1.μ, params1.μ, params2.μ, params2.μ], t=[params1.t, 0, params2.t, 0], Δ=[params1.Δ, params1.Δ, params2.Δ, params2.Δ], V=[params1.V, 0, params2.V, 0], θ=[0, θ1, 0, θ2], ϕ=[params1.ϕ, params1.ϕ, params2.ϕ, params2.ϕ], h=[params1.h, params1.h, params2.h, params2.h], U=[params1.U, params1.U, params2.U, params2.U], Δ1=[params1.Δ1, 0, params2.Δ1, 0])
        H1 = Matrix(QuantumDots.BD1_hamiltonian(b1; params1...))
        H2 = Matrix(QuantumDots.BD1_hamiltonian(b2; params2...))

        h12w = Matrix(wedge([h1, I], bs, b12w) + wedge([I, h2], bs, b12w))
        h12 = Matrix(QuantumDots.BD1_hamiltonian(b12; params12...))

        v12w = fermionic_kron([eigvecs(Matrix(h1))[:, 1], eigvecs(Matrix(H2))[:, 1]], Hs, H12w)
        v12 = eigvecs(h12)[:, 1]
        v12ww = eigvecs(h12w)[:, 1]
        sort(abs.(v12w)) - sort(abs.(v12))
        @test sum(abs, v12w) ≈ sum(abs, v12)
        @test sum(abs, v12w) ≈ sum(abs, v12ww)
        @test diff(eigvals(h12w)) ≈ diff(eigvals(h12))

        # Test zero-mode wedge
        H1 = hilbert_space(1:0, qn)
        H2 = hilbert_space(1:1, qn)
        c1 = fermions(H1)
        c2 = fermions(H2)
        @test wedge([I(1), I(1)], [H1, H1], H1) == I(1)
        @test wedge([I(1), c2[1]], [H1, H2], H2) == c2[1]
    end

    #Test basis compatibility
    H1 = hilbert_space(1:2, ParityConservation())
    H2 = hilbert_space(2:4, ParityConservation())
    @test_throws ArgumentError wedge(H1, H2)
end


struct PhaseMap
    phases::Matrix{Int}
    fockstates::Vector{FockNumber}
end
struct LazyPhaseMap{M} <: AbstractMatrix{Int}
    fockstates::Vector{FockNumber}
end
Base.length(p::LazyPhaseMap) = length(p.fockstates)
Base.ndims(::LazyPhaseMap) = 2
function Base.size(p::LazyPhaseMap, d::Int)
    d < 1 && error("arraysize: dimension out of range")
    d in (1, 2) ? length(p.fockstates) : 1
end
Base.size(p::LazyPhaseMap{M}) where {M} = (length(p.fockstates), length(p.fockstates))
function Base.show(io::IO, p::LazyPhaseMap{M}) where {M}
    print(io, "LazyPhaseMap{$M}(")
    show(io, p.fockstates)
    print(")")
end
Base.show(io::IO, ::MIME"text/plain", p::LazyPhaseMap) = show(io, p)
Base.getindex(p::LazyPhaseMap{M}, n1::Int, n2::Int) where {M} = phase_factor_f(p.fockstates[n1], p.fockstates[n2], M)
function phase_map(fockstates, M::Int)
    phases = zeros(Int, length(fockstates), length(fockstates))
    for (n1, f1) in enumerate(fockstates)
        for (n2, f2) in enumerate(fockstates)
            phases[n1, n2] = phase_factor_f(f1, f2, M)
        end
    end
    PhaseMap(phases, fockstates)
end
phase_map(N::Int) = phase_map(map(FockNumber, 0:2^N-1), N)
phase_map(b::AbstractFockHilbertSpace) = phase_map(collect(focknumbers(b)), length(b.jw))
LazyPhaseMap(N::Int) = LazyPhaseMap{N}(map(FockNumber, 0:2^N-1))
SparseArrays.HigherOrderFns.is_supported_sparse_broadcast(::LazyPhaseMap, rest...) = SparseArrays.HigherOrderFns.is_supported_sparse_broadcast(rest...)
(p::PhaseMap)(op::AbstractMatrix) = p.phases .* op
(p::LazyPhaseMap)(op::AbstractMatrix) = p .* op
@testitem "phasemap" begin
    using LinearAlgebra
    # see App 2 in https://arxiv.org/pdf/2006.03087
    ns = 1:4
    phis = Dict(zip(ns, QuantumDots.phase_map.(ns)))
    lazyphis = Dict(zip(ns, QuantumDots.LazyPhaseMap.(ns)))
    @test all(sum(phis[n].phases .== -1) == (2^n - 2) * 2^n / 2 for n in ns)
    @test all(sum(phis[n].phases .== -1) == (2^n - 2) * 2^n / 2 for n in ns)

    for N in ns
        H = SimpleFockHilbertSpace(1:N)
        c = fermions(H)
        # q = QubitBasis(1:N)
        q = QubitOperators(H)
        @test all(map(n -> q[n] == phis[N](c[n]), 1:N))
        c2 = map(n -> phis[N](c[n]), 1:N)
        @test phis[N](phis[N](c[1])) == c[1]
        # c is fermionic
        @test all([c[n] * c[n2] == -c[n2] * c[n] for n in 1:N, n2 in 1:N])
        @test all([c[n]' * c[n2] == -c[n2] * c[n]' + I * (n == n2) for n in 1:N, n2 in 1:N])
        # c2 is hardcore bosons
        @test all([c2[n] * c2[n2] == c2[n2] * c2[n] for n in 1:N, n2 in 1:N])
        @test all([c2[n]' * c2[n2] == (-c2[n2] * c2[n]' + I) * (n == n2) + (n !== n2) * (c2[n2] * c2[n]') for n in 1:N, n2 in 1:N])
    end

    H1 = hilbert_space(1:1)
    c1 = fermions(H1)
    H2 = hilbert_space(2:2)
    c2 = fermions(H2)
    H12 = hilbert_space(1:2)
    c12 = fermions(H2)
    p1 = QuantumDots.LazyPhaseMap(1)
    p2 = QuantumDots.phase_map(2)
    @test QuantumDots.fermionic_tensor_product_with_kron_and_maps((c1[1], I(2)), (p1, p1), p2) == c12[1]
    @test QuantumDots.fermionic_tensor_product_with_kron_and_maps((I(2), c2[2]), (p1, p1), p2) == c12[2]

    ms = (rand(2, 2), rand(2, 2))
    @test QuantumDots.fermionic_tensor_product_with_kron_and_maps(ms, (p1, p1), p2) == fermionic_kron(ms, (H1, H2), H12)
end

function fermionic_tensor_product_with_kron_and_maps(ops, phis, phi)
    phi(kron(reverse(map((phi, op) -> phi(op), phis, ops))...))
end

## kron, i.e. wedge without phase factors
Base.kron(ms, bs, b::AbstractHilbertSpace; kwargs...) = fermionic_kron(ms, bs, b, false; kwargs...)

canonical_embedding(m, b, bnew) = embedding(m, b, bnew, false)


"""
    partial_trace(m::AbstractMatrix,  bHfull::AbstractHilbertSpace, Hsub::AbstractHilbertSpace)

Compute the partial trace of a matrix `m`, leaving the subsystem defined by the basis `bsub`.
"""
function partial_trace(m::AbstractMatrix{T}, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, phase_factors=use_partial_trace_phase_factors(H, Hsub)) where {T}
    mout = zeros(T, size(Hsub))
    partial_trace!(mout, m, H, Hsub, phase_factors)
end

use_partial_trace_phase_factors(H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace) = use_wedge_phase_factors((H,), Hsub)

partial_trace(Hs::Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}, phase_factors=use_partial_trace_phase_factors(first(Hs), last(Hs))) = m -> partial_trace(m, first(Hs), last(Hs), phase_factors)
partial_trace(m, Hs::Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}, phase_factors=use_partial_trace_phase_factors(first(Hs), last(Hs))) = partial_trace(m, first(Hs), last(Hs), phase_factors)

"""
    partial_trace!(mout, m::AbstractMatrix, H::AbstractHilbertSpace, Hout::AbstractHilbertSpace, phase_factors)

Compute the fermionic partial trace of a matrix `m` in basis `H`, leaving only the subsystems specified by `labels`. The result is stored in `mout`, and `Hout` determines the ordering of the basis states.
"""
function partial_trace!(mout, m::AbstractMatrix, H::AbstractHilbertSpace, Hout::AbstractHilbertSpace, phase_factors=use_partial_trace_phase_factors(H, Hout))
    M = length(H.jw)
    labels = collect(keys(Hout))
    if phase_factors
        consistent_ordering(labels, H.jw) || throw(ArgumentError("Subsystem must be ordered in the same way as the full system"))
    end
    N = length(labels)
    fill!(mout, zero(eltype(mout)))
    outinds = siteindices(labels, H.jw)
    bitmask = FockNumber(2^M - 1) - focknbr_from_site_indices(outinds)
    outbits(f) = map(i -> _bit(f, i), outinds)
    fockstates = focknumbers(H)
    for f1 in fockstates, f2 in fockstates
        if (f1 & bitmask) != (f2 & bitmask)
            continue
        end
        newfocknbr1 = focknbr_from_bits(outbits(f1))
        newfocknbr2 = focknbr_from_bits(outbits(f2))
        s1 = phase_factors ? phase_factor_f(f1, f2, M) : 1
        s2 = phase_factors ? phase_factor_f(newfocknbr1, newfocknbr2, N) : 1
        s = s2 * s1
        mout[focktoind(newfocknbr1, Hout), focktoind(newfocknbr2, Hout)] += s * m[focktoind(f1, H), focktoind(f2, H)]
    end
    return mout
end

"""
    partial_transpose(m::AbstractMatrix, b::AbstractHilbertSpace, labels)

Compute the fermionic partial transpose of a matrix `m` in subsystem denoted by `labels`.
"""
function partial_transpose(m::AbstractMatrix, b::AbstractHilbertSpace, labels, phase_factors=use_partial_transpose_phase_factors(b))
    mout = zero(m)
    partial_transpose!(mout, m, b, labels, phase_factors)
end
function partial_transpose!(mout, m::AbstractMatrix, b::AbstractHilbertSpace, labels, phase_factors=use_partial_transpose_phase_factors(b))
    @warn "partial_transpose may not be physically meaningful" maxlog = 10
    M = nbr_of_modes(b)
    fill!(mout, zero(eltype(mout)))
    outinds = siteindices(labels, b)
    bitmask = FockNumber(2^M - 1) - focknbr_from_site_indices(outinds)
    outbits(f) = map(i -> _bit(f, i), outinds)
    fockstates = focknumbers(b)
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
use_partial_transpose_phase_factors(H::AbstractHilbertSpace) = isfermionic(H)

@testitem "Partial transpose" begin
    using LinearAlgebra
    import QuantumDots: partial_transpose
    qn = ParityConservation()
    H1 = hilbert_space(1:1, qn)
    H2 = hilbert_space(2:2, qn)
    H12 = hilbert_space(1:2, qn)
    c1 = fermions(H1)
    c2 = fermions(H2)
    c12 = fermions(H12)
    A = rand(ComplexF64, 2, 2)
    B = rand(ComplexF64, 2, 2)
    C = fermionic_kron((A, B), (c1, c2), c12)
    Cpt = partial_transpose(C, c12, (1,))
    Cpt2 = fermionic_kron((transpose(A), B), (c1, c2), c12)
    @test Cpt ≈ Cpt2

    ## Larger system
    labels = 1:4
    N = length(labels)
    HN = hilbert_space(labels, qn)
    cN = fermions(HN)
    Hs = [hilbert_space(i:i, qn) for i in labels]
    cs = map(fermions, Hs)
    Ms = [rand(ComplexF64, 2, 2) for _ in labels]
    M = fermionic_kron(Ms, Hs, HN)

    single_subsystems = [(i,) for i in 1:4]
    for (k,) in single_subsystems
        Mpt = partial_transpose(M, HN, (k,))
        Mpt2 = fermionic_kron([(n == k) ? transpose(M) : M for (n, M) in enumerate(Ms)], Hs, HN)
        @test Mpt ≈ Mpt2
    end
    pair_iterator = [(i, j) for i in 1:4, j in 1:4 if i != j]
    triple_iterator = [(i, j, k) for i in 1:4, j in 1:4, k in 1:4 if length(unique((i, j, k))) == 3]
    for (i, j) in pair_iterator
        Mpt = partial_transpose(M, cN, (i, j))
        Mpt2 = fermionic_kron([(n == i || n == j) ? transpose(M) : M for (n, M) in enumerate(Ms)], Hs, HN)
        @test Mpt ≈ Mpt2
    end
    for (i, j, k) in triple_iterator
        Mpt = partial_transpose(M, cN, (i, j, k))
        Mpt2 = fermionic_kron([(n == i || n == j || n == k) ? transpose(M) : M for (n, M) in enumerate(Ms)], Hs, Hn)
        @test Mpt ≈ Mpt2
    end
    Mpt = partial_transpose(M, cN, labels)
    Mpt2 = fermionic_kron([transpose(M) for M in Ms], Hs, cN)
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

FockSplitter(H::AbstractHilbertSpace, bs) = FockSplitter(H.jw, map(b -> b.jw, bs))
FockMapper(Hs, H::AbstractHilbertSpace) = FockMapper(map(b -> b.jw, Hs), H.jw)
use_reshape_phase_factors(H::AbstractHilbertSpace, Hs) = use_wedge_phase_factors(Hs, H)
use_reshape_phase_factors(Hs, H::AbstractHilbertSpace) = use_wedge_phase_factors(Hs, H)

function project_on_parities(op::AbstractMatrix, b, bs, parities)
    length(bs) == length(parities) || throw(ArgumentError("The number of parities must match the number of subsystems"))
    for (bsub, parity) in zip(bs, parities)
        op = project_on_subparity(op, b, bsub, parity)
    end
    return op
end

function project_on_subparity(op::AbstractMatrix, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, parity)
    P = embedding(parityoperator(Hsub), Hsub, H)
    return project_on_parity(op, P, parity)
end

project_on_parity(op::AbstractMatrix, H::AbstractHilbertSpace, parity) = project_on_parity(op, parityoperator(H), parity)

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
    bs = [SimpleFockHilbertSpace(2k-1:2k) for k in 1:3]
    b = wedge(bs)
    op = rand(ComplexF64, size(b))
    local_parity_iter = (1, -1)
    all_parities = Base.product([local_parity_iter for _ in 1:length(bs)]...)
    @test sum(project_on_parities(op, b, bs, parities) for parities in all_parities) ≈ op

    ops = [rand(ComplexF64, size(b)) for b in bs]
    for parities in all_parities
        projected_ops = [project_on_parity(op, bsub, parity) for (op, bsub, parity) in zip(ops, bs, parities)]
        op = wedge(projected_ops, bs, b)
        @test op ≈ project_on_parities(op, b, bs, parities)
    end
end


@testitem "Embedding unitary action" begin
    # Appendix C.4
    import QuantumDots: embedding_unitary, canonical_embedding, bipartite_embedding_unitary
    using LinearAlgebra
    HA = hilbert_space((1, 3))
    HB = hilbert_space((2, 4))
    cA = fermions(HA)
    cB = fermions(HB)
    H = hilbert_space(1:4)
    c = fermions(H)
    Hs = (HA, HB)
    @test embedding_unitary(Hs, H) == embedding_unitary([[1, 3], [2, 4]], H)
    @test embedding(cA[1], HA, H) ≈ fermionic_kron((cA[1], I), Hs, H) ≈ fermionic_kron((I, cA[1]), (HB, HA), H)
    Ux = embedding_unitary(Hs, H)
    Ux2 = bipartite_embedding_unitary(HA, HB, H)
    @test Ux ≈ Ux2
    @test embedding(cA[1], HA, H) ≈ Ux * canonical_embedding(cA[1], HA, H) * Ux'
end
