"""
    wedge(bs)

Compute the wedge product of a list of `FermionBasis` objects. The symmetry of the resulting basis is computed by promote_symmetry.
"""
wedge(bs::AbstractVector{<:AbstractHilbertSpace}) = foldl(wedge, bs)
wedge(bs::Tuple) = foldl(wedge, bs)
# wedge(b1::B, bs::Vararg) where {N,B<:FermionBasis} = foldl(wedge, bs, init=b1)
function wedge(b1::FermionBasis, b2::FermionBasis)
    newlabels = vcat(collect(keys(b1)), collect(keys(b2)))
    if length(unique(newlabels)) != length(newlabels)
        throw(ArgumentError("The labels of the two bases are not disjoint"))
    end
    qn = promote_symmetry(b1.symmetry, b2.symmetry)
    FermionBasis(newlabels; qn)
end

promote_symmetry(s1::FockSymmetry{<:Any,<:Any,<:Any,F}, s2::FockSymmetry{<:Any,<:Any,<:Any,F}) where {F} = s1.conserved_quantity
promote_symmetry(::FockSymmetry{<:Any,<:Any,<:Any,F1}, ::FockSymmetry{<:Any,<:Any,<:Any,F2}) where {F1,F2} = NoSymmetry()
promote_symmetry(::NoSymmetry, ::S) where {S} = NoSymmetry()
promote_symmetry(::S, ::NoSymmetry) where {S} = NoSymmetry()
promote_symmetry(::NoSymmetry, ::NoSymmetry) = NoSymmetry()

function check_wedge_basis_compatibility(b1::FermionBasis{M1}, b2::FermionBasis{M2}, b3::FermionBasis{M3}) where {M1,M2,M3}
    if M1 + M2 != M3
        throw(ArgumentError("The combined basis does not have the correct number of sites"))
    end
    if vcat(collect(keys(b1)), collect(keys(b2))) != collect(keys(b3))
        throw(ArgumentError("The labels of the output basis are not the same (or ordered the same) as the labels of the input bases. $(keys(b1)) * $(keys(b2)) != $(keys(b3))"))
    end
end

get_fockstates(::FermionBasis{M,<:Any,NoSymmetry}) where {M} = Iterators.map(FockNumber, 0:2^M-1)
get_fockstates(b::FermionBasis) = get_fockstates(b.symmetry)
get_fockstates(sym::FockSymmetry) = sym.indtofockdict
get_fockstates(b::FermionBasisTemplate{<:Any,S}) where {S<:AbstractSymmetry} = get_fockstates(b.sym)
get_fockstates(b::FermionBasisTemplate{<:Any,NoSymmetry}) = Iterators.map(FockNumber, 0:2^length(keys(b))-1)
"""
    fermionic_kron(ms::AbstractVector, bs::AbstractVector{<:FermionBasis}, b::FermionBasis=wedge(bs))

Compute the fermionic tensor product of matrices or vectors in `ms` with respect to the fermion bases `bs`, respectively. Return a matrix in the fermion basis `b`, which defaults to the wedge product of `bs`.
"""
function fermionic_kron(ms, bs, b::FermionBasis=fermionic_kron(bs), phase_factors=use_wedge_phase_factors(bs, b); match_labels=true)
    N = ndims(first(ms))
    mout = allocate_wedge_result(ms, bs)

    fockmapper = if match_labels
        fermionpositions = map(Base.Fix2(siteindices, b.jw) ∘ collect ∘ keys, bs)
        FockMapper(fermionpositions)
    else
        Ms = map(nbr_of_modes, bs)
        shifts = (0, cumsum(Ms)...)
        FockShifter(shifts)
    end

    if N == 1
        return fermionic_kron_vec!(mout, Tuple(ms), Tuple(bs), b, fockmapper)
    elseif N == 2
        return fermionic_kron_mat!(mout, Tuple(ms), Tuple(bs), b, fockmapper, phase_factors)
    end
    throw(ArgumentError("Only 1D or 2D arrays are supported"))
end
uniform_to_sparse_type(::Type{UniformScaling{T}}) where {T} = SparseMatrixCSC{T,Int}
uniform_to_sparse_type(::Type{T}) where {T} = T
function allocate_wedge_result(ms, bs)
    T = Base.promote_eltype(ms...)
    N = ndims(first(ms))
    types = map(uniform_to_sparse_type ∘ typeof, ms)
    MT = Base.promote_op(kron, types...)
    dimlengths = map(length ∘ get_fockstates, bs)
    Nout = prod(dimlengths)
    _mout = Zeros(T, ntuple(j -> Nout, N))
    try
        convert(MT, _mout)
    catch
        Array(_mout)
    end
end

wedge_iterator(m, ::FermionBasis) = findall(!iszero, m)
wedge_iterator(::UniformScaling, b::FermionBasis) = diagind(I(length(get_fockstates(b))), IndexCartesian())
wedge_iterator(::UniformScaling, b::FermionBasisTemplate) = diagind(I(length(get_fockstates(b))), IndexCartesian())

use_wedge_phase_factors(bs, b::FermionBasis) = true

function fermionic_kron_mat!(mout, ms::Tuple, bs::Tuple, b::FermionBasis, fockmapper, phase_factors=use_wedge_phase_factors(bs, b))
    fill!(mout, zero(eltype(mout)))
    jw = b.jw
    partition = map(collect ∘ keys, bs) # using collect here turns out to be a bit faster
    isorderedpartition(partition, jw) || throw(ArgumentError("The partition must be ordered according to jw"))

    inds = Base.product(map(wedge_iterator, ms, bs)...)
    for I in inds
        I1 = map(i -> i[1], I)
        I2 = map(i -> i[2], I)
        fock1 = map(indtofock, I1, bs)
        fullfock1 = fockmapper(fock1)
        outind1 = focktoind(fullfock1, b)
        fock2 = map(indtofock, I2, bs)
        fullfock2 = fockmapper(fock2)
        outind2 = focktoind(fullfock2, b)
        s = phase_factors ? phase_factor_h(fullfock1, fullfock2, partition, jw) : 1
        v = mapreduce((m, i1, i2) -> m[i1, i2], *, ms, I1, I2)
        mout[outind1, outind2] += v * s
    end
    return mout
end

function fermionic_kron_vec!(mout, ms::Tuple, bs::Tuple, b::FermionBasis, fockmapper)
    fill!(mout, zero(eltype(mout)))
    U = embedding_unitary(bs, b)
    dimlengths = map(length ∘ get_fockstates, bs)
    inds = CartesianIndices(Tuple(dimlengths))
    for I in inds
        TI = Tuple(I)
        fock = map(indtofock, TI, bs)
        fullfock = fockmapper(fock)
        outind = focktoind(fullfock, b)
        mout[outind] += mapreduce((i1, m) -> m[i1], *, TI, ms)
    end
    return U * mout
end

embedding_unitary(partition, c::FermionBasis) = embedding_unitary(partition, get_fockstates(c), c.jw)
embedding_unitary(cs::Union{<:AbstractVector{B},<:NTuple{N,B}}, c::FermionBasis) where {B<:FermionBasis,N} = embedding_unitary(map(keys, cs), c)

bipartite_embedding_unitary(X, Xbar, c::FermionBasis) = bipartite_embedding_unitary(X, Xbar, get_fockstates(c), c.jw)
bipartite_embedding_unitary(X::FermionBasis, Xbar::FermionBasis, c::FermionBasis) = bipartite_embedding_unitary(keys(X), keys(Xbar), get_fockstates(c), c.jw)


"""
    fermionic_embedding(m, b, bnew)

Compute the fermionic embedding of a matrix `m` in the basis `b` into the basis `bnew`.
"""
function fermionic_embedding(m, b, bnew, phase_factors=true)
    # See eq. 20 in J. Phys. A: Math. Theor. 54 (2021) 393001
    bbar_labs = setdiff(collect(keys(bnew)), collect(keys(b))) # arrays to keep order
    qn = NoSymmetry()
    bbar = FermionBasisTemplate(JordanWignerOrdering(bbar_labs), qn)
    bs = (b, bbar)
    return fermionic_kron((m, I), bs, bnew, phase_factors)
end

"""
    wedge(ms, bs, b)

Compute the ordered product of the fermionic embeddings of the matrices `ms` in the bases `bs` into the basis `b`.
"""
function wedge(ms, bs, b)
    # See eq. 26 in J. Phys. A: Math. Theor. 54 (2021) 393001
    isorderedpartition(bs, b) || throw(ArgumentError("The subsystems must be a partition consistent with the jordan-wigner ordering of the full system"))
    return mapreduce(((m, fine_basis),) -> fermionic_embedding(m, fine_basis, b), *, zip(ms, bs))
end


@testitem "Fermionic tensor product properties" begin
    # Properties from J. Phys. A: Math. Theor. 54 (2021) 393001
    # Eq. 16
    using Random, Base.Iterators, LinearAlgebra
    import QuantumDots: fermionic_embedding, wedge, embedding_unitary, canonical_embedding

    Random.seed!(1)
    N = 7
    rough_size = 5
    fine_size = 3
    rough_partitions = sort.(collect(partition(randperm(N), rough_size)))
    # divide each part of rough partition into finer partitions
    fine_partitions = map(rough_partition -> sort.(collect(partition(shuffle(rough_partition), fine_size))), rough_partitions)
    c = FermionBasis(1:N)
    cs_rough = [FermionBasis(r_p) for r_p in rough_partitions]
    cs_fine = map(f_p_list -> FermionBasis.(f_p_list), fine_partitions)

    ops_rough = map(r_p -> rand(ComplexF64, 2^length(r_p), 2^length(r_p)), rough_partitions)
    ops_fine = map(f_p_list -> [rand(ComplexF64, 2^length(f_p), 2^length(f_p)) for f_p in f_p_list], fine_partitions)

    # Associativity (Eq. 16)
    rhs = fermionic_kron(reduce(vcat, ops_fine), reduce(vcat, cs_fine), c)
    finewedges = [fermionic_kron(ops_vec, cs_vec, c_rough) for (ops_vec, cs_vec, c_rough) in zip(ops_fine, cs_fine, cs_rough)]
    lhs = fermionic_kron(finewedges, cs_rough, c)
    @test lhs ≈ rhs

    rhs = kron(reduce(vcat, ops_fine), reduce(vcat, cs_fine), c)
    lhs = kron([kron(ops_vec, cs_vec, c_rough) for (ops_vec, cs_vec, c_rough) in zip(ops_fine, cs_fine, cs_rough)], cs_rough, c)
    @test lhs ≈ rhs

    physical_ops_rough = [project_on_parity(op, c, 1) for (op, c) in zip(ops_rough, cs_rough)]

    # Eq. 18
    As = ops_rough
    Bs = map(r_p -> rand(ComplexF64, 2^length(r_p), 2^length(r_p)), rough_partitions)
    lhs = tr(fermionic_kron(As, cs_rough, c)' * fermionic_kron(Bs, cs_rough, c))
    rhs = mapreduce((A, B) -> tr(A' * B), *, As, Bs)
    @test lhs ≈ rhs

    # Fermionic embedding

    # Eq. 19 
    As_modes = [rand(ComplexF64, 2, 2) for _ in 1:N]
    ξ = vcat(fine_partitions...)
    ξbases = vcat(cs_fine...)
    modebases = [FermionBasis(j:j) for j in 1:N]
    lhs = prod(j -> fermionic_embedding(As_modes[j], modebases[j], c), 1:N)
    rhs_ordered_prod(X, basis) = mapreduce(j -> fermionic_embedding(As_modes[j], modebases[j], basis), *, X)
    rhs = fermionic_kron([rhs_ordered_prod(X, b) for (X, b) in zip(ξ, ξbases)], ξbases, c)
    @test lhs ≈ rhs

    # Associativity (Eq. 21)
    @test fermionic_embedding(fermionic_embedding(ops_fine[1][1], cs_fine[1][1], cs_rough[1]), cs_rough[1], c) ≈ fermionic_embedding(ops_fine[1][1], cs_fine[1][1], c)
    @test all(map(cs_rough, cs_fine, ops_fine) do cr, cfs, ofs
        all(map(cfs, ofs) do cf, of
            fermionic_embedding(fermionic_embedding(of, cf, cr), cr, c) ≈ fermionic_embedding(of, cf, c)
        end)
    end)

    ## Eq. 22
    cX = cs_rough[1]
    Ux = embedding_unitary(rough_partitions, c)
    A = ops_rough[1]
    @test Ux !== I
    @test fermionic_embedding(A, cX, c) ≈ Ux * canonical_embedding(A, cX, c) * Ux'
    # Eq. 93
    @test wedge(physical_ops_rough, cs_rough, c) ≈ Ux * kron(physical_ops_rough, cs_rough, c) * Ux'

    # Eq. 23
    X = rough_partitions[1]
    cX = cs_rough[1]
    A = ops_rough[1]
    B = rand(ComplexF64, 2^length(X), 2^length(X))
    #Eq 5a and 5br are satisfied also when embedding matrices in larger subsystems
    @test fermionic_embedding(A, cX, c)' ≈ fermionic_embedding(A', cX, c)
    @test canonical_embedding(A, cX, c) * canonical_embedding(B, cX, c) ≈ canonical_embedding(A * B, cX, c)
    for cmode in modebases
        #Eq 5bl
        local A = rand(ComplexF64, 2, 2)
        local B = rand(ComplexF64, 2, 2)
        @test fermionic_embedding(A, cmode, c) * fermionic_embedding(B, cmode, c) ≈ fermionic_embedding(A * B, cmode, c)
    end

    # Ordered product of embeddings

    # Eq. 31
    A = ops_rough[1]
    X = rough_partitions[1]
    Xbar = setdiff(1:N, X)
    cX = cs_rough[1]
    cXbar = FermionBasis(Xbar)
    corr = fermionic_embedding(A, cX, c)
    @test corr ≈ fermionic_kron([A, I], [cX, cXbar], c) ≈ wedge([A, I], [cX, cXbar], c) ≈ wedge([I, A], [cXbar, cX], c)

    # Eq. 32
    @test wedge(As_modes, modebases, c) ≈ fermionic_kron(As_modes, modebases, c)

    ## Fermionic partial trace

    # Eq. 36
    X = rough_partitions[1]
    A = ops_rough[1]
    B = rand(ComplexF64, 2^N, 2^N)
    cX = cs_rough[1]
    lhs = tr(fermionic_embedding(A, cX, c)' * B)
    rhs = tr(A' * partial_trace(B, c, cX))
    @test lhs ≈ rhs

    # Eq. 38 (using A, X, cX, cXbar from above)
    B = rand(ComplexF64, 2^length(Xbar), 2^length(Xbar))
    cs = [cX, cXbar]
    ops = [A, B]
    @test partial_trace(fermionic_kron(ops, cs, c), c, cX) ≈ partial_trace(wedge(ops, cs, c), c, cX) ≈
          partial_trace(wedge(reverse(ops), reverse(cs), c), c, cX) ≈ A * tr(B)

    # Eq. 39
    A = rand(ComplexF64, 2^N, 2^N)
    X = fine_partitions[1][1]
    Y = rough_partitions[1]
    bX = cs_fine[1][1]
    bY = cs_rough[1]
    bZ = c
    Z = 1:N
    rhs = partial_trace(A, bZ, bX)
    lhs = partial_trace(partial_trace(A, bZ, bY), bY, bX)
    @test lhs ≈ rhs

    # Eq. 41
    bY = c
    @test partial_trace(A', bY, bX) ≈ partial_trace(A, bY, bX,)'

    # Eq. 95
    ξ = rough_partitions
    Asphys = physical_ops_rough
    Bs = map(X -> rand(ComplexF64, 2^length(X), 2^length(X)), ξ)
    Bsphys = [project_on_parity(B, c, 1) for (B, c) in zip(Bs, cs_rough)]
    lhs1 = wedge(Asphys, cs_rough, c) * wedge(Bsphys, cs_rough, c)
    rhs1 = wedge(Asphys .* Bsphys, cs_rough, c)
    @test lhs1 ≈ rhs1
    @test wedge(Asphys, cs_rough, c)' ≈ wedge(adjoint.(Asphys), cs_rough, c)

    ## Unitary equivalence between wedge and kron
    ops = reduce(vcat, ops_fine)
    cs = reduce(vcat, cs_fine)
    physical_ops = [project_on_parity(op, c, 1) for (op, c) in zip(ops, cs)]
    # Eq. 93 implies that the unitary equivalence holds for the physical operators
    @test svdvals(Matrix(wedge(physical_ops, cs, c))) ≈ svdvals(Matrix(kron(physical_ops, cs, c)))
    # However, it is more general. The unitary equivalence holds as long as all except at most one of the operators has a definite parity:

    numberops = map(numberoperator, cs)
    Uemb = embedding_unitary(cs, c)
    fine_partition = reduce(vcat, fine_partitions)
    for parities in Base.product([[-1, 1] for _ in 1:length(cs)]...)
        projected_ops = [project_on_parity(op, c, p) for (op, c, p) in zip(ops, cs, parities)] # project on local parity
        opsk = [[projected_ops[1:k-1]..., ops[k], projected_ops[k+1:end]...] for k in 1:length(ops)] # switch out one operator of definite parity for an operator of indefinite parity
        embedding_prods = [wedge(ops, cs, c) for ops in opsk]
        kron_prods = [kron(ops, cs, c) for ops in opsk]

        @test all(svdvals(Matrix(op1)) ≈ svdvals(Matrix(op2)) for (op1, op2) in zip(embedding_prods, kron_prods))
    end

    # Explicit construction of unitary equivalence in case of all even (except one) 
    function phase(k, f)
        Xkmask = QuantumDots.focknbr_from_site_labels(fine_partition[k], c.jw)
        iseven(count_ones(f & Xkmask)) && return 1
        phase = 1
        for r in 1:k-1
            Xrmask = QuantumDots.focknbr_from_site_labels(fine_partition[r], c.jw)
            phase *= (-1)^(count_ones(f & Xrmask))
        end
        return phase
    end
    opsk = [[physical_ops[1:k-1]..., ops[k], physical_ops[k+1:end]...] for k in 1:length(ops)]
    unitaries = [Diagonal([phase(k, f) for f in QuantumDots.get_fockstates(c)]) * Uemb for k in 1:length(opsk)]
    embedding_prods = [wedge(ops, cs, c) for ops in opsk]
    kron_prods = [kron(ops, cs, c) for ops in opsk]
    @test all(op1 ≈ U * op2 * U for (op1, op2, U) in zip(embedding_prods, kron_prods, unitaries))

end


@testitem "Wedge" begin
    using Random, LinearAlgebra
    import SparseArrays: SparseMatrixCSC
    Random.seed!(1234)

    for qn in [NoSymmetry(), ParityConservation(), FermionConservation()]
        b1 = FermionBasis(1:1; qn)
        b2 = FermionBasis(1:3; qn)
        @test_throws ArgumentError wedge(b1, b2)
        b2 = FermionBasis(2:3; qn)
        b3 = FermionBasis(1:3; qn)
        b3w = wedge(b1, b2)
        @test b3w == wedge((b1, b2)) == wedge([b1, b2])
        @test norm(map(-, b3w, b3)) == 0
        bs = [b1, b2]

        #test that they keep sparsity
        @test typeof(wedge((b1[1], b2[2]), bs, b3)) == typeof(b1[1])
        @test typeof(kron((b1[1], b2[2]), bs, b3)) == typeof(b1[1])
        @test typeof(wedge((b1[1], I), bs, b3)) == typeof(b1[1])
        @test typeof(kron((b1[1], I), bs, b3)) == typeof(b1[1])
        @test wedge((I, I), bs, b3) isa SparseMatrixCSC
        @test kron((I, I), bs, b3) isa SparseMatrixCSC

        O1 = isodd.(QuantumDots.numberoperator(b1))
        O2 = isodd.(QuantumDots.numberoperator(b2))
        for P1 in [O1, I - O1], P2 in [O2, I - O2] #Loop over different parity sectors because of superselection. Otherwise, minus signs come into play
            v1 = P1 * rand(2)
            v2 = P2 * rand(4)
            v3 = fermionic_kron([v1, v2], bs, b3)
            for k1 in keys(b1), k2 in keys(b2)
                b1f = b1[k1]
                b2f = b2[k2]
                b3f = b3[k2] * b3[k1]
                b3fw = wedge([b1f, b2f], bs, b3)
                v3w = fermionic_kron([b1f * v1, b2f * v2], bs, b3)
                v3f = b3f * v3
                @test v3f == v3w || v3f == -v3w #Vectors are the same up to a sign
            end
        end

        # Test wedge of matrices
        P1 = QuantumDots.parityoperator(b1)
        P2 = QuantumDots.parityoperator(b2)
        P3 = QuantumDots.parityoperator(b3)
        wedge([P1, P2], bs, b3) ≈ P3


        rho1 = rand(2, 2)
        rho2 = rand(4, 4)
        rho3 = wedge([rho1, rho2], bs, b3)
        for P1 in [P1 + I, I - P1], P2 in [P2 + I, I - P2] #Loop over different parity sectors because of superselection. Otherwise, minus signs come into play
            m1 = P1 * rho1 * P1
            m2 = P2 * rho2 * P2
            P3 = wedge([P1, P2], bs, b3)
            m3 = P3 * rho3 * P3
            @test wedge([m1, m2], bs, b3) == m3
        end

        H1 = Matrix(0.5b1[1]' * b1[1])
        H2 = Matrix(-0.1b2[2]' * b2[2] + 0.3b2[3]' * b2[3] + (b2[2]' * b2[3] + hc))
        vals1, vecs1 = eigen(H1)
        vals2, vecs2 = eigen(H2)
        H3 = Matrix(0.5b3[1]' * b3[1] - 0.1b3[2]' * b3[2] + 0.3b3[3]' * b3[3] + (b3[2]' * b3[3] + hc))
        vals3, vecs3 = eigen(H3)

        # test wedging with I (UniformScaling)
        H3w = wedge([H1, I], bs, b3) + wedge([I, H2], bs, b3)
        @test H3w == H3
        @test wedge([I, I], bs, b3) == one(H3)

        vals3w = map(sum, Base.product(vals1, vals2)) |> vec
        p = sortperm(vals3w)
        vals3w[p] ≈ vals3

        vecs3w = vec(map(v12 -> fermionic_kron([v12[1], v12[2]], bs, b3), Base.product(eachcol(vecs1), eachcol(vecs2))))[p]
        @test all(map((v3, v3w) -> abs(dot(v3, v3w)) ≈ norm(v3) * norm(v3w), eachcol(vecs3), vecs3w))

        β = 0.7
        rho1 = exp(-β * H1)
        rmul!(rho1, 1 / tr(rho1))
        rho2 = exp(-β * H2)
        rmul!(rho2, 1 / tr(rho2))
        rho3 = exp(-β * H3)
        rmul!(rho3, 1 / tr(rho3))
        rho3w = wedge([rho1, rho2], bs, b3)
        @test rho3w ≈ rho3
        @test partial_trace(rho3, b3, b1) ≈ rho1
        @test partial_trace(rho3, b3, b2) ≈ rho2
        @test wedge([blockdiagonal(rho1, b1), blockdiagonal(rho2, b2)], bs, b3) ≈ wedge([blockdiagonal(rho1, b1), rho2], bs, b3)
        @test wedge([blockdiagonal(rho1, b1), blockdiagonal(rho2, b2)], bs, b3) ≈ rho3

        # Test BD1_hamiltonian
        b1 = FermionBasis(1:2, (:↑, :↓); qn)
        b2 = FermionBasis(3:4, (:↑, :↓); qn)
        b12 = FermionBasis(1:4, (:↑, :↓); qn)
        b12w = wedge(b1, b2)
        bs = [b1, b2]
        θ1 = 0.5
        θ2 = 0.2
        params1 = (; μ=1, t=0.5, Δ=2.0, V=0, θ=parameter(θ1, :diff), ϕ=1.0, h=4.0, U=2.0, Δ1=0.1)
        params2 = (; μ=1, t=0.1, Δ=1.0, V=0, θ=parameter(θ2, :diff), ϕ=5.0, h=1.0, U=10.0, Δ1=-1.0)
        params12 = (; μ=[params1.μ, params1.μ, params2.μ, params2.μ], t=[params1.t, 0, params2.t, 0], Δ=[params1.Δ, params1.Δ, params2.Δ, params2.Δ], V=[params1.V, 0, params2.V, 0], θ=[0, θ1, 0, θ2], ϕ=[params1.ϕ, params1.ϕ, params2.ϕ, params2.ϕ], h=[params1.h, params1.h, params2.h, params2.h], U=[params1.U, params1.U, params2.U, params2.U], Δ1=[params1.Δ1, 0, params2.Δ1, 0])
        H1 = Matrix(QuantumDots.BD1_hamiltonian(b1; params1...))
        H2 = Matrix(QuantumDots.BD1_hamiltonian(b2; params2...))

        H12w = Matrix(wedge([H1, I], bs, b12w) + wedge([I, H2], bs, b12w))
        H12 = Matrix(QuantumDots.BD1_hamiltonian(b12; params12...))

        v12w = fermionic_kron([eigvecs(Matrix(H1))[:, 1], eigvecs(Matrix(H2))[:, 1]], bs, b12w)
        v12 = eigvecs(H12)[:, 1]
        v12ww = eigvecs(H12w)[:, 1]
        sort(abs.(v12w)) - sort(abs.(v12))
        @test sum(abs, v12w) ≈ sum(abs, v12)
        @test sum(abs, v12w) ≈ sum(abs, v12ww)
        @test diff(eigvals(H12w)) ≈ diff(eigvals(H12))

        # Test zero-mode wedge
        c1 = FermionBasis(1:0; qn)
        c2 = FermionBasis(1:1; qn)
        @test wedge([I(1), I(1)], [c1, c1], c1) == I(1)
        @test wedge([I(1), c2[1]], [c1, c2], c2) == c2[1]
    end

    #Test basis compatibility
    b1 = FermionBasis(1:2; qn=QuantumDots.parity)
    b2 = FermionBasis(2:4; qn=QuantumDots.parity)
    @test_throws ArgumentError wedge(b1, b2)
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
phase_map(b::FermionBasis) = phase_map(collect(get_fockstates(b)), nbr_of_modes(b))
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
        c = FermionBasis(1:N)
        q = QubitBasis(1:N)
        @test all(map((c, q) -> q == phis[N](c), c, q))
        c2 = map(c -> phis[N](c), c)
        @test phis[N](phis[N](c[1])) == c[1]
        # c is fermionic
        @test all([c[n] * c[n2] == -c[n2] * c[n] for n in 1:N, n2 in 1:N])
        @test all([c[n]' * c[n2] == -c[n2] * c[n]' + I * (n == n2) for n in 1:N, n2 in 1:N])
        # c2 is hardcore bosons
        @test all([c2[n] * c2[n2] == c2[n2] * c2[n] for n in 1:N, n2 in 1:N])
        @test all([c2[n]' * c2[n2] == (-c2[n2] * c2[n]' + I) * (n == n2) + (n !== n2) * (c2[n2] * c2[n]') for n in 1:N, n2 in 1:N])
    end

    c1 = FermionBasis(1:1)
    c2 = FermionBasis(2:2)
    c12 = FermionBasis(1:2)
    p1 = QuantumDots.LazyPhaseMap(1)
    p2 = QuantumDots.phase_map(2)
    @test QuantumDots.fermionic_tensor_product_with_kron_and_maps((c1[1], I(2)), (p1, p1), p2) == c12[1]
    @test QuantumDots.fermionic_tensor_product_with_kron_and_maps((I(2), c2[2]), (p1, p1), p2) == c12[2]

    ms = (rand(2, 2), rand(2, 2))
    @test QuantumDots.fermionic_tensor_product_with_kron_and_maps(ms, (p1, p1), p2) == fermionic_kron(ms, (c1, c2), c12)
end

function fermionic_tensor_product_with_kron_and_maps(ops, phis, phi)
    phi(kron(reverse(map((phi, op) -> phi(op), phis, ops))...))
end

## kron, i.e. wedge without phase factors
Base.kron(ms, bs, b::AbstractManyBodyBasis; kwargs...) = fermionic_kron(ms, bs, b, false; kwargs...)

canonical_embedding(m, b, bnew) = fermionic_embedding(m, b, bnew, false)


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

FockSplitter(b::AbstractManyBodyBasis, bs) = FockSplitter(b.jw, map(b -> b.jw, bs))
FockMapper(bs, b::AbstractManyBodyBasis) = FockMapper(map(b -> b.jw, bs), b.jw)
use_reshape_phase_factors(b::FermionBasis, bs) = true

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


@testitem "Embedding unitary action" begin
    # Appendix C.4
    import QuantumDots: embedding_unitary, canonical_embedding, bipartite_embedding_unitary
    using LinearAlgebra
    cA = FermionBasis((1, 3))
    cB = FermionBasis((2, 4))
    c = FermionBasis((1, 2, 3, 4))
    @test embedding_unitary((cA, cB), c) == embedding_unitary([[1, 3], [2, 4]], c)
    @test fermionic_embedding(cA[1], cA, c) ≈ fermionic_kron((cA[1], I), (cA, cB), c) ≈ fermionic_kron((I, cA[1]), (cB, cA), c)
    Ux = embedding_unitary((cA, cB), c)
    Ux2 = bipartite_embedding_unitary(cA, cB, c)
    @test Ux ≈ Ux2
    @test fermionic_embedding(cA[1], cA, c) ≈ Ux * canonical_embedding(cA[1], cA, c) * Ux'
end
