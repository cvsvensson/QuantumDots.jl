"""
    wedge(b1::FermionBasis, b2::FermionBasis)

Compute the wedge product of two `FermionBasis` objects. The symmetry of the resulting basis is computed by promote_symmetry.
"""
function wedge(b1::FermionBasis, b2::FermionBasis)
    newlabels = vcat(collect(keys(b1)), collect(keys(b2)))
    if length(unique(newlabels)) != length(newlabels)
        throw(ArgumentError("The labels of the two bases are not disjoint"))
    end
    qn = promote_symmetry(b1.symmetry, b2.symmetry)
    FermionBasis(newlabels; qn)
end
wedge(b1::FermionBasis, bs...) = foldl(wedge, bs, init=b1)

promote_symmetry(s1::AbelianFockSymmetry{<:Any,<:Any,<:Any,F}, s2::AbelianFockSymmetry{<:Any,<:Any,<:Any,F}) where {F} = s1.conserved_quantity
promote_symmetry(::AbelianFockSymmetry{<:Any,<:Any,<:Any,F1}, ::AbelianFockSymmetry{<:Any,<:Any,<:Any,F2}) where {F1,F2} = NoSymmetry()
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
get_fockstates(sym::AbelianFockSymmetry) = sym.indtofockdict
"""
    wedge(ms::AbstractVector, bs::AbstractVector{<:FermionBasis}, b::FermionBasis=wedge(bs...))

Compute the wedge product of matrices or vectors in `ms` with respect to the fermion bases `bs`, respectively. Return a matrix in the fermion basis `b`, which defaults to the wedge product of `bs`.
"""
function wedge(ms, bs, b::FermionBasis=wedge(bs...); match_labels=true)
    T = Base.promote_eltype(ms...)
    N = ndims(first(ms))
    MT = Base.promote_op(kron, Array{T,N}, Array{T,N}, filter(!Base.Fix2(<:, UniformScaling), map(typeof, ms))...) # Array{T,N} is there as a fallback make if there aren't enough arguments
    dimlengths = map(length ∘ get_fockstates, bs)
    Nout = prod(dimlengths)
    fockmapper = if match_labels
        fermionpositions = map(Base.Fix2(siteindices, b.jw) ∘ Tuple ∘ keys, bs)
        FockMapper(fermionpositions)
    else
        Ms = map(nbr_of_fermions, bs)
        shifts = (0, cumsum(Ms)...)
        FockShifter(shifts)
    end
    mout = convert(MT, zeros(T, ntuple(j -> Nout, N)))
    if N == 1
        return wedge_vec!(mout, Tuple(ms), Tuple(bs), b, fockmapper)
    elseif N == 2
        return wedge_mat!(mout, Tuple(ms), Tuple(bs), b, fockmapper)
    end
    throw(ArgumentError("Only 1D or 2D arrays are supported"))
end

struct FockMapper{P}
    fermionpositions::P
end
struct FockShifter{M}
    shifts::M
end
(fm::FockMapper)(f::NTuple{N,FockNumber}) where {N} = mapreduce(insert_bits, +, f, fm.fermionpositions)
(fs::FockShifter)(f::NTuple{N,FockNumber}) where {N} = mapreduce((f, M) -> shift_right(f, M), +, f, fs.shifts)
shift_right(f::FockNumber, M) = FockNumber(f.f << M)

function wedge_mat!(mout, ms::Tuple, bs::Tuple, b::FermionBasis, fockmapper)
    fill!(mout, zero(eltype(mout)))
    Ms = map(nbr_of_fermions, bs)
    Mout = sum(Ms)
    dimlengths = map(length ∘ get_fockstates, bs)
    inds = CartesianIndices(dimlengths)
    for I in inds
        TI = Tuple(I)
        fock1 = map(indtofock, TI, bs)
        fullfock1 = fockmapper(fock1)
        outind = focktoind(fullfock1, b)
        for I2 in inds
            TI2 = Tuple(I2)
            fock2 = map(indtofock, TI2, bs)
            fullfock2 = fockmapper(fock2)
            v = mapreduce((m, b, i1, f1, i2, f2, M) -> m[i1, i2] * phase_factor(f1, f2, M), *, ms, bs, TI, fock1, TI2, fock2, Ms)
            mout[outind, focktoind(fullfock2, b)] += v * phase_factor(fullfock1, fullfock2, Mout)
        end
    end
    return mout
end
function wedge_vec!(mout, ms::Tuple, bs::Tuple, b::FermionBasis, fockmapper)
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

"""
    embedding_unitary(partition, fockstates, jw)

    Compute the unitary matrix that maps between the tensor embedding and the fermionic embedding in the physical. 
    # Arguments
    - `partition`: A partition of the labels in `jw` into disjoint sets.
    - `fockstates`: The fock states in the basis
    - `jw`: The Jordan-Wigner ordering.
"""
function embedding_unitary(partition, fockstates, jw::JordanWignerOrdering)
    #for locally physical algebra, ie only for even operators or states of well-defined parity
    #if ξ is ordered, the phases are +1. 
    # Note that the jordan wigner modes are ordered in reverse from the labels, but this is taken care of by direction of the jwstring below
    phases = ones(Int, length(fockstates))
    for (s, Xs) in enumerate(partition)
        mask = focknbr_from_site_labels(Xs, jw)
        for (r, Xr) in Iterators.drop(enumerate(partition), s)
            for li in Xr
                i = siteindex(li, jw)
                for (n, f) in zip(eachindex(phases), fockstates)
                    if _bit(f, i)
                        phases[n] *= jwstring_right(i, mask & f)
                    end
                end
            end
        end
    end
    return Diagonal(phases)
end
embedding_unitary(partition, c::FermionBasis) = embedding_unitary(partition, get_fockstates(c), c.jw)
embedding_unitary(cs::Union{<:AbstractVector{B},<:NTuple{N,B}}, c::FermionBasis) where {B<:FermionBasis,N} = embedding_unitary(map(keys, cs), c)

@testitem "Embedding unitary" begin
    # Appendix C.4
    using LinearAlgebra
    jw = JordanWignerOrdering(1:2)
    fockstates = sort(map(FockNumber, 0:3), by=Base.Fix2(bits, 2))

    @test QuantumDots.embedding_unitary([[1], [2]], fockstates, jw) == I
    @test QuantumDots.embedding_unitary([[2], [1]], fockstates, jw) == Diagonal([1, 1, 1, -1])

    # N = 3
    jw = JordanWignerOrdering(1:3)
    fockstates = sort(map(FockNumber, 0:7), by=Base.Fix2(bits, 3))
    U(p) = QuantumDots.embedding_unitary(p, fockstates, jw)
    @test U([[1], [2], [3]]) == U([[1, 2], [3]]) == U([[1], [2, 3]]) == I

    @test U([[2], [1], [3]]) == Diagonal([1, 1, 1, 1, 1, 1, -1, -1])
    @test U([[2], [3], [1]]) == Diagonal([1, 1, 1, 1, 1, -1, -1, 1])
    @test U([[3], [1], [2]]) == Diagonal([1, 1, 1, -1, 1, -1, 1, 1])
    @test U([[3], [2], [1]]) == Diagonal([1, 1, 1, -1, 1, -1, -1, -1])
    @test U([[1], [3], [2]]) == Diagonal([1, 1, 1, -1, 1, 1, 1, -1])

    @test U([[2], [1, 3]]) == Diagonal([1, 1, 1, 1, 1, 1, -1, -1])
    @test U([[3], [1, 2]]) == Diagonal([1, 1, 1, -1, 1, -1, 1, 1])

    @test U([[1, 3], [2]]) == Diagonal([1, 1, 1, -1, 1, 1, 1, -1])
    @test U([[2, 3], [1]]) == Diagonal([1, 1, 1, 1, 1, -1, -1, 1])

    ##
    cA = FermionBasis((1, 3))
    cB = FermionBasis((2, 4))
    c = FermionBasis((1, 2, 3, 4))
    @test QuantumDots.embedding_unitary((cA, cB), c) == QuantumDots.embedding_unitary([[1, 3], [2, 4]], c)
    @test fermionic_embedding(cA[1], cA, c) ≈ wedge((cA[1], I), (cA, cB), c) ≈ wedge((I, cA[1]), (cB, cA), c)
    @test fermionic_embedding(cB[2], cB, c) ≈ wedge((I, cB[2]), (cA, cB), c) ≈ wedge((cB[2], I), (cB, cA), c)
    @test fermionic_embedding(cB[2], cB, c) * fermionic_embedding(cA[1], cA, c) ≈ wedge([cA[1], cB[2]], (cA, cB), c)

    Ux = QuantumDots.embedding_unitary((cB, cA), c)
    @test fermionic_embedding(cA[1], cA, c) ≈ Ux * QuantumDots.canonical_embedding(cA[1], cA, c) * Ux'
    # wedge([cA[1], cB[2]], (cA,cB), c)
end

"""
    fermionic_embedding(m, b, bnew)

Compute the fermionic embedding of a matrix `m` in the basis `b` into the basis `bnew`.
"""
function fermionic_embedding(m, b, bnew)
    # See eq. 20 in J. Phys. A: Math. Theor. 54 (2021) 393001
    bbar_labs = setdiff(collect(keys(bnew)), collect(keys(b))) # arrays to keep order
    # qn = promote_symmetry(b.symmetry, bnew.symmetry)
    qn = NoSymmetry()
    bbar = FermionBasis(bbar_labs; qn)
    return wedge((m, I), (b, bbar), bnew)
end

"""
    ordered_prod_of_embeddings(ms, bs, b)

Compute the ordered product of the fermionic embeddings of the matrices `ms` in the bases `bs` into the basis `b`.
"""
function ordered_prod_of_embeddings(ms, bs, b)
    # See eq. 26 in J. Phys. A: Math. Theor. 54 (2021) 393001
    # note that the multiplication is done in the reverse order
    return mapreduce(((m, fine_basis),) -> fermionic_embedding(m, fine_basis, b), *, zip(reverse(ms), reverse(bs)))
end


@testitem "Wedge properties" begin
    # Properties from J. Phys. A: Math. Theor. 54 (2021) 393001
    # Eq. 16
    using Random, Base.Iterators, LinearAlgebra
    import QuantumDots: fermionic_embedding, ordered_prod_of_embeddings

    Random.seed!(1234)
    N = 7
    rough_size = 5
    fine_size = 3
    rough_partitions = (collect(partition(randperm(N), rough_size)))
    # divide each part of rough partition into finer partitions
    fine_partitions = map(rough_partition -> (collect(partition(shuffle(rough_partition), fine_size))), rough_partitions)
    c = FermionBasis(1:N)
    cs_rough = [FermionBasis(r_p) for r_p in rough_partitions]
    cs_fine = map(f_p_list -> FermionBasis.(f_p_list), fine_partitions)

    ordered_rough_partitions = map(p -> p[randperm(length(p))], partition(1:N, rough_size))
    fine_partitions_from_ordered_rough = map(rough_partition -> (collect(partition(shuffle(rough_partition), fine_size))), rough_partitions)
    ordered_cs_rough = [FermionBasis(r_p) for r_p in ordered_rough_partitions]

    # Associativity (Eq. 16)
    ops_rough = map(r_p -> rand(ComplexF64, 2^length(r_p), 2^length(r_p)), rough_partitions)
    ops_fine = map(f_p_list -> [rand(ComplexF64, 2^length(f_p), 2^length(f_p)) for f_p in f_p_list], fine_partitions)
    rhs = wedge(reduce(vcat, ops_fine), reduce(vcat, cs_fine), c)
    lhs = wedge([wedge(ops_vec, cs_vec, c_rough) for (ops_vec, cs_vec, c_rough) in zip(ops_fine, cs_fine, cs_rough)], cs_rough, c)
    @test lhs ≈ rhs

    physical_ops_rough = map((op, c) -> (parityoperator(c) + I) / 2 * op * (parityoperator(c) + I) / 2, ops_rough, cs_rough)

    # Eq. 18
    As = ops_rough
    Bs = map(r_p -> rand(ComplexF64, 2^length(r_p), 2^length(r_p)), rough_partitions)
    lhs = tr(wedge(As, cs_rough, c)' * wedge(Bs, cs_rough, c))
    rhs = mapreduce((A, B) -> tr(A' * B), *, As, Bs)
    @test lhs ≈ rhs

    # Fermionic embedding

    # Eq. 19 (note that the ordered product is reversed from the article above)
    As_modes = [rand(ComplexF64, 2, 2) for _ in 1:N]
    ξ = vcat(fine_partitions...)
    ξbases = vcat(cs_fine...)
    modebases = [FermionBasis(j:j) for j in 1:N]
    lhs = mapreduce(j -> fermionic_embedding(As_modes[j], modebases[j], c), *, N:-1:1)
    rhs_ordered_prod(X, basis) = mapreduce(j -> fermionic_embedding(As_modes[j], modebases[j], basis), *, reverse(X))
    rhs = wedge([rhs_ordered_prod(X, b) for (X, b) in zip(ξ, ξbases)], ξbases, c)
    @test lhs ≈ rhs

    # Associativity (Eq. 21)
    @test fermionic_embedding(fermionic_embedding(ops_fine[1][1], cs_fine[1][1], cs_rough[1]), cs_rough[1], c) ≈ fermionic_embedding(ops_fine[1][1], cs_fine[1][1], c)
    @test all(map(cs_rough, cs_fine, ops_fine) do cr, cfs, ofs
        all(map(cfs, ofs) do cf, of
            fermionic_embedding(fermionic_embedding(of, cf, cr), cr, c) ≈ fermionic_embedding(of, cf, c)
        end)
    end)

    # Eq. 22
    cX = cs_rough[1]
    Ux = QuantumDots.embedding_unitary(rough_partitions, c)

    # ordered_rough_cs = FermionBasis.(ordered_rough_partitions)
    # cX = ordered_rough_cs[1]
    # Ux = QuantumDots.embedding_unitary(ordered_rough_cs, c)

    A = ops_rough[1]
    Aphys = physical_ops_rough[1]
    canon_emb = QuantumDots.canonical_embedding
    @test fermionic_embedding(A, cX, c) ≈ Ux * canon_emb(A, cX, c) * Ux'
    @test fermionic_embedding(Aphys, cX, c) ≈ Ux * canon_emb(Aphys, cX, c) * Ux'

    ordered_prod_of_embeddings(ops_rough, cs_rough, c) - Ux * kron(ops_rough, cs_rough, c) * Ux'

    orderedUx = QuantumDots.embedding_unitary(ordered_rough_partitions, c)
    @test orderedUx ≈ I
    @test ordered_prod_of_embeddings(physical_ops_rough, ordered_cs_rough, c) ≈ kron(physical_ops_rough, ordered_cs_rough, c)

    # Eq. 23
    X = rough_partitions[1]
    cX = cs_rough[1]
    A = ops_rough[1]
    B = rand(ComplexF64, 2^length(X), 2^length(X))

    @test canon_emb(A, cX, c) * canon_emb(B, cX, c) ≈ canon_emb(A * B, cX, c)
    @test fermionic_embedding(A, cX, c) * fermionic_embedding(B, cX, c) ≈ fermionic_embedding(A * B, cX, c)
    @test fermionic_embedding(A, cX, c)' ≈ fermionic_embedding(A', cX, c)

    # Ordered product of embeddings

    # Eq. 31
    A = ops_rough[1]
    X = rough_partitions[1]
    Xbar = setdiff(1:N, X)
    cX = cs_rough[1]
    cXbar = FermionBasis(Xbar)
    corr = fermionic_embedding(A, cX, c)
    @test corr ≈ wedge([A, I], [cX, cXbar], c) ≈ ordered_prod_of_embeddings([A, I], [cX, cXbar], c) ≈ ordered_prod_of_embeddings([I, A], [cXbar, cX], c)

    # Eq. 32
    @test ordered_prod_of_embeddings(As_modes, modebases, c) ≈ wedge(As_modes, modebases, c)

    # Fermionic partial trace

    # Eq. 36
    X = rough_partitions[1]
    A = ops_rough[1]
    B = rand(ComplexF64, 2^N, 2^N)
    cX = cs_rough[1]
    lhs = tr(fermionic_embedding(A, cX, c)' * B)
    rhs = tr(A' * partial_trace(B, cX, c))
    @test lhs ≈ rhs

    # Eq. 39
    A = rand(ComplexF64, 2^N, 2^N)
    X = fine_partitions[1][1]
    Y = rough_partitions[1]
    bX = cs_fine[1][1]
    bY = cs_rough[1]
    bZ = c
    Z = 1:N
    rhs = partial_trace(A, bX, bZ)
    lhs = partial_trace(partial_trace(A, bY, bZ), bX, bY)
    @test lhs ≈ rhs

    # Eq. 41
    bY = c
    @test partial_trace(A', bX, bY) ≈ partial_trace(A, bX, bY)'

    # Eq. 95
    ξ = rough_partitions
    As = ops_rough
    Bs = map(X -> rand(ComplexF64, 2^length(X), 2^length(X)), ξ)
    lhs1 = ordered_prod_of_embeddings(As, cs_rough, c) * ordered_prod_of_embeddings(Bs, cs_rough, c)
    rhs1 = ordered_prod_of_embeddings(As .* Bs, cs_rough, c)
    @test lhs1 ≈ rhs1
    @test ordered_prod_of_embeddings(As, cs_rough, c)' ≈ ordered_prod_of_embeddings(adjoint.(As), cs_rough, c)
end


@testitem "Wedge" begin
    using Random, LinearAlgebra
    Random.seed!(1234)

    for qn in [NoSymmetry(), ParityConservation(), FermionConservation()]
        b1 = FermionBasis(1:1; qn)
        b2 = FermionBasis(1:3; qn)
        @test_throws ArgumentError wedge(b1, b2)
        b2 = FermionBasis(2:3; qn)
        b3 = FermionBasis(1:3; qn)
        b3w = wedge(b1, b2)
        @test norm(map(-, b3w, b3)) == 0
        bs = [b1, b2]

        O1 = isodd.(QuantumDots.numberoperator(b1))
        O2 = isodd.(QuantumDots.numberoperator(b2))
        for P1 in [O1, I - O1], P2 in [O2, I - O2] #Loop over different parity sectors because of superselection. Otherwise, minus signs come into play
            v1 = P1 * rand(2)
            v2 = P2 * rand(4)
            v3 = wedge([v1, v2], bs)
            for k1 in keys(b1), k2 in keys(b2)
                b1f = b1[k1]
                b2f = b2[k2]
                b3f = b3[k2] * b3[k1]
                b3fw = wedge([b1f, b2f], bs, b3)
                v3w = wedge([b1f * v1, b2f * v2], bs, b3)
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

        vecs3w = vec(map(v12 -> wedge([v12[1], v12[2]], bs, b3), Base.product(eachcol(vecs1), eachcol(vecs2))))[p]
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
        bs = [b1, b2]
        @test partial_trace(wedge([rho1, rho2], bs, b3), b1, b3) ≈ rho1
        @test partial_trace(wedge([rho1, rho2], bs, b3), b2, b3) ≈ rho2
        @test wedge([blockdiagonal(rho1, b1), blockdiagonal(rho2, b2)], bs, b3) ≈ wedge([blockdiagonal(rho1, b1), rho2], bs, b3)
        @test wedge([blockdiagonal(rho1, b1), blockdiagonal(rho2, b2)], bs, b3) ≈ wedge([rho1, rho2], bs, b3)

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

        H12w = wedge([H1, I], bs, b12w) + wedge([I, H2], bs, b12w)
        H12 = Matrix(QuantumDots.BD1_hamiltonian(b12; params12...))

        v12w = wedge([eigvecs(Matrix(H1))[:, 1], eigvecs(Matrix(H2))[:, 1]], bs, b12w)
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

        # Test not matching labels
        c1 = FermionBasis(1:1; qn)
        c2 = FermionBasis(2:2; qn)
        c13 = FermionBasis([1, 3]; qn)
        c123 = FermionBasis(1:3; qn)
        @test wedge([c2[2], c13[3]], [c2, c13], c123) == c123[3] * c123[2]
        @test wedge([c1[1], c13[3]], [c1, c13], c123; match_labels=false) == c123[3] * c123[1]

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
Base.getindex(p::LazyPhaseMap{M}, n1::Int, n2::Int) where {M} = phase_factor(p.fockstates[n1], p.fockstates[n2], M)
function phase_map(fockstates, M::Int)
    phases = zeros(Int, length(fockstates), length(fockstates))
    for (n1, f1) in enumerate(fockstates)
        for (n2, f2) in enumerate(fockstates)
            phases[n1, n2] = phase_factor(f1, f2, M)
        end
    end
    PhaseMap(phases, fockstates)
end
phase_map(N::Int) = phase_map(map(FockNumber, 0:2^N-1), N)
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
    c2 = FermionBasis(1:2)
    p1 = QuantumDots.LazyPhaseMap(1)
    p2 = QuantumDots.phase_map(2)
    @test QuantumDots.fermionic_tensor_product((c1[1], I(2)), (p1, p1), p2) == c2[1]
    @test QuantumDots.fermionic_tensor_product((I(2), c1[1]), (p1, p1), p2) == c2[2]
end

function fermionic_tensor_product(ops, phis, phi)
    phi(kron(reverse(map((phi, op) -> phi(op), phis, ops))...))
end

## kron, i.e. wedge without phase factors
function Base.kron(ms, bs, b::FermionBasis=wedge(bs...); match_labels=true)
    T = Base.promote_eltype(ms...)
    N = ndims(first(ms))
    MT = Base.promote_op(kron, Array{T,N}, Array{T,N}, filter(!Base.Fix2(<:, UniformScaling), map(typeof, ms))...) # Array{T,N} is there as a fallback make if there aren't enough arguments
    dimlengths = map(length ∘ get_fockstates, bs)
    Nout = prod(dimlengths)
    fockmapper = if match_labels
        fermionpositions = map(Base.Fix2(siteindices, b.jw) ∘ Tuple ∘ keys, bs)
        FockMapper(fermionpositions)
    else
        Ms = map(nbr_of_fermions, bs)
        shifts = (0, cumsum(Ms)...)
        FockShifter(shifts)
    end
    mout = convert(MT, zeros(T, ntuple(j -> Nout, N)))
    if N == 1
        return kron_vec!(mout, Tuple(ms), Tuple(bs), b, fockmapper)
    elseif N == 2
        return kron_mat!(mout, Tuple(ms), Tuple(bs), b, fockmapper)
    end
    throw(ArgumentError("Only 1D or 2D arrays are supported"))
end

function kron_mat!(mout, ms::Tuple, bs::Tuple, b::FermionBasis, fockmapper)
    fill!(mout, zero(eltype(mout)))
    Ms = map(nbr_of_fermions, bs)
    Mout = sum(Ms)
    dimlengths = map(length ∘ get_fockstates, bs)
    inds = CartesianIndices(dimlengths)
    for I in inds
        TI = Tuple(I)
        fock1 = map(indtofock, TI, bs)
        fullfock1 = fockmapper(fock1)
        outind = focktoind(fullfock1, b)
        for I2 in inds
            TI2 = Tuple(I2)
            fock2 = map(indtofock, TI2, bs)
            fullfock2 = fockmapper(fock2)
            v = mapreduce((m, b, i1, i2) -> m[i1, i2], *, ms, bs, TI, TI2)
            mout[outind, focktoind(fullfock2, b)] += v
        end
    end
    return mout
end

function kron_vec!(mout, ms::Tuple, bs::Tuple, b::FermionBasis, fockmapper)
    fill!(mout, zero(eltype(mout)))
    dimlengths = map(length ∘ get_fockstates, bs)
    inds = CartesianIndices(Tuple(dimlengths))
    for I in inds
        TI = Tuple(I)
        fock = map(indtofock, TI, bs)
        fullfock = fockmapper(fock)
        outind = focktoind(fullfock, b)
        mout[outind] += mapreduce((i1, m) -> m[i1], *, TI, ms)
    end
    return mout
end

function canonical_embedding(m, b, bnew)
    bbar_labs = setdiff(collect(keys(bnew)), collect(keys(b)))
    # qn = promote_symmetry(b.symmetry, bnew.symmetry)
    qn = NoSymmetry()
    bbar = FermionBasis(bbar_labs; qn)
    return kron((m, I), (b, bbar), bnew)
end