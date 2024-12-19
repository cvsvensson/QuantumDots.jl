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

get_fockstates(::FermionBasis{M,<:Any,NoSymmetry}) where {M} = 0:2^M-1
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
        fermionpositions = map(Base.Fix2(siteindices, b) ∘ Tuple ∘ keys, bs)
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
(fm::FockMapper)(f::NTuple{N,Int}) where {N} = mapreduce(insert_bits, +, f, fm.fermionpositions)
(fs::FockShifter)(f::NTuple{N,Int}) where {N} = mapreduce((f, M) -> 2^M * f, +, f, fs.shifts)
get_partition(f::FockMapper) = f.fermionpositions
get_partition(f::FockShifter) = map((s1, s2) -> s1+1:s2, f.shifts, Iterators.drop(f.shifts, 1))

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
    partition = get_partition(fockmapper)
    U = embedding_unitary(partition, get_fockstates(b))
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

function embedding_unitary(partition, fockstates)
    #for locally physical algebra, ie only for even operators or states of well-defined parity
    #if ξ is ordered, the phases are +1. 
    # Note that the jordan wigner modes are ordered in reverse from the labels, but this is taken care of by direction of the jwstring below
    phases = ones(Int, length(fockstates))
    for (s, Xs) in enumerate(partition)
        mask = focknbr_from_site_indices(Xs)
        for (r, Xr) in Iterators.drop(enumerate(partition), s)
            for i in Xr
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



@testitem "Wedge properties" begin
    # Properties from J. Phys. A: Math. Theor. 54 (2021) 393001
    # Eq. 16
    using Random, Base.Iterators, LinearAlgebra
    Random.seed!(1234)
    N = 7
    rough_size = 5
    fine_size = 3
    rough_partitions = collect(partition(randperm(N), rough_size))
    # divide each part of rough partition into finer partitions
    fine_partitions = map(rough_partition -> collect(partition(shuffle(rough_partition), fine_size)), rough_partitions)
    c = FermionBasis(1:N)
    cs_rough = [FermionBasis(r_p) for r_p in rough_partitions]
    cs_fine = map(f_p_list -> FermionBasis.(f_p_list), fine_partitions)
    ops_rough = map(r_p -> rand(ComplexF64, 2^length(r_p), 2^length(r_p)), rough_partitions)
    ops_fine = map(f_p_list -> [rand(ComplexF64, 2^length(f_p), 2^length(f_p)) for f_p in f_p_list], fine_partitions)
    rhs = wedge(reduce(vcat, ops_fine), reduce(vcat, cs_fine), c)
    lhs = wedge([wedge(ops_vec, cs_vec, c_rough) for (ops_vec, cs_vec, c_rough) in zip(ops_fine, cs_fine, cs_rough)], cs_rough, c)
    @test lhs ≈ rhs

    # Eq. 18
    As = ops_rough
    Bs = map(r_p -> rand(ComplexF64, 2^length(r_p), 2^length(r_p)), rough_partitions)
    lhs = tr(wedge(As, cs_rough, c)' * wedge(Bs, cs_rough, c))
    rhs = mapreduce((A, B) -> tr(A' * B), *, As, Bs)
    @test lhs ≈ rhs
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
        @test norm(b3w .- b3) == 0
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
    fockstates::Vector{Int}
end
struct LazyPhaseMap{M} <: AbstractMatrix{Int}
    fockstates::Vector{Int}
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
function phase_map(fockstates::AbstractVector, M::Int)
    phases = zeros(Int, length(fockstates), length(fockstates))
    for (n1, f1) in enumerate(fockstates)
        for (n2, f2) in enumerate(fockstates)
            phases[n1, n2] = phase_factor(f1, f2, M)
        end
    end
    PhaseMap(phases, fockstates)
end
phase_map(N::Int) = phase_map(0:2^N-1, N)
LazyPhaseMap(N::Int) = LazyPhaseMap{N}(0:2^N-1)
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
