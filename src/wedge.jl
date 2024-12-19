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
    T = promote_type(map(eltype, ms)...)
    dimlengths = map(length ∘ get_fockstates, bs)
    Nout = prod(dimlengths)
    fockmapper = if match_labels
        FockMapper(map(Base.Fix2(siteindices, b) ∘ Tuple ∘ keys, bs))
    else
        Ms = map(nbr_of_fermions, bs)
        shifts = (0, cumsum(Ms)...)
        FockShifter(shifts)
    end
    if ndims(first(ms)) == 1
        mout = zeros(T, Nout)
        return wedge_vec!(mout, Tuple(ms), Tuple(bs), b, fockmapper)
    elseif ndims(first(ms)) == 2
        mout = zeros(T, Nout, Nout)
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
