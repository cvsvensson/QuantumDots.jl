function qubit_sparse_matrix(qubit_number, totalsize, sym)
    mat = spzeros(Int, totalsize, totalsize)
    _fill!(mat, fs -> lower_qubit(qubit_number, fs), sym)
    mat
end

function lower_qubit(digitposition, statefocknbr)
    cdag = focknbr_from_site_index(digitposition)
    newfocknbr = cdag ⊻ statefocknbr
    allowed = !iszero(cdag & statefocknbr)
    return allowed * newfocknbr, allowed
end

struct QubitBasis{M,D,Sym,L} <: AbstractManyBodyBasis
    dict::D
    symmetry::Sym
    jw::JordanWignerOrdering{L}
end
function QubitBasis(iters...; qn=NoSymmetry(), kwargs...)
    labels = handle_labels(iters...)
    labelvec = collect(labels)[:]
    jw = JordanWignerOrdering(labelvec)
    M = length(labels)
    labelled_symmetry = instantiate(qn, jw)
    fockstates = map(FockNumber, get(kwargs, :fockstates, 0:2^M-1))
    sym_concrete = focksymmetry(fockstates, labelled_symmetry)
    reps = ntuple(n -> qubit_sparse_matrix(n, length(fockstates), sym_concrete), M)
    d = OrderedDict(zip(labels, reps))
    QubitBasis{M,typeof(d),typeof(sym_concrete),_label_type(jw)}(d, sym_concrete, jw)
end
Base.getindex(b::QubitBasis, i) = b.dict[i]
function Base.getindex(b::QubitBasis, args...)
    if length(args) == length(first(keys(b)))
        return b.dict[args]
    else
        op = QubitOp{last(args)}()
        return qubit_operator(b[ntuple(i -> args[i], length(args) - 1)...], op)
    end
end

struct QubitOp{S} end
qubit_operator(c, ::QubitOp{:Z}) = 2c'c - I
qubit_operator(c, ::QubitOp{:X}) = c + c'
qubit_operator(c, ::QubitOp{:Y}) = 1im * (c' - c)
qubit_operator(c, ::QubitOp{:I}) = 0c + I
qns(b::QubitBasis) = qns(b.symmetry)


Base.keys(b::QubitBasis) = keys(b.dict)
Base.show(io::IO, ::MIME"text/plain", b::QubitBasis) = show(io, b)
Base.show(io::IO, b::QubitBasis{M,D,Sym}) where {M,D,Sym} = print(io, "QubitBasis{$M,$D,$Sym}:\nkeys = ", keys(b))
Base.iterate(b::QubitBasis) = iterate(values(b.dict))
Base.iterate(b::QubitBasis, state) = iterate(values(b.dict), state)
Base.length(::QubitBasis{M}) where {M} = M
symmetry(basis::QubitBasis) = basis.symmetry
Base.eltype(b::QubitBasis) = eltype(b.dict)
Base.keytype(b::QubitBasis) = keytype(b.dict)

get_fockstates(::QubitBasis{M,<:Any,NoSymmetry}) where {M} = Iterators.map(FockNumber, 0:2^M-1)
get_fockstates(b::QubitBasis) = get_fockstates(b.symmetry)

function partial_trace!(mout, m::AbstractMatrix{T}, labels, b::QubitBasis{M}, sym::AbstractSymmetry=NoSymmetry()) where {T,M}
    N = length(labels)
    fill!(mout, zero(eltype(mout)))
    outinds::NTuple{N,Int} = siteindices(labels, b)
    bitmask = FockNumber(2^M - 1) - focknbr_from_site_indices(outinds)
    outbits(f) = map(i -> _bit(f, i), outinds)
    fockstates = get_fockstates(b)
    for f1 in get_fockstates(b), f2 in get_fockstates(b)
        if (f1 & bitmask) != (f2 & bitmask)
            continue
        end
        newfocknbr1 = focknbr_from_bits(outbits(f1))
        newfocknbr2 = focknbr_from_bits(outbits(f2))
        mout[focktoind(newfocknbr1, sym), focktoind(newfocknbr2, sym)] += m[focktoind(f1, b), focktoind(f2, b)]
    end
    return mout
end

function bloch_vector(ρ::AbstractMatrix, label, basis::QubitBasis)
    map(op -> real(tr(ρ * basis[label, op])), [:X, :Y, :Z]) / 2^length(basis)
end


@testitem "QubitBasis" begin
    using SparseArrays, Random, LinearAlgebra
    Random.seed!(1234)

    N = 2
    B = QubitBasis(1:N)
    @test length(B) == N
    Bspin = QubitBasis(1:N, (:↑, :↓); qn=FermionConservation())
    @test length(Bspin) == 2N
    @test B[1] isa SparseMatrixCSC
    @test Bspin[1, :↑] isa SparseMatrixCSC
    @test parityoperator(B) isa SparseMatrixCSC
    @test parityoperator(Bspin) isa SparseMatrixCSC
    @test B[1] + B[1]' ≈ B[1, :X]
    @test 2B[1]'B[1] - I ≈ B[1, :Z]
    @test 1im * (B[1]' - B[1]) ≈ B[1, :Y]
    @test I ≈ B[1, :I]
    @test QuantumDots.bloch_vector(B[1, :X] + B[1, :Y] + B[1, :Z], 1, B) ≈ [1, 1, 1]
    @test pretty_print(B[1, :X], B) |> isnothing
    @test pretty_print(B[1, :X][:, 1], B) |> isnothing
    @test pretty_print(Bspin[1, :↑, :X], Bspin) |> isnothing
    @test pretty_print(Bspin[1, :↑, :X][:, 1], Bspin) |> isnothing

    (c,) = QuantumDots.cell(1, B)
    @test c == B[1]
    (c1, c2) = QuantumDots.cell(1, Bspin)
    @test c1 == Bspin[1, :↑]
    @test c2 == Bspin[1, :↓]

    a = QubitBasis(1:3)
    @test all(f == a[n] for (n, f) in enumerate(a))
    v = [QuantumDots.indtofock(i, a) for i in 1:8]
    t1 = QuantumDots.tensor(v, a)
    t2 = [i1 + 2i2 + 4i3 for i1 in (0, 1), i2 in (0, 1), i3 in (0, 1)]
    @test t1 == FockNumber.(t2)

    a = QubitBasis(1:3; qn=QuantumDots.parity)
    v = [QuantumDots.indtofock(i, a) for i in 1:8]
    t1 = QuantumDots.tensor(v, a)
    t2 = [i1 + 2i2 + 4i3 for i1 in (0, 1), i2 in (0, 1), i3 in (0, 1)]
    @test t1 == FockNumber.(t2)

    v2 = rand(8)
    @test sort(QuantumDots.svd(v2, (1,), a).S .^ 2) ≈ eigvals(partial_trace(v2, (1,), a))

    c = QubitBasis(1:2, (:a, :b))
    cparity = QubitBasis(1:2, (:a, :b); qn=QuantumDots.parity)
    ρ = Matrix(Hermitian(rand(2^4, 2^4) .- 0.5))
    ρ = ρ / tr(ρ)
    function bilinears(c, labels)
        ops = reduce(vcat, [[c[l], c[l]'] for l in labels])
        return [op1 * op2 for (op1, op2) in Base.product(ops, ops)]
    end
    function bilinear_equality(c, csub, ρ)
        subsystem = Tuple(keys(csub))
        ρsub = partial_trace(ρ, csub, c)
        @test tr(ρsub) ≈ 1
        all((tr(op1 * ρ) ≈ tr(op2 * ρsub)) for (op1, op2) in zip(bilinears(c, subsystem), bilinears(csub, subsystem)))
    end
    function get_subsystems(c, N)
        t = collect(Base.product(ntuple(i -> keys(c), N)...))
        (t[I] for I in CartesianIndices(t) if issorted(Tuple(I)) && allunique(Tuple(I)))
    end
    for N in 1:4
        @test all(bilinear_equality(c, QubitBasis(subsystem), ρ) for subsystem in get_subsystems(c, N))
        @test all(bilinear_equality(c, QubitBasis(subsystem; qn=QuantumDots.parity), ρ) for subsystem in get_subsystems(c, N))
        @test all(bilinear_equality(c, QubitBasis(subsystem; qn=QuantumDots.parity), ρ) for subsystem in get_subsystems(cparity, N))
        @test all(bilinear_equality(c, QubitBasis(subsystem), ρ) for subsystem in get_subsystems(cparity, N))
    end
    bilinear_equality(c, QubitBasis(((1, :b), (1, :a))), ρ)
end