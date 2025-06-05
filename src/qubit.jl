function qubit_sparse_matrix(qubit_number, H::AbstractFockHilbertSpace)
    mat = spzeros(Int, size(H))
    _fill!(mat, f -> lower_qubit(qubit_number, f), H)
    mat
end

function lower_qubit(digitposition, statefocknbr)
    cdag = focknbr_from_site_index(digitposition)
    newfocknbr = cdag ⊻ statefocknbr
    allowed = !iszero(cdag & statefocknbr)
    return allowed * newfocknbr, allowed
end

# struct QubitBasis{M,D,Sym,L} <: AbstractManyBodyBasis
#     dict::D
#     symmetry::Sym
#     jw::JordanWignerOrdering{L}
# end
# function QubitBasis(iters...; qn=NoSymmetry(), kwargs...)
#     labels = handle_labels(iters...)
#     labelvec = collect(labels)[:]
#     jw = JordanWignerOrdering(labelvec)
#     M = length(labels)
#     labelled_symmetry = instantiate(qn, jw)
#     fockstates = map(FockNumber, get(kwargs, :fockstates, 0:2^M-1))
#     sym_concrete = focksymmetry(fockstates, labelled_symmetry)
#     reps = ntuple(n -> qubit_sparse_matrix(n, length(fockstates), sym_concrete), M)
#     d = OrderedDict(zip(labels, reps))
#     QubitBasis{M,typeof(d),typeof(sym_concrete),eltype(jw)}(d, sym_concrete, jw)
# end
# Base.getindex(b::QubitBasis, i) = b.dict[i]
# function Base.getindex(b::QubitBasis, args...)
#     if length(args) == length(first(keys(b)))
#         return b.dict[args]
#     else
#         op = QubitOp{last(args)}()
#         return qubit_operator(b[ntuple(i -> args[i], length(args) - 1)...], op)
#     end
# end

struct QubitOp{S} end
function QubitOperators(H::AbstractFockHilbertSpace)
    M = length(H.jw)
    labelvec = keys(H.jw)
    reps = [1.0 * complex(qubit_sparse_matrix(n, H)) for n in 1:M]
    ops = OrderedDict{Any,eltype(reps)}(zip(labelvec, reps))
    for (k, op) in Base.product(labelvec, (:Z, :X, :Y, :H))
        ops[k, op] = qubit_operator(ops[k], QubitOp{op}())
    end
    ops
end
qubit_operator(c, ::QubitOp{:Z}) = -2c'c + I
qubit_operator(c, ::QubitOp{:X}) = c + c'
qubit_operator(c, ::QubitOp{:Y}) = 1im * (c' - c)
qubit_operator(c, ::QubitOp{:I}) = 0c + I
qubit_operator(c, ::QubitOp{:H}) = 1 / sqrt(2) * (qubit_operator(c, QubitOp{:Z}()) + qubit_operator(c, QubitOp{:X}())) # Hadamard operator


@testitem "QubitBasis" begin
    using SparseArrays, Random, LinearAlgebra
    using QuantumDots: QubitOperators, qubit_hilbert_space
    Random.seed!(1234)

    N = 2
    H = qubit_hilbert_space(1:N)
    B = QubitOperators(H)
    @test B[1] isa SparseMatrixCSC
    @test B[1] + B[1]' ≈ B[1, :X]
    @test I - 2B[1]'B[1] ≈ B[1, :Z]
    @test 1im * (B[1]' - B[1]) ≈ B[1, :Y]
    @test pretty_print(B[1, :X], H) |> isnothing
    @test pretty_print(B[1, :X][:, 1], H) |> isnothing

    H = qubit_hilbert_space(1:3)
    a = QubitOperators(H)
    Hs = (qubit_hilbert_space(1:1), qubit_hilbert_space(2:2), qubit_hilbert_space(3:3))
    v = [QuantumDots.indtofock(i, H) for i in 1:8]
    t1 = reshape(v, H, Hs)
    t2 = [i1 + 2i2 + 4i3 for i1 in (0, 1), i2 in (0, 1), i3 in (0, 1)]
    @test t1 == FockNumber.(t2)

    v2 = rand(8)
    H1 = qubit_hilbert_space(1:1)
    H2 = qubit_hilbert_space(2:3)
    H = wedge(H1, H2)
    @test !QuantumDots.isfermionic(H)
    @test sort(svdvals(reshape(v2, H, (H1, H2))) .^ 2) ≈ eigvals(partial_trace(v2, H, H1))

    ## Test that partial trace is the adjoint of embedding
    using LinearMaps
    ptmap = LinearMap(rhovec -> vec(partial_trace(reshape(rhovec, size(H)), H, H1)), prod(size(H1)), prod(size(H)))
    embeddingmap = LinearMap(rhovec -> vec(fermionic_embedding(reshape(rhovec, size(H1)), H1, H)), prod(size(H)), prod(size(H1)))
    @test Matrix(ptmap) ≈ Matrix(embeddingmap)'

end