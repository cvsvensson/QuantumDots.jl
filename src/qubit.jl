function qubit_sparse_matrix(qubit_number, totalsize, sym)
    mat = spzeros(Int, totalsize, totalsize)
    _fill!(mat, fs -> lower_qubit(qubit_number, fs), sym)
    mat
end

function lower_qubit(digitposition, statefocknbr)
    cdag = focknbr(digitposition)
    newfocknbr = cdag ⊻ statefocknbr
    allowed = !iszero(cdag & statefocknbr)
    return allowed * newfocknbr, allowed
end

struct QubitBasis{M,S,T,Sym} <: AbstractManyBodyBasis
    dict::Dictionary{S,T}
    symmetry::Sym
    function QubitBasis(qubits, sym::Sym) where {Sym<:AbstractSymmetry}
        M = length(qubits)
        S = eltype(qubits)
        reps = ntuple(n -> qubit_sparse_matrix(n, 2^M, sym), M)
        new{M,S,eltype(reps),Sym}(Dictionary(qubits, reps), sym)
    end
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
labels(b::QubitBasis) = keys(b).values
Base.show(io::IO, ::MIME"text/plain", b::QubitBasis) = show(io, b)
Base.show(io::IO, b::QubitBasis{M,S,T,Sym}) where {M,S,T,Sym} = print(io, "QubitBasis{$M,$S,$T,$Sym}:\nkeys = ", keys(b))
Base.iterate(b::QubitBasis) = Base.iterate(b.dict)
Base.iterate(b::QubitBasis, state) = Base.iterate(b.dict, state)
Base.length(::QubitBasis{M}) where {M} = M
QubitBasis(iters...; qn=NoSymmetry()) = QubitBasis(Base.product(iters...), symmetry(Base.product(iters...), qn))
QubitBasis(iter; qn=NoSymmetry()) = QubitBasis(iter, symmetry(iter, qn))
symmetry(basis::QubitBasis) = basis.symmetry



function partial_trace!(mout, m::AbstractMatrix{T}, labels, b::QubitBasis{M}, sym::AbstractSymmetry=NoSymmetry()) where {T,M}
    N = length(labels)
    fill!(mout, zero(eltype(mout)))
    outinds::NTuple{N,Int} = siteindices(labels, b)
    bitmask = 2^M - 1 - focknbr(outinds)
    outbits(f) = map(i -> _bit(f, i), outinds)
    for f1 in UnitRange{UInt64}(0, 2^M - 1), f2 in UnitRange{UInt64}(0, 2^M - 1)
        if (f1 & bitmask) != (f2 & bitmask)
            continue
        end
        newfocknbr1 = focknbr(outbits(f1))
        newfocknbr2 = focknbr(outbits(f2))
        mout[focktoind(newfocknbr1, sym), focktoind(newfocknbr2, sym)] += m[focktoind(f1, b), focktoind(f2, b)]
    end
    return mout
end

function bloch_vector(ρ::AbstractMatrix, label, basis::QubitBasis)
    map(op->tr(ρ*basis[label, op]), [:X, :Y, :Z])
end
