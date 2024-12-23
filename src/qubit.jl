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

struct QubitBasis{M,D,Sym} <: AbstractManyBodyBasis
    dict::D
    symmetry::Sym
end
function QubitBasis(iters...; qn = NoSymmetry(), kwargs...) 
    labels = handle_labels(iters...)
    M = length(labels)
    labelled_symmetry = instantiate(qn, labels)
    fockstates = get(kwargs, :fockstates, 0:2^M-1)
    sym_concrete = focksymmetry(fockstates, labelled_symmetry)
    reps = ntuple(n -> qubit_sparse_matrix(n, length(fockstates), sym_concrete), M)
    d = OrderedDict(zip(labels, reps))
    QubitBasis{M,typeof(d),typeof(sym_concrete)}(d, sym_concrete)
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


function partial_trace!(mout, m::AbstractMatrix{T}, labels, b::QubitBasis{M}, sym::AbstractSymmetry=NoSymmetry()) where {T,M}
    N = length(labels)
    fill!(mout, zero(eltype(mout)))
    outinds::NTuple{N,Int} = siteindices(labels, b)
    bitmask = 2^M - 1 - focknbr_from_site_indices(outinds)
    outbits(f) = map(i -> _bit(f, i), outinds)
    for f1 in UnitRange{UInt64}(0, 2^M - 1), f2 in UnitRange{UInt64}(0, 2^M - 1)
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
