struct QArray{N,QNs,A,S}
    blocks::Dictionary{QNs,A}
    dirs::NTuple{N,Bool}
    symmetry::NTuple{N,S}
    function QArray(qns::Vector{QNs}, blocks::Vector{A}, sym::NTuple{N,S},dirs = ntuple(i->false,N)) where {A<:AbstractArray,QNs, S,N}
        @assert length(qns) == length(blocks)
        @assert ndims(A) == length(first(qns)) == N
        new{ndims(A) ,QNs,A, S}(Dictionary(qns,blocks), dirs, sym)
    end
end
QArray(qns::Vector{NTuple{2,QN}}, blocks::Vector{<:AbstractMatrix}, sym::NTuple{2}; dirs = (false, true)) where QN = QArray(qns,blocks, sym,dirs)

Base.getindex(a::QArray, qns::AbstractQuantumNumber...) = getindex(a,qns)
Base.getindex(a::QArray, qns::NTuple{N,<:AbstractQuantumNumber}) where N = a.blocks[qns]
function Base.getindex(a::QArray, qninds::NTuple{N,<:QNIndex}) where N 
    qns = map(qn, qninds)
    inds = map(ind, qninds)
    a.blocks[qns][inds...]
end

qns(a::QArray) = collect(keys(a.blocks))

Base.adjoint(v::QArray{1}) = QArray(qns(v), collect(conj.(v.blocks)), v.symmetry, map(!,v.dirs))
Base.adjoint(m::QArray{2}) = QArray(qns(m), collect(adjoint.(m.blocks)), reverse(m.symmetry), reverse(map(!,m.dirs)))

function Base.:*(M::QArray{2}, v::QArray{1})
    @assert last(M.dirs) != only(v.dirs) "Directions don't match"
    matches, finalqns = findqnmatches(qns(M),(2,), qns(v),(1,))
    blocksizes = [blocksize(qns[1], M.symmetry[1]) for qns in finalqns]
    outblocks = [zeros(eltype(v), bz) for bz in blocksizes]
    outdirs = (first(M.dirs),)
    outsym = (first(M.symmetry),)
    outv = QArray(finalqns, outblocks, outsym, outdirs)
    for (qnsM,qnsv) in matches
        finalqn = (first(qnsM),)
        outv[finalqn] .+= M[qnsM]*v[qnsv]
    end
    return outv
end
function Base.:*(v::QArray{1}, M::QArray{2})
    @assert first(M.dirs) != only(v.dirs) "Directions don't match"
    matches, finalqns = findqnmatches(qns(M),(1,), qns(v),(1,))
    blocksizes = [blocksize(qns[1], M.symmetry[2]) for qns in finalqns]
    outblocks = [zeros(eltype(v), bz) for bz in blocksizes]
    outdirs = (last(M.dirs),)
    outsym = (last(M.symmetry),)
    outv = QArray(finalqns, outblocks, outsym, outdirs)
    for (qnsM,qnsv) in matches
        finalqn = (last(qnsM),)
        outv[finalqn] .+= parent(transpose(v[qnsv])*M[qnsM])
    end
    return outv
end
function Base.:*(v1::QArray{1}, v2::QArray{1})
    @assert only(v1.dirs) != only(v2.dirs) "Directions don't match"
    matches, _ = findqnmatches(qns(v1),(1,), qns(v2),(1,))
    out = zero(promote_type(eltype(v1),eltype(v2)))
    for (qnsv1,qnsv2) in matches
        out += transpose(v1[qnsv1])*v2[qnsv2]
    end
    return out
end
function Base.:*(m1::QArray{2}, m2::QArray{2})
    @assert last(m1.dirs) != first(m2.dirs) "Directions don't match"
    matches, finalqns = findqnmatches(qns(m1),(2,), qns(m2),(1,))
    blocksizes = [(blocksize(qns[1], m1.symmetry[1]), blocksize(qns[2], m2.symmetry[2])) for qns in finalqns]
    outblocks = [zeros(promote_type(eltype(m1),eltype(m2)), bz) for bz in blocksizes]
    outdirs = (first(m1.dirs),last(m1.dirs))
    outsym = (first(m1.symmetry), last(m2.symmetry))
    outm = QArray(finalqns, outblocks, outsym, outdirs)
    for (qnsm1,qnsm2) in matches
        finalqn = (first(qnsm1),last(qnsm2))
        outm[finalqn] .+= m2[qnsm1]*m2[qnsm2]
    end
    return outv
end

Base.eltype(::QArray{<:Any,<:Any,A}) where A = eltype(A)
Base.size(a::QArray) = Base.size.(a.symmetry)

function Base.Array(a::QArray)
    out = zeros(eltype(a), size(a))
    for qn in qns(a)
        for I in CartesianIndices(a[qn])
            inds = Tuple(I)
            qninds = map(QNIndex, qn,inds)
            fullinds = map(qnindtoind,qninds,a.symmetry)
            # println(fullinds)
            out[fullinds...] = a[qninds]
        end
    end
    out
end


_select(t1,t2) = map(t->t1[t],t2)
function findqnmatches(qns1,inds1::NTuple{N,Int}, qns2, inds2::NTuple{N,Int}) where N
    matches = [(qn1, qn2) for (qn1,qn2) in Base.product(qns1,qns2) if _select(qn1,inds1) == _select(qn2,inds2)]
    finalqns = unique([_select(first(match),inds1) for match in matches]) ##FIXME: finalqns are determined by remaining indices
    return matches, finalqns
end