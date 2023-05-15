struct QArray{N,QNs,A,S}
    blocks::Dictionary{QNs,A}
    symmetry::NTuple{N,S}
    dirs::NTuple{N,Bool}
    function QArray(blocks::Dictionary{QNs,A}, sym,dirs = ntuple(i->false,N)) where {A<:AbstractArray,QNs}
        S = eltype(syms)
        N = ndims(A)
        new{ndims(A) ,QNs,A, S}(blocks, sym, dirs)
    end
end
function QArray(qns::Vector, blocks::Vector, sym::NTuple{N,<:},dirs = ntuple(i->false,N)) where {N}
    @assert length(qns) == length(blocks)
    @assert ndims(A) == length(first(qns)) == N
    QArray(Dictionary(qns,blocks), sym, dirs)
end
QArray(qns::Vector{NTuple{2,QN}}, blocks::Vector{<:AbstractMatrix}, sym::NTuple{2}; dirs = (false, true)) where QN = QArray(qns,blocks, sym,dirs)
symmetry(A::QArray) = A.symmetry
dirs(A::QArray) = A.dirs
Base.getindex(a::QArray, qns::AbstractQuantumNumber...) = getindex(a,qns)
Base.getindex(a::QArray, qns::NTuple{N,<:AbstractQuantumNumber}) where N = a.blocks[qns]
function Base.getindex(a::QArray, qninds::NTuple{N,<:QNIndex}) where N 
    qns = map(qn, qninds)
    inds = map(ind, qninds)
    a.blocks[qns][inds...]
end

qns(a::QArray) = keys(a.blocks).values

Base.adjoint(v::QArray{1}) = QArray(map(conj, v.blocks), v.symmetry, map(!,v.dirs))
# Base.adjoint(m::QArray{2}) = QArray(map(adjoint, m.blocks), reverse(m.symmetry), reverse(map(!,m.dirs)))
Base.adjoint(m::QArray{2}) = QArray(map(reverse,keys(m.blocks).values), map(adjoint, m.blocks.values), reverse(m.symmetry), reverse(map(!,m.dirs)))

tuplecat(t1,t2) = (t1...,t2...)
# function contract(a1::QArray, oind1,cind1,a2::QArray,oind2,cind2)
#     Tout = promote_type(eltype(a1),eltype(a2))
#     selo1 = Base.Fix2(_select, oind1)
#     selc1 = Base.Fix2(_select, cind1)
#     selo2 = Base.Fix2(_select, oind2)
#     selc2 = Base.Fix2(_select, cind2)
#     @assert selc1(a1.dirs) != selc2(a2.dirs) "Directions don't match"
#     matches = findqnmatches(qns(a1), cind1, qns(a2), cind2)
#     finalqns12 = unique((_select(first(match), oind1), _select(last(match), oind2)) for match in matches) ##FIXME: finalqns are determined by remaining indices
#     finalqns = map(qns -> tuplecat(qns...),finalqns12)
#     outsym1 = selo1(a1.symmetry)
#     outsym2 = selo2(a2.symmetry)
#     outsym = tuplecat(outsym1, outsym2)
#     blocksizes = [tuplecat(map(blocksize, qns1, outsym1),
#      map(blocksize, qns2, outsym2))  for (qns1,qns2) in finalqns12]
#     outblocks = map(Base.Fix1(zeros,Tout),blocksizes)
#     outdirs = tuplecat(selo1(a1.dirs), selo2(a2.dirs))
#     outv = QArray(finalqns, outblocks, outsym, outdirs)
    
#     outinds1 = ntuple(identity,length(oind1))
#     outinds2 = ntuple(i->length(oind1) +i,length(oind2))
#     outinds = tuplecat(outinds1,outinds2)
#     for (qns1,qns2) in matches
#         finalqn = tuplecat(selo1(qns1),selo2(qns2))
#         # println(a1[qns1])
#         # println(a2[qns2])
#         @tullio outv[finalqn][outinds...] += a1[qns1][outinds1...]*a2[qns2][outinds1...]
#         # outv[finalqn] .+= a1[qns1]*a2[qns2]
#         #expr = contraction(oind1,cind1,oind2,cind2)
#         #println(expr)
#         #eval(expr)

#     end
#     return outv
# end

# function Base.:*(a1::QArray{2}, a2::QArray{1})
#     cind1 = (2,)
#     oind1 = (1,) #Tuple(setdiff(ntuple(identity,N),inds1))
#     cind2 = (1,)
#     oind2 = ()
#     return contract(a1,oind1,cind1,a2,oind2,cind2)
# end
# Base.:*(a1::QArray{N1}, a2::QArray{N2}) where {N1,N2} = contract(a1,ntuple(identity,N1-1),(N1,),a2,ntuple(i->i+1,N2-1),(1,))

Base.:*(x::Number, A::QArray) = QArray(map(Base.Fix1(*,x),A.blocks), symmetry(A), dirs(A))
Base.:*(A::QArray, x::Number) = QArray(map(Base.Fix2(*,x),A.blocks), symmetry(A), dirs(A))
function findmatches(a1::QArray, oind1,cind1,a2::QArray,oind2,cind2)
    Tout = promote_type(eltype(a1),eltype(a2))
    selo1 = Base.Fix2(_select, oind1)
    selc1 = Base.Fix2(_select, cind1)
    selo2 = Base.Fix2(_select, oind2)
    selc2 = Base.Fix2(_select, cind2)
    @assert selc1(a1.dirs) != selc2(a2.dirs) "Directions don't match"
    matches = findqnmatches(qns(a1), cind1, qns(a2), cind2)
    finalqns12 = unique((_select(first(match), oind1), _select(last(match), oind2)) for match in matches) ##FIXME: finalqns are determined by remaining indices
    finalqns = map(qns -> tuplecat(qns...),finalqns12)
    outsym1 = selo1(a1.symmetry)
    outsym2 = selo2(a2.symmetry)
    outsym = tuplecat(outsym1, outsym2)
    blocksizes = [tuplecat(map(blocksize, qns1, outsym1),
        map(blocksize, qns2, outsym2))  for (qns1,qns2) in finalqns12]
    outblocks = map(Base.Fix1(zeros,Tout),blocksizes)
    outdirs = tuplecat(selo1(a1.dirs), selo2(a2.dirs))
    out = QArray(finalqns, outblocks, outsym, outdirs)

    # outinds1 = ntuple(identity,length(oind1))
    # outinds2 = ntuple(i->length(oind1) +i,length(oind2))
    # outinds = tuplecat(outinds1,outinds2)
    return out, matches
end

function Base.:*(v::QArray{1}, M::QArray{2})
    outv, matches = findmatches(v,(),(1,),M,(2,),(1,))
    for (qnsv,qnsM) in matches
        finalqn = (last(qnsM),)
        outv[finalqn] .+= parent(transpose(v[qnsv])*M[qnsM])
    end
    return outv
end
function Base.:*(M::QArray{2}, v::QArray{1})
    outv, matches = findmatches(M,(1,),(2,),v,(),(1,))
    for (qnsM,qnsv) in matches
        finalqn = (first(qnsM),)
        outv[finalqn] .+= parent(M[qnsM]*v[qnsv])
    end
    return outv
end

function Base.:*(v1::QArray{1}, v2::QArray{1})
    @assert only(v1.dirs) != only(v2.dirs) "Directions don't match"
    matches = findqnmatches(qns(v1),(1,), qns(v2),(1,))
    out = zero(promote_type(eltype(v1),eltype(v2)))
    for (qnsv1,qnsv2) in matches
        out += transpose(v1[qnsv1])*v2[qnsv2]
    end
    return out
end
function Base.:*(m1::QArray{2}, m2::QArray{2})
    outm, matches = findmatches(m1,(1,),(2,),m2,(2,),(1,))
    for (qnsm1,qnsm2) in matches
        finalqn = (first(qnsm1),last(qnsm2))
        outm[finalqn] .+= m1[qnsm1]*m2[qnsm2]
    end
    return outm
end

Base.eltype(::QArray{<:Any,<:Any,A}) where A = eltype(A)
Base.size(a::QArray) = qnsize.(a.symmetry)
Base.ndims(::QArray{N}) where N = N
function Base.Array(a::QArray)
    # out = Array{eltype(a), ndims(a)}(undef, size(a))
    out = zeros(eltype(a), size(a))
    for qn in qns(a)
        for I in CartesianIndices(a[qn])
            inds = Tuple(I)
            qninds = map(QNIndex, qn,inds)
            fullinds = map(qnindtoind,qninds,a.symmetry)
            out[fullinds...] = a[qninds]
        end
    end
    out
end


_select(t1,t2) = map(t->t1[t],t2)
function findqnmatches(qns1,inds1::NTuple{N,Int}, qns2, inds2::NTuple{N,Int}) where N
    return [(qn1, qn2) for (qn1,qn2) in Base.product(qns1,qns2) if _select(qn1,inds1) == _select(qn2,inds2)]
end

isblockdiagonal(A::QArray{2}) = all((==)(q...) for q in qns(A))
