function QArray(qns::Vector{QNs}, blocks::Vector{A}, sym::NTuple{N,S}) where {A<:AbstractArray,QNs, S,N}
    @assert length(qns) == length(blocks)
    @assert ndims(A) == length(first(qns)) == N
    QArray(Dictionary(qns,blocks), sym)
end
QArray(qns::Vector{NTuple{2,QN}}, blocks::Vector{<:AbstractMatrix}, sym::NTuple{2}) where QN = QArray(qns, blocks, sym)
blockstructure(A::QArray) = A.blockstructure
Base.getindex(a::QArray, qns::AbstractQuantumNumber...) = getindex(a,qns)
Base.getindex(a::QArray, qns::NTuple{N,<:AbstractQuantumNumber}) where N = a.blocks[qns]
function Base.getindex(a::QArray, qninds::Vararg{<:QNIndex}) 
    qns = map(qn, qninds)
    inds = map(ind, qninds)
    a.blocks[qns][inds...]
end
function Base.setindex!(a::QArray,value, qninds::Vararg{<:QNIndex}) 
    qns = map(qn, qninds)
    inds = map(ind, qninds)
    a.blocks[qns][inds...] = value
end
function Base.setindex!(a::QArray,value, qns::Vararg{<:AbstractQuantumNumber}) 
    a.blocks[qns] = value
end
function Base.setindex!(a::QArray,value, qns::NTuple{<:Any,<:AbstractQuantumNumber}) 
    a.blocks[qns] = value
end

qns(a::QArray) = keys(a.blocks).values

Base.adjoint(v::QArray{1}) = QArray(map(conj, v.blocks), v.blockstructure)
Base.adjoint(m::QArray{2}) = QArray(map(reverse,keys(m.blocks).values), map(adjoint, m.blocks.values), reverse(m.blockstructure))

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
Base.similar(::Type{SparseMatrixCSC{T,Int64}}, dims::Dims{2}) where T = spzeros(T,dims...)
function Base.:+(A::QArray{N,QNs,TA,S}, B::QArray{N,QNs,TB,S}) where {N,QNs,TA,TB,S}
    @assert blockstructure(A) == blockstructure(B)
    TC = final_storage(TA, TB)
    finalqns = unique([qns(A)..., qns(B)...])
    blocksizes = [map(blocksize, qns, blockstructure(A)) for qns in finalqns]
    outblocks = map(Base.Fix1(similar,TC), blocksizes)
    C = QArray(finalqns, outblocks, blockstructure(A))
    for qn in qns(A)
        C[qn] += A[qn]
    end
    for qn in qns(B)
        C[qn] += B[qn]
    end
    C
end

# final_storage(::Type{A},::Type{B}) where {A,B} = promote_type(A,B)
final_storage(::Type{SparseMatrixCSC{A1,B1}},::Type{SparseMatrixCSC{A2,B2}}) where {A1,B1, A2,B2}= SparseMatrixCSC{promote_type(A1,A2),promote_type(B1,B2)}
final_storage(::Type{SparseMatrixCSC{A1,B1}},::Type{Adjoint{A2,SparseMatrixCSC{A2,B2}}}) where {A1,B1, A2,B2} = SparseMatrixCSC{promote_type(A1,A2),promote_type(B1,B2)}
final_storage(::Type{Adjoint{A2,SparseMatrixCSC{A2,B2}}},::Type{SparseMatrixCSC{A1,B1}})  where {A1,B1, A2,B2}= SparseMatrixCSC{promote_type(A1,A2),promote_type(B1,B2)}

Base.:*(x::Number, A::QArray) = QArray(map(Base.Fix1(*,x),A.blocks), blockstructure(A))
Base.:*(A::QArray, x::Number) = QArray(map(Base.Fix2(*,x),A.blocks), blockstructure(A))
function findmatches(a1::QArray, oind1,cind1,a2::QArray,oind2,cind2)
    # Tout = promote_type(eltype(a1),eltype(a2))
    Tout = final_storage(eltype(a1.blocks), eltype(a2.blocks))
    selo1 = Base.Fix2(_select, oind1)
    selc1 = Base.Fix2(_select, cind1)
    selo2 = Base.Fix2(_select, oind2)
    selc2 = Base.Fix2(_select, cind2)
    matches = findqnmatches(qns(a1), cind1, qns(a2), cind2)
    finalqns12 = unique((_select(first(match), oind1), _select(last(match), oind2)) for match in matches) ##FIXME: finalqns are determined by remaining indices
    finalqns = map(qns -> tuplecat(qns...),finalqns12)
    outsym1 = selo1(a1.blockstructure)
    outsym2 = selo2(a2.blockstructure)
    outsym = tuplecat(outsym1, outsym2)
    blocksizes = [tuplecat(map(blocksize, qns1, outsym1),
        map(blocksize, qns2, outsym2))  for (qns1,qns2) in finalqns12]
    # outblocks = map(Base.Fix1(zeros,Tout), blocksizes)
    outblocks = map(Base.Fix1(similar,Tout), blocksizes)
    out = QArray(finalqns, outblocks, outsym)

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
Base.size(a::QArray) = qnsize.(a.blockstructure)
Base.ndims(::QArray{N}) where N = N
function Base.Array(a::QArray)
    # out = Array{eltype(a), ndims(a)}(undef, size(a))
    out = zeros(eltype(a), size(a))
    for qn in qns(a)
        for I in CartesianIndices(a[qn])
            inds = Tuple(I)
            qninds = map(QNIndex, qn,inds)
            fullinds = map(qnindtoind,qninds,a.blockstructure)
            out[fullinds...] = a[qninds...]
        end
    end
    out
end


_select(t1,t2) = map(t->t1[t],t2)
function findqnmatches(qns1,inds1::NTuple{N,Int}, qns2, inds2::NTuple{N,Int}) where N
    return [(qn1, qn2) for (qn1,qn2) in Base.product(qns1,qns2) if _select(qn1,inds1) == _select(qn2,inds2)]
end

isblockdiagonal(A::QArray{2}) = all((==)(q...) for q in qns(A))


function LinearAlgebra.eigen(A::QArray)
    if isblockdiagonal(A)
    end
end