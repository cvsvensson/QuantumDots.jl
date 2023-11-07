function khatri_rao_lazy_dissipator(L, kv::KhatriRaoVectorizer)
    L2 = L' * L
    inds = kv.inds
    T = eltype(L)
    prodmaps = LinearMap{T}[]
    summaps = LinearMap{T}[]
    for ind1 in inds, ind2 in inds
        Lblock = L[ind1, ind2]
        leftprodmap = LinearMap{T}(Lblock)
        rightprodmap = LinearMap{T}(conj(Lblock))
        push!(prodmaps, kron(rightprodmap, leftprodmap))
        if ind1 == ind2
            L2block = L2[ind1, ind2]
            leftsummap = LinearMap{T}(L2block)
            rightsummap = LinearMap{T}(transpose(L2block))
            push!(summaps, kronsum(rightsummap, leftsummap))
        end
    end
    hvcat(length(inds), prodmaps...) - 1 / 2 * cat(summaps...; dims=(1, 2))
end
function khatri_rao_dissipator(L::AbstractMatrix{T}, kv::KhatriRaoVectorizer; rate=one(T)) where {T}
    N = kv.cumsumsquared[end]
    out = zeros(T, N, N)
    mulcache = zero(L)
    kroncaches = [zeros(T, length(ind), length(ind)) for ind in kv.vectorinds]
    khatri_rao_dissipator!(out, L, rate, kv, kroncaches, mulcache)
    return out
end

_kroncache_subblock(kroncache::AbstractVector, k, newinds) = kroncache[k]
_kroncache_subblock(kroncache::AbstractMatrix, k, newinds) = @view(kroncache[newinds[k], newinds[k]])
function khatri_rao_dissipator!(out, L::AbstractMatrix, rate, kv::KhatriRaoVectorizer, kroncaches, mulcache)
    mul!(mulcache, L', L, 1 / 2, 0)
    L2 = mulcache
    inds = kv.inds
    blocksizes = kv.sizes
    newinds = kv.vectorinds
    for k1 in eachindex(inds), k2 in eachindex(inds)
        ind1 = inds[k1]
        ind2 = inds[k2]
        Lblock = @view(L[ind1, ind2])
        kron!(@view(out[newinds[k1], newinds[k2]]), transpose(Lblock'), Lblock)
        if k1 == k2
            L2block = @view(L2[ind1, ind2])
            id = I(blocksizes[k2])
            kroncache = _kroncache_subblock(kroncaches, k1, newinds)
            kron!(kroncache, transpose(L2block), id)
            out[newinds[k1], newinds[k2]] .-= kroncache
            kron!(kroncache, id, L2block)
            out[newinds[k1], newinds[k2]] .-= kroncache
        end
    end
    return out .*= rate
end

function khatri_rao_lazy(L1, L2, kv::KhatriRaoVectorizer)
    blocksizes = kv.sizes
    inds = kv.inds
    T = promote_type(eltype(L1), eltype(L2))
    maps = LinearMap{T}[]
    for i in eachindex(blocksizes), j in eachindex(blocksizes)
        L1bij = L1[inds[i], inds[j]]
        L2bij = L2[inds[i], inds[j]]
        l1 = LinearMap{T}(L1bij)
        l2 = LinearMap{T}(L2bij)
        push!(maps, kron(l1, l2))
    end
    hvcat(length(inds), maps...)
end

function khatri_rao(L1::AbstractMatrix{T1}, L2::AbstractMatrix{T2}, kv::KhatriRaoVectorizer) where {T1,T2}
    T = promote_type(T1, T2)
    KR = zeros(T, kv.cumsumsquared[end], kv.cumsumsquared[end])
    khatri_rao!(KR, L1, L2, kv)
end
function khatri_rao!(KR, L1, L2, kv::KhatriRaoVectorizer)
    khatri_rao!(KR, L1, L2, kv.inds, kv.vectorinds)
end
function khatri_rao!(KR, L1, L2, inds, finalinds)
    #TODO: Put in checks
    for i in eachindex(inds), j in eachindex(inds)
        l1 = @view(L1[inds[i], inds[j]])
        l2 = @view(L2[inds[i], inds[j]])
        kron!(@view(KR[finalinds[i], finalinds[j]]), l1, l2)
    end
    return KR
end

function khatri_rao(L1::Diagonal{T1}, L2::Diagonal{T2}, blocksizes) where {T1,T2}
    inds = sizestoinds(blocksizes)
    T = promote_type(T1, T2)
    indsd = sizestoinds(blocksizes .^ 2)
    d = zeros(T, sum(abs2, blocksizes))
    for i in eachindex(blocksizes)
        l1 = Diagonal(@view(L1.diag[inds[i]]))
        l2 = Diagonal(@view(L2.diag[inds[i]]))
        kron!(Diagonal(@view(d[indsd[i]])), l1, l2)
    end
    return Diagonal(d)
end

khatri_rao(L1::Diagonal, L2::Diagonal) = kron(L1, L2)
khatri_rao(L1::BlockDiagonal, L2::BlockDiagonal) = cat([kron(B1, B2) for (B1, B2) in zip(blocks(L1), blocks(L2))]...; dims=(1, 2))
function khatri_rao(L1::BlockDiagonal, L2::BlockDiagonal, kv::KhatriRaoVectorizer)
    if kv.sizes == first.(blocksizes(L1)) == first.(blocksizes(L2)) == last.(blocksizes(L1)) == last.(blocksizes(L2))
        return khatri_rao(L1, L2)
    else
        return khatri_rao(cat(L1.blocks...; dims=(1, 2)), cat(L2.blocks...; dims=(1, 2)), kv)
    end
end

khatri_rao_lazy_commutator(A, blocksizes) = khatri_rao_lazy(one(A), A, blocksizes) - khatri_rao_lazy(transpose(A), one(A), blocksizes)
khatri_rao_commutator(A, blocksizes) = khatri_rao(one(A), A, blocksizes) - khatri_rao(transpose(A), one(A), blocksizes)
khatri_rao_commutator(A::BlockDiagonal{<:Any,<:Diagonal}, blocksizes) = khatri_rao_commutator(Diagonal(A), blocksizes)