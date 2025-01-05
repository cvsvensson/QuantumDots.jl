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
    kroncaches = materialize_kroncache(T, kv)
    khatri_rao_dissipator!(out, L, rate, kv, kroncaches, mulcache, ADD_SUPEROP())
    return out
end

# _kroncache_subblock(kroncaches::AbstractVector, k) = kroncache[k]
_kroncache_subblock(kroncaches::AbstractMatrix, k1, k2) = kroncaches[k1, k2]
function materialize_kroncache(::Type{T}, kv::KhatriRaoVectorizer) where {T}
    return [zeros(T, length(ind1), length(ind2)) for ind1 in kv.vectorinds, ind2 in kv.vectorinds]
end
function khatri_rao_dissipator!(out, L::AbstractMatrix, rate, kv::KhatriRaoVectorizer, kroncaches, mulcache, mode::ADD_SUPEROP)
    mul!(mulcache, L', L, 1 / 2, 0)
    L2 = mulcache
    inds = kv.inds
    blocksizes = kv.sizes
    newinds = kv.vectorinds
    for k1 in eachindex(inds), k2 in eachindex(inds)
        ind1 = inds[k1]
        ind2 = inds[k2]
        Lblock = @view(L[ind1, ind2])
        kroncache = _kroncache_subblock(kroncaches, k1, k2)
        # kron!(@view(out[newinds[k1], newinds[k2]]), transpose(Lblock'), Lblock)
        kron!(kroncache, transpose(Lblock'), Lblock)
        for (n1, m1) in enumerate(newinds[k1]), (n2, m2) in enumerate(newinds[k2])
            out[m1, m2] += rate * kroncache[n1, n2] #remove + if mode is RESET_SUPEROP
        end
        if k1 == k2
            L2block = @view(L2[ind1, ind2])
            # id = I(blocksizes[k2])
            id = Eye(blocksizes[k2])
            # kroncache = _kroncache_subblock(kroncaches, k1, k2)
            kron!(kroncache, transpose(L2block), id)
            for (n1, m1) in enumerate(newinds[k1]), (n2, m2) in enumerate(newinds[k2])
                out[m1, m2] -= rate * kroncache[n1, n2]
            end
            kron!(kroncache, id, L2block)
            for (n1, m1) in enumerate(newinds[k1]), (n2, m2) in enumerate(newinds[k2])
                out[m1, m2] -= rate * kroncache[n1, n2]
            end
            # out[newinds[k1], newinds[k2]] .-= kroncache
        end
    end
    return out #.*= rate
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

function khatri_rao(L1::Diagonal{T1}, L2::Diagonal{T2}, kv::KhatriRaoVectorizer) where {T1,T2}
    inds = kv.inds
    T = promote_type(T1, T2)
    indsout = kv.vectorinds
    d = zeros(T, last(kv.cumsumsquared))
    for (inds, indsout) in zip(inds, indsout)
        l1 = Diagonal(@view(L1.diag[inds]))
        l2 = Diagonal(@view(L2.diag[inds]))
        kron!(Diagonal(@view(d[indsout])), l1, l2)
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

khatri_rao_lazy_commutator(A, blocksizes) = khatri_rao_lazy(kr_one(A), A, blocksizes) - khatri_rao_lazy(transpose(A), kr_one(A), blocksizes)
khatri_rao_commutator(A, blocksizes) = khatri_rao(kr_one(A), A, blocksizes) - khatri_rao(transpose(A), kr_one(A), blocksizes)
khatri_rao_commutator(A::BlockDiagonal{<:Any,<:Diagonal}, blocksizes) = khatri_rao_commutator(Diagonal(A), blocksizes)

kr_one(m::BlockDiagonal) = BlockDiagonal(kr_one.(blocks(m)))
kr_one(m) = Eye{eltype(m)}(size(m, 1))
