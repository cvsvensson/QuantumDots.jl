function khatri_rao_lazy_dissipator(L, blocksizes)
    L2 = L' * L
    inds = sizestoinds(blocksizes)
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
function khatri_rao_dissipator(L, blocksizes)
    L2 = L' * L
    inds = sizestoinds(blocksizes)
    T = eltype(L)
    newinds = sizestoinds(blocksizes .^ 2)
    N = sum(abs2, blocksizes)
    D = zeros(T, N, N)
    for k1 in eachindex(inds), k2 in eachindex(inds)
        ind1 = inds[k1]
        ind2 = inds[k2]
        Lblock = L[ind1, ind2]
        leftprodmap = Lblock
        rightprodmap = conj(Lblock)
        kron!(@view(D[newinds[k1], newinds[k2]]), rightprodmap, leftprodmap)
        if k1 == k2
            L2block = L2[ind1, ind2]
            leftsummap = L2block
            rightsummap = transpose(L2block)
            id = I(blocksizes[k2])
            D[newinds[k1], newinds[k2]] .+= -1 / 2 .* (kron(rightsummap, id) .+ kron(id, leftsummap))
        end
    end
    return D
end

function khatri_rao_lazy(L1, L2, blocksizes)
    inds = sizestoinds(blocksizes)
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
function khatri_rao(L1, L2, blocksizes)
    inds = sizestoinds(blocksizes)
    T = promote_type(eltype(L1), eltype(L2))
    maps = Matrix{T}[]
    for i in eachindex(blocksizes), j in eachindex(blocksizes)
        l1 = L1[inds[i], inds[j]]
        l2 = L2[inds[i], inds[j]]
        push!(maps, kron(l1, l2))
    end
    hvcat(length(inds), maps...)
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
khatri_rao(L1::BlockDiagonal, L2::BlockDiagonal) = cat([khatri_rao(B1, B2) for (B1, B2) in zip(blocks(L1), blocks(L2))]...; dims=(1, 2))
function khatri_rao(L1::BlockDiagonal, L2::BlockDiagonal, bz)
    if bz == first.(blocksizes(L1)) == first.(blocksizes(L2)) == last.(blocksizes(L1)) == last.(blocksizes(L2))
        return khatri_rao(L1, L2)
    else
        return khatri_rao(cat(L1.blocks...; dims=(1, 2)), cat(L2.blocks...; dims=(1, 2)), bz)
    end
end

khatri_rao_lazy_commutator(A, blocksizes) = khatri_rao_lazy(one(A), A, blocksizes) - khatri_rao_lazy(transpose(A), one(A), blocksizes)
khatri_rao_commutator(A, blocksizes) = khatri_rao(one(A), A, blocksizes) - khatri_rao(transpose(A), one(A), blocksizes)
khatri_rao_commutator(A::BlockDiagonal{<:Any,<:Diagonal}, blocksizes) = khatri_rao_commutator(Diagonal(A), blocksizes)