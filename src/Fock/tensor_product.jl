
"""
    embedding_unitary(partition, fockstates, jw)

    Compute the unitary matrix that maps between the tensor embedding and the fermionic embedding in the physical subspace. 
    # Arguments
    - `partition`: A partition of the labels in `jw` into disjoint sets.
    - `fockstates`: The fock states in the basis
    - `jw`: The Jordan-Wigner ordering.
"""
function embedding_unitary(partition, fockstates, jw::JordanWignerOrdering)
    #for locally physical algebra, ie only for even operators or states of well-defined parity
    #if ξ is ordered, the phases are +1. 
    # Note that the jordan wigner modes are ordered in reverse from the labels, but this is taken care of by direction of the jwstring below
    isorderedpartition(partition, jw) || throw(ArgumentError("The partition must be ordered according to jw"))

    phases = ones(Int, length(fockstates))
    for (s, Xs) in enumerate(partition)
        mask = focknbr_from_site_labels(Xs, jw)
        for (r, Xr) in Iterators.drop(enumerate(partition), s)
            for li in Xr
                i = siteindex(li, jw)
                for (n, f) in zip(eachindex(phases), fockstates)
                    if _bit(f, i)
                        phases[n] *= jwstring_anti(i, mask & f)
                    end
                end
            end
        end
    end
    return Diagonal(phases)
end


function bipartite_embedding_unitary(X, Xbar, fockstates, jw::JordanWignerOrdering)
    #(122a)
    ispartition((X, Xbar), jw) || throw(ArgumentError("The partition must be ordered according to jw"))
    # length(X) + length(Xbar) == length(jw.labels) || throw(ArgumentError("The union of the labels in X and Xbar must be the same as the labels in jw"))
    # all(haskey(jw.ordering, k) for k in Iterators.flatten((X, Xbar))) || throw(ArgumentError("The labels in X and Xbar must be the same as the labels in jw"))
    phases = ones(Int, length(fockstates))
    mask = focknbr_from_site_labels(X, jw)
    for li in Xbar
        i = siteindex(li, jw)
        for (n, f) in zip(eachindex(phases), fockstates)
            if _bit(f, i)
                phases[n] *= jwstring_anti(i, mask & f)
            end
        end
    end
    return Diagonal(phases)
end

@testitem "Embedding unitary" begin
    # Appendix C.4
    import QuantumDots: embedding_unitary, canonical_embedding, bipartite_embedding_unitary
    using LinearAlgebra
    jw = JordanWignerOrdering(1:2)
    fockstates = sort(map(FockNumber, 0:3), by=Base.Fix2(bits, 2))

    @test embedding_unitary([[1], [2]], fockstates, jw) == I
    @test embedding_unitary([[2], [1]], fockstates, jw) == Diagonal([1, 1, 1, -1])

    # N = 3
    jw = JordanWignerOrdering(1:3)
    fockstates = sort(map(FockNumber, 0:7), by=Base.Fix2(bits, 3))
    U(p) = embedding_unitary(p, fockstates, jw)
    @test U([[1], [2], [3]]) == U([[1, 2], [3]]) == U([[1], [2, 3]]) == I

    @test U([[2], [1], [3]]) == Diagonal([1, 1, 1, 1, 1, 1, -1, -1])
    @test U([[2], [3], [1]]) == Diagonal([1, 1, 1, 1, 1, -1, -1, 1])
    @test U([[3], [1], [2]]) == Diagonal([1, 1, 1, -1, 1, -1, 1, 1])
    @test U([[3], [2], [1]]) == Diagonal([1, 1, 1, -1, 1, -1, -1, -1])
    @test U([[1], [3], [2]]) == Diagonal([1, 1, 1, -1, 1, 1, 1, -1])

    @test U([[2], [1, 3]]) == Diagonal([1, 1, 1, 1, 1, 1, -1, -1])
    @test U([[3], [1, 2]]) == Diagonal([1, 1, 1, -1, 1, -1, 1, 1])

    @test U([[1, 3], [2]]) == Diagonal([1, 1, 1, -1, 1, 1, 1, -1])
    @test U([[2, 3], [1]]) == Diagonal([1, 1, 1, 1, 1, -1, -1, 1])
end
