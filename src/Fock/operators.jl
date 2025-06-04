"""
    removefermion(digitposition, f::FockNumber)

Return (newfocknbr, fermionstatistics) where `newfocknbr` is the state obtained by removing a fermion at `digitposition` from `f` and `fermionstatistics` is the phase from the Jordan-Wigner string, or 0 if the operation is not allowed.
"""
function removefermion(digitposition, f::FockNumber)
    cdag = focknbr_from_site_index(digitposition)
    newfocknbr = cdag ⊻ f
    allowed = !iszero(cdag & f)
    fermionstatistics = jwstring(digitposition, f)
    return allowed * newfocknbr, allowed * fermionstatistics
end

"""
    parityoperator(H::AbstractFockHilbertSpace)

Return the parity operator of `H`.
"""
function parityoperator(H::AbstractFockHilbertSpace)
    fs = focknumbers(H)
    N = length(fs)
    mat = spzeros(Int, N, N)
    _fill!(mat, fs -> (fs, parity(fs)), H)
    return mat
end

"""
    numberoperator(basis::AbstractFockHilbertSpace)

Return the number operator of `H`.
"""
function numberoperator(H::AbstractFockHilbertSpace)
    fs = focknumbers(H)
    N = length(fs)
    mat = spzeros(Int, N, N)
    _fill!(mat, fs -> (fs, fermionnumber(fs)), H)
    return mat
end

function _fill!(mat, op, H::AbstractFockHilbertSpace)
    for ind in axes(mat, 2)
        newfockstate, amp = op(indtofock(ind, H))
        newind = focktoind(newfockstate, H)
        mat[newind, ind] += amp
    end
    return mat
end

function togglefermions(digitpositions, daggers, focknbr)
    newfocknbr = 0
    allowed = true
    fermionstatistics = 1
    for (digitpos, dagger) in zip(digitpositions, daggers)
        op = FockNumber(1 << (digitpos - 1)) #2^(digitpos - 1) but faster
        if dagger
            newfocknbr = op | focknbr
            # Check if there already was a fermion at the site.
            allowed = iszero(op & focknbr)
        else
            newfocknbr = op ⊻ focknbr
            # Check if the site was empty.
            allowed = !iszero(op & focknbr)
        end
        # return directly if we create/annihilate an occupied/empty state
        if !allowed
            return newfocknbr, allowed * fermionstatistics
        end
        fermionstatistics *= jwstring(digitpos, focknbr)
        focknbr = newfocknbr
    end
    # fermionstatistics better way?
    return newfocknbr, allowed * fermionstatistics
end


"""
    fermion_sparse_matrix(fermion_number, H::AbstractFockHilbertSpace)

Constructs a sparse matrix of size representing a fermionic annihilation operator at bit position `fermion_number` on the Hilbert space H. 
"""
function fermion_sparse_matrix(fermion_number, H::AbstractFockHilbertSpace)
    fs = focknumbers(H)
    N = length(fs)
    amps = Int[]
    ininds = Int[]
    outinds = Int[]
    sizehint!(amps, N)
    sizehint!(ininds, N)
    sizehint!(outinds, N)
    for f in fs
        n = focktoind(f, H)
        newfockstate, amp = removefermion(fermion_number, f)
        if !iszero(amp)
            push!(amps, amp)
            push!(ininds, n)
            push!(outinds, focktoind(newfockstate, H))
        end
    end
    return sparse(outinds, ininds, amps, N, N)
end


@testitem "Parity and number operator" begin
    using LinearAlgebra
    using QuantumDots: FockHilbertSpace, parityoperator, numberoperator, SymmetricFockHilbertSpace, fermion_sparse_matrix
    numopvariant(H) = sum(l -> fermion_sparse_matrix(l, H)' * fermion_sparse_matrix(l, H), 1:2)
    H = FockHilbertSpace(1:2)
    @test parityoperator(H) == Diagonal([1, -1, -1, 1])
    @test numberoperator(H) == Diagonal([0, 1, 1, 2]) == numopvariant(H)

    H = SymmetricFockHilbertSpace(1:2, ParityConservation())
    @test parityoperator(H) == Diagonal([-1, -1, 1, 1])
    @test numberoperator(H) == Diagonal([1, 1, 0, 2]) == numopvariant(H)

    H = SymmetricFockHilbertSpace(1:2, FermionConservation())
    @test parityoperator(H) == Diagonal([1, -1, -1, 1])
    @test numberoperator(H) == Diagonal([0, 1, 1, 2]) == numopvariant(H)

    ## Truncated Hilbert space
    focknumbers = map(FockNumber, 0:2)
    H = FockHilbertSpace(1:2, focknumbers)
    @test parityoperator(H) == Diagonal([1, -1, -1])
    @test numberoperator(H) == Diagonal([0, 1, 1])
    H = SymmetricFockHilbertSpace(1:2, ParityConservation(), focknumbers)
    @test parityoperator(H) == Diagonal([-1, -1, 1])
    @test numberoperator(H) == Diagonal([1, 1, 0])
    H = SymmetricFockHilbertSpace(1:2, FermionConservation(), focknumbers)
    @test parityoperator(H) == Diagonal([1, -1, -1])
    @test numberoperator(H) == Diagonal([0, 1, 1])

    focknumbers = map(FockNumber, 2:2)
    H = FockHilbertSpace(1:2, focknumbers)
    @test parityoperator(H) == Diagonal([-1])
    @test numberoperator(H) == Diagonal([1])
    H = SymmetricFockHilbertSpace(1:2, ParityConservation(), focknumbers)
    @test parityoperator(H) == Diagonal([-1])
    @test numberoperator(H) == Diagonal([1])
    H = SymmetricFockHilbertSpace(1:2, FermionConservation(), focknumbers)
    @test parityoperator(H) == Diagonal([-1])
    @test numberoperator(H) == Diagonal([1])

end
