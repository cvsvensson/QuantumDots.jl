"""
    jwstring(site, focknbr)
    
Count the number of fermions to the right of site.
"""
jwstring(site, focknbr) = jwstring_right(site, focknbr)
jwstring_right(site, focknbr) = iseven(count_ones(focknbr >> site)) ? 1 : -1
jwstring_left(site, focknbr) = iseven(count_ones(focknbr) - count_ones(focknbr >> (site - 1))) ? 1 : -1


"""
    removefermion(digitposition, statefocknbr)

Return (newfocknbr, fermionstatistics) where `newfocknbr` is the state obtained by removing a fermion at `digitposition` from `statefocknbr` and `fermionstatistics` is the phase from the Jordan-Wigner string, or 0 if the operation is not allowed.
"""
function removefermion(digitposition, statefocknbr)
    cdag = focknbr_from_site_index(digitposition)
    newfocknbr = cdag ⊻ statefocknbr
    allowed = !iszero(cdag & statefocknbr)
    fermionstatistics = jwstring_right(digitposition, statefocknbr)
    return allowed * newfocknbr, allowed * fermionstatistics
end

"""
    parityoperator(basis::AbstractBasis)

Return the parity operator of `basis`.
"""
function parityoperator(basis::AbstractBasis)
    mat = spzeros(Int, 2^length(basis), 2^length(basis))
    _fill!(mat, fs -> (fs, parity(fs)), basis.symmetry)
    return mat
end

"""
    numberoperator(basis::FermionBasis)

Return the number operator of `basis`.
"""
function numberoperator(basis::FermionBasis)
    mat = spzeros(Int, 2^length(basis), 2^length(basis))
    _fill!(mat, fs -> (fs, fermionnumber(fs)), basis.symmetry)
    return mat
end

function _fill!(mat, op, ::NoSymmetry)
    for ind in axes(mat, 2)
        newfockstate, amp = op(ind - 1)
        newind = newfockstate + 1
        mat[newind, ind] += amp
    end
    return mat
end

function _fill!(mat, op, sym::AbelianFockSymmetry)
    for ind in axes(mat, 2)
        newfockstate, amp = op(indtofock(ind, sym))
        newind = focktoind(newfockstate, sym)
        mat[newind, ind] += amp
    end
    return mat
end

function togglefermions(digitpositions, daggers, focknbr)
    newfocknbr = 0
    allowed = true
    fermionstatistics = 1
    for (digitpos, dagger) in zip(digitpositions, daggers)
        op = 1 << (digitpos - 1) #2^(digitpos - 1) but faster
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
