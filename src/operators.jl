"""
    jwstring(site,focknbr)
    
Count the number of fermions to the right of site.
"""
jwstring(site,focknbr) = (-1)^(count_ones(focknbr >> site))

siteindices(ps,bin) = map(p->siteindex(p,bin),ps)

index(basisstate::Integer,::FermionBasis) = basisstate + 1
basisstate(ind::Integer,::FermionBasis) = ind - 1


function togglefermions(digitpositions, daggers, focknbr)
    newfocknbr = 0
    allowed = true
    fermionstatistics = 1
    for (digitpos, dagger) in zip(digitpositions, daggers)
        op = 2^(digitpos - 1)
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
