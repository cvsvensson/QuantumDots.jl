abstract type AbstractOperator end

"""
    jwstring(site,focknbr)
    
Count the number of fermions to the right of site.
"""
jwstring(site,focknbr) = (-1)^(count_ones(focknbr >> site))
jwstring(f::Fermion{S},::Val{S2},focknbr) where {S,S2} = jwstring(digitposition(f,Val(S2)),focknbr) 

struct FermionCreationOperator{ID} <: AbstractOperator end
FermionCreationOperator(a) = FermionCreationOperator{a}()
FermionCreationOperator(args...) = FermionCreationOperator{args}()
FermionCreationOperator(::Fermion{ID}) where ID = FermionCreationOperator{ID}()

function addfermion(digitpos::Integer,focknbr)
    cdag = 2^(digitpos-1)
    newfocknbr = cdag | focknbr
    # Check if there already was a fermion at the site. 
    allowed = iszero(cdag & focknbr) # or maybe count_ones(newfocknbr) == 1 + count_ones(focknbr)? 
    fermionstatistics = jwstring(digitpos,focknbr) #1 or -1, depending on the nbr of fermions to the right of site
    return newfocknbr, allowed * fermionstatistics
end
