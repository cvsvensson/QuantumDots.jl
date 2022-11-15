abstract type AbstractOperator end

"""
    jwstring(site,focknbr)
    
Count the number of fermions to the right of site.
"""
jwstring(site,focknbr) = (-1)^(count_ones(focknbr >> site))

struct CreationOperator{P} <: AbstractOperator end
CreationOperator(::P) where P = CreationOperator{P}()
FermionCreationOperator(id::Symbol) = CreationOperator{Fermion{id}}()
particle(::CreationOperator{P}) where P = P()

function addfermion(digitpos::Integer,focknbr)
    cdag = 2^(digitpos-1)
    newfocknbr = cdag | focknbr
    # Check if there already was a fermion at the site. 
    allowed = iszero(cdag & focknbr) # or maybe count_ones(newfocknbr) == 1 + count_ones(focknbr)? 
    fermionstatistics = jwstring(digitpos,focknbr) #1 or -1, depending on the nbr of fermions to the right of site
    return ((newfocknbr, allowed * fermionstatistics),)
end
