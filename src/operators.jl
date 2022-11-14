abstract type AbstractOperator end

"""
    jwstring(site,focknbr)
    
Count the number of fermions to the right of site.
"""
jwstring(site,focknbr) = (-1)^(count_ones(focknbr >> site))
jwstring(f::Fermion{S},ψ::FermionBasisState{S2}) where {S,S2} = jwstring(f,Val(S2),focknbr(ψ))
jwstring(f::Fermion{S},::Val{S2},focknbr) where {S,S2} = jwstring(digitposition(f,Val(S2)),focknbr) 

struct CreationOperator{P} <: AbstractOperator
    particle::P
end
CreationOperator(i::Integer,S::Symbol) = CreationOperator(Fermion{S}(i))
CreationOperator(S::Symbol,i::Integer) = CreationOperator(i,S)
CreationOperator{S}(i::Integer) where S = CreationOperator{Fermion{S}}(Fermion{S}(i))
species(c::CreationOperator) = species(c.particle)

Base.:*(Cdag::CreationOperator{Fermion{S}}, state::FermionBasisState{SS}) where {S,SS} = addfermion(digitposition(Cdag.particle,Val(SS)), focknbr(state))
digitposition(site::Integer,cell_length::Integer,species_index::Integer) = (site-1)*cell_length + species_index
digitposition(f::Fermion{S},::Val{SS}) where {S,SS} =  digitposition(f.site,length(SS),cellindex(Val(S),Val(SS)))


function addfermion(digitpos::Integer,focknbr)
    cdag = 2^(digitpos-1)
    newfocknbr = cdag | focknbr
    # Check if there already was a fermion at the site. 
    allowed = iszero(cdag & focknbr) # or maybe count_ones(newfocknbr) == 1 + count_ones(focknbr)? 
    fermionstatistics = jwstring(digitpos,focknbr) #1 or -1, depending on the nbr of fermions to the right of site
    return newfocknbr, allowed * fermionstatistics
end
