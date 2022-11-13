abstract type AbstractOperator end

"""
    jwstring(site,focknbr)
    
Count the number of fermions to the right of site.
"""
jwstring(site,focknbr) = (-1)^(count_ones(focknbr >> site))
jwstring(f::Fermion{S},::Val{S2},focknbr) where {S,S2} = jwstring(f.site + (S2 < S ? 1 : 0), focknbr) #Assuming we species are ordered by unicode order
jwstring(site,f::FermionBasisState{S},::Val{S2}) where {S,S2} = jwstring(site + (S2 < S ? 1 : 0), focknbr(f,S2)) #Assuming we species are ordered by unicode order
function jwstring(f::Fermion{S},focknbrs,::FermionFockBasis{S2}) where {S,S2}
    prod(fs->jwstring(f,fs[2],fs[1]), zip(focknbrs,S2))
end

site_offsets(::Val{S},::Val{SS}) where {S,SS} = map(s->comp_isless(Val(s),Val(S)) ? 0 : 1,SS)
function jwstring(f::Fermion{S},ψ::FermionBasisState{S2}) where {S,S2} #This is allocation-less
    offsets = site_offsets(Val(S),Val(S2))
    prod(ov->jwstring(f.site +ov[1], ov[2]), zip(offsets,focknbrs(ψ)))
end

function jwstring2(f::Fermion{S},ψ::FermionBasisState{S2}) where {S,S2} #This allocates
    offsets = S2 .< S
    prod(ov->jwstring(f.site +ov[1], ov[2]), zip(offsets,focknbrs(ψ)))
end

@inline @generated function comp_isless(::Val{s1},::Val{s2}) where {s1,s2}
    b = s1<s2
    :($b)
end

struct CreationOperator{P} <: AbstractOperator
    particle::P
end
species(c::CreationOperator) = species(c.particle)

Base.:*(Cdag::CreationOperator{Fermion{P}}, state::FermionBasisState) where P = addfermion(Cdag.site, focknbr(state,P))


struct FermionState{M,Tv,Ti}
    sparsevector::SparseVector{Tv,Ti}
    species::NTuple{M,Symbol}
end


function addfermion(site::Integer,focknbr)
    cdag = 2^(site-1)
    newfocknbr = cdag | focknbr
    # Check if there already was a fermion at the site. 
    allowed = iszero(cdag & focknbr) # or maybe count_ones(newfocknbr) == 1 + count_ones(focknbr)? 
    fermionstatistics = jwstring(site,focknbr) #1 or -1, depending on the nbr of fermions to the right of site
    return newfocknbr, allowed * fermionstatistics
end
