focknbr(bits::Union{BitVector,Vector{Bool}}) = mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits))
focknbr(sites::Vector{<:Integer}) = mapreduce(site -> 2^(site-1),+, sites)
focknbr(sites::NTuple{N,<:Integer}) where N = mapreduce(site -> 2^(site-1),+, sites)
focknbr(sites::Vector{<:Integer},cell_length, species_index=1) = mapreduce(site->2^(digitposition(site,cell_length,species_index)-1),+, sites)
bits(s::Integer,N) = digits(Bool,s, base=2, pad=N)

Base.adjoint(f::Fermion) = CreationOperator((f,),(true,))
Fermion(args...) = Fermion(args)
FermionBasis(N::Integer) = FermionBasis(ntuple(i->i,N))
particles(b::FermionBasis) = Dict(zip(b.ids,Fermion.(b.ids)))
Base.eltype(::Fermion) = Int

FermionBasis(chainlength::Integer,species) = FermionBasis(Tuple(Base.product(1:chainlength,species)))
FermionBasis(chainlength::Integer,species::Symbol) = FermionBasis(ntuple(i->(i,species),chainlength))
nbr_of_fermions(::FermionBasis{M}) where M = M
Base.length(b::FermionBasis) = 2^nbr_of_fermions(b)
siteindex(f::Fermion,b::FermionBasis) = findfirst(x->x==f.id,b.ids)::Int
Base.eachindex(b::FermionBasis) = 1:length(b)

function addfermion(digitpositions,state) #Currently only works for a single creation operator
    cdag = focknbr(digitpositions)
    newfocknbr = cdag | state
    allowed = iszero(cdag & state) && allunique(digitpositions) 
    fermionstatistics = jwstring(digitpositions[1],state) 
    return allowed * newfocknbr, allowed * fermionstatistics
end