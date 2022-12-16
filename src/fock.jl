focknbr(bits::Union{BitVector,Vector{Bool}}) = mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits))
focknbr(site::Integer) = 2^(site-1)
focknbr(sites::Vector{<:Integer}) = mapreduce(focknbr,+, sites)
focknbr(sites::NTuple{N,<:Integer}) where N = mapreduce(site -> 2^(site-1),+, sites)
focknbr(sites::Vector{<:Integer},cell_length, species_index=1) = mapreduce(site->2^(digitposition(site,cell_length,species_index)-1),+, sites)
bits(s::Integer,N) = digits(Bool,s, base=2, pad=N)
parity(fs::Int) = (-1)^count_ones(fs)


# Base.length(b::FermionBasis) = 2^nbr_of_fermions(b)
# siteindex(f::Fermion,b::FermionBasis) = findfirst(x->x==f,b.fermions)::Int
