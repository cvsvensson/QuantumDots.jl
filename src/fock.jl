focknbr(bits::Union{BitVector,Vector{Bool}}) = mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits))
focknbr(site::Integer) = 2^(site-1)
focknbr(sites::Vector{<:Integer}) = mapreduce(focknbr,+, sites)
focknbr(sites::NTuple{N,<:Integer}) where N = mapreduce(site -> 2^(site-1),+, sites)
focknbr(sites::Vector{<:Integer},cell_length, species_index=1) = mapreduce(site->2^(digitposition(site,cell_length,species_index)-1),+, sites)
bits(s::Integer,N) = digits(Bool,s, base=2, pad=N)
parity(fs::Int) = (-1)^count_ones(fs)
fermionnumber(fs::Int) = count_ones(fs)

siteindex(id::S,b::FermionBasis{<:Any,S}) where S = findfirst(x->x==id,keys(b.dict))::Int
siteindex(ids::Union{NTuple{<:Any,S},Vector{S}},b::FermionBasis{<:Any,S}) where S = map(id->siteindex(id,b),ids)::Int

