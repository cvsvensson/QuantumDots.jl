focknbr(bits::Union{BitVector,Vector{Bool},NTuple{<:Any,Bool}}) = mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits))
focknbr(site::Integer) = 2^(site-1)
focknbr(sites::Vector{<:Integer}) = mapreduce(focknbr,+, sites)
focknbr(sites::NTuple{N,<:Integer}) where N = mapreduce(site -> 2^(site-1),+, sites)
focknbr(sites::Vector{<:Integer},cell_length, species_index=1) = mapreduce(site->2^(digitposition(site,cell_length,species_index)-1),+, sites)
bits(s::Integer,N) = digits(Bool,s, base=2, pad=N)
parity(fs::Int) = (-1)^count_ones(fs)
fermionnumber(fs::Int) = count_ones(fs)

siteindex(id::S,b::FermionBasis{<:Any,S}) where S = findfirst(x->x==id,keys(b.dict))::Int
siteindices(ids::Union{NTuple{M,S},Vector{S}}, b::FermionBasis{<:Any,S}) where {M,S} = map(id->siteindex(id,b),ids)#::Int

function tensor(v::AbstractVector{T}, b::FermionBasis{M}) where {T,M}
    @assert length(v) == 2^M
    t = Array{T,M}(undef,ntuple(i->2,M))
    for I in CartesianIndices(t)
        fs = focknbr(Bool.(Tuple(I) .- 1))
        t[I] = v[focktoind(fs,b)]
    end
    return t
end

function partialtrace(v::AbstractVector{T}, labels::NTuple{N}, b::FermionBasis{M}) where {T,N,M}
    cinds = siteindices(labels, b) #::NTuple{N,Int} = map(label->findfirst(l->label==l, keys(b.dict)), labels)
    _partialtrace(tensor(v,b), cinds)
end
function _partialtrace(t::AbstractArray{<:Any,N}, cinds::NTuple{NC}) where {N,NC}
    ncinds::NTuple{N-NC,Int} = Tuple(setdiff(ntuple(identity,N),cinds))
    tp = permutedims(t,(ncinds...,cinds...))
    tpr = reshape(tp,prod(i->size(tp,i),ncinds, init=1),prod(i->size(tp,i),cinds,init=1))
    tpr*tpr'
end

function LinearAlgebra.svd(v::AbstractVector, leftlabels::NTuple{N}, b::FermionBasis{M}) where {N,M}
    linds = siteindices(leftlabels, b)
    rinds::NTuple{M-N,Int} = Tuple(setdiff(ntuple(identity,M),linds))
    t = tensor(v,b)
    tp = permutedims(t,(linds..., rinds...))
    tpr = reshape(tp,prod(i->size(tp,i),linds, init=1), prod(i->size(tp,i),rinds,init=1))
    svd(tpr)
end