focknbr(bits::Union{BitVector,Vector{Bool},NTuple{<:Any,Bool}}) = mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits))
focknbr(site::Integer) = 2^(site-1)
focknbr(sites::Vector{<:Integer}) = mapreduce(focknbr,+, sites; init = 0)
focknbr(sites::NTuple{N,<:Integer}) where N = mapreduce(site -> 2^(site-1),+, sites)
focknbr(sites::Vector{<:Integer},cell_length, species_index=1) = mapreduce(site->2^(digitposition(site,cell_length,species_index)-1),+, sites)
bits(s::Integer,N) = digits(Bool,s, base=2, pad=N)
parity(fs::Int) = (-1)^count_ones(fs)
fermionnumber(fs::Int) = count_ones(fs)

siteindex(id::S,b::FermionBasis{<:Any,S}) where S = findfirst(x->x==id,collect(keys(b.dict)))::Int
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

function reduced_density_matrix(v::AbstractVector{T}, labels::NTuple{N}, b::FermionBasis{M}) where {T,N,M}
    outinds = siteindices(labels, b) #::NTuple{N,Int} = map(label->findfirst(l->label==l, keys(b.dict)), labels)
    #_partialtrace(tensor(v,b), outinds)
    mat = Matrix(tensor(v,b), outinds)
    mat*mat'
end
function partialtrace(t::AbstractArray{<:Any,N}, cinds::NTuple{NC}) where {N,NC}
    ncinds::NTuple{N-NC,Int} = Tuple(setdiff(ntuple(identity,N),cinds))
    Matrix(t,ncinds,cinds)
    mat*mat'
end

function Base.Matrix(t::AbstractArray{<:Any,N}, leftindices::NTuple{NL,Int}) where {N,NL}
    rightindices::NTuple{N-NL,Int} = Tuple(setdiff(ntuple(identity,N), leftindices))
    Matrix(t,leftindices,rightindices)
end
function Base.Matrix(t::AbstractArray{<:Any,N}, leftindices::NTuple{NL,Int}, rightindices::NTuple{NR,Int}) where {N,NL,NR}
    @assert NL+NR == N
    tperm = permutedims(t,(leftindices...,rightindices...))
    lsize = prod(i->size(tperm,i), leftindices, init=1)
    rsize = prod(i->size(tperm,i), rightindices, init=1)
    reshape(tperm, lsize, rsize)
end

function LinearAlgebra.svd(v::AbstractVector, leftlabels::NTuple{N}, b::FermionBasis{M}) where {N,M}
    linds = siteindices(leftlabels, b)
    t = tensor(v,b)
    svd(Matrix(t,linds))
end