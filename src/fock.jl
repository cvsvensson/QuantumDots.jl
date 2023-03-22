focknbr(bits::Union{BitVector,Vector{Bool},NTuple{<:Any,Bool}}) = mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits))
focknbr(site::Integer) = 2^(site-1)
focknbr(sites::Vector{<:Integer}) = mapreduce(focknbr,+, sites)
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
        t[I] = v[focktoind(fs,b)] #* parity(fs)
    end
    return t
end
##https://iopscience.iop.org/article/10.1088/1751-8121/ac0646/pdf (10c)
_bit(f,k) = Bool(sign(f & 2^(k-1)))
function phase_factor(focknbr1,focknbr2,subinds) 
    bitmask = focknbr(subinds)
    prod(i-> (jwstring(i, bitmask & focknbr1)*jwstring(i, bitmask & focknbr2))^_bit(focknbr2,i),subinds)
end
function partialtrace(t::AbstractArray{<:Any,N}, cinds::NTuple{NC}) where {N,NC}
    ncinds::NTuple{N-NC,Int} = Tuple(setdiff(ntuple(identity,N),cinds))
    Matrix(t,ncinds,cinds)
    mat*mat'
end
reduced_density_matrix(v::AbstractVector, labels, b::FermionBasis) = reduced_density_matrix(v*v',labels,b)
function reduced_density_matrix(m::AbstractMatrix{T}, labels::NTuple{N}, b::FermionBasis{M}) where {N,T,M}
    outinds::NTuple{N,Int} = siteindices(labels, b)
    @assert all(diff([outinds...]) .> 0) "Subsystems must be ordered in the same way as the full system"
    #ininds::NTuple{N,Int} = Tuple(setdiff(ntuple(identity,N),outinds))
    mout = zeros(T,2^(N),2^(N))
    bitmask = 2^M - 1 - focknbr(outinds)
    outbits(f) = map(i->_bit(f,i),outinds)
    for f1 in 0:2^M-1, f2 in 0:2^M-1
        if (f1 & bitmask) != (f2 & bitmask)
            continue
        end
        newfocknbr1 = focknbr(outbits(f1))
        newfocknbr2 = focknbr(outbits(f2))
        s1 = phase_factor(f1,f2,ntuple(identity,M))
        s2 = phase_factor(newfocknbr1,newfocknbr2, ntuple(identity,N))
        s = s2*s1
        mout[focktoind(newfocknbr1, NoSymmetry()), focktoind(newfocknbr2, NoSymmetry())] += s*m[focktoind(f1,b),focktoind(f2,b)]#*s2
    end
    return mout
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