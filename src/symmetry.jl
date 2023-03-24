blockinds(i::Integer,sym::AbelianFockSymmetry) = blockinds(i, values(sym.qntoblocksizes))
blockinds(inds::Dims,sym::AbelianFockSymmetry) = map(i->blockinds(i, sym),inds)
blockinds(sym::AbelianFockSymmetry) = sizestoinds(values(sym.qntoblocksizes))

qninds(qns::Tuple,sym::AbelianFockSymmetry) = map(qn->qninds(qn,sym), qns)
qninds(qn,sym::AbelianFockSymmetry) = sym.qntoinds[qn]
blockinds(inds::Dims,sizes) = map(n->blockinds(n,sizes),inds)
blockinds(i::Integer,sizes) = sizestoinds(sizes)[i]

sizestoinds(sizes) = accumulate((a,b)->last(a) .+ (1:b), sizes,init=0:0)::Vector{UnitRange{Int}}
abstract type AbstractQuantumNumber end

struct QNIndex{T}
    qn::T
    ind::Int
end
qn(qnind::QNIndex) = qnind.qn
ind(qnind::QNIndex) = qnind.ind

struct Z2Symmetry{N} end
qns(::Z2Symmetry) = Z2.([false, true])
struct Z2 <: AbstractQuantumNumber
    val::Bool
end

struct U1Symmetry{N} end
qns(::U1Symmetry{N}) where N = U1.(0:N)
struct U1 <: AbstractQuantumNumber
    val::Integer
end


focktoqnind(fs, sym::Z2Symmetry) = indtoqnind(fs+1, sym)
qnindtoind(qnind::QNIndex{Z2}, ::Z2Symmetry{N}) where N = qnind.ind + (qnind.qn.val ? 2^(N-1) : 0)
indtoqnind(ind,::Z2Symmetry{N}) where N = ind < 2^(N-1) ? QNIndex(Z2(false),ind) : QNIndex(Z2(true), ind - 2^(N-1))
blocksize(::Z2,::Z2Symmetry{N}) where N = 2^(N-1)
Base.:(==)(a::Z2,b::Z2) = a.val == b.val
Base.size(::Z2Symmetry{N}) where N = 2^N


focktoqnind(fs, sym::U1Symmetry) = indtoqnind(fs+1, sym)
qnindtoind(qnind::QNIndex{U1}, ::U1Symmetry{N}) where N = sum(binomial(N,m) for m in 0:qn(qnind).val-1; init=0) + ind(qnind)
function indtoqnind(ind,::U1Symmetry{N}) where N
    qn = count_ones(ind-1)
    return QNIndex(U1(qn), ind - sum(binomial(N,m) for m in 0:qn-1; init=0))
end
blocksize(n::U1,::U1Symmetry{N}) where N = binomial(N,n.val)
Base.:(==)(a::U1,b::U1) = a.val == b.val
Base.size(::U1Symmetry{N}) where N = 2^N


function symmetry(fermionids::NTuple{M}, qn) where M
    qntooldinds = group(ind->qn(ind-1), 1:2^M)
    sortkeys!(qntooldinds)
    oldindfromnew = vcat(qntooldinds...)
    blocksizes = map(length,qntooldinds)
    newindfromold = map(first,sort(collect(enumerate(oldindfromnew)),by=last))
    indtofockdict = oldindfromnew .- 1
    indtofock(ind) = indtofockdict[ind]
    focktoinddict = Dictionary(0:(2^M - 1), newindfromold)
    qntoinds = map(oldinds->map(oldind->newindfromold[oldind],oldinds), qntooldinds)
    qntofockstates = map(oldinds-> oldinds .-1 , qntooldinds)
    AbelianFockSymmetry(indtofockdict,focktoinddict,blocksizes,qntofockstates,qntoinds,qn)
end

indtofock(ind, sym::AbelianFockSymmetry) = sym.indtofockdict[ind]
focktoind(f, sym::AbelianFockSymmetry) = sym.focktoinddict[f]

function fermion_sparse_matrix(fermion_number,totalsize,sym::AbelianFockSymmetry)
    mat = spzeros(Int,totalsize,totalsize)
    _fill!(mat, fs -> removefermion(fermion_number,fs), sym)
    mat
end

blockdiagonal(m::AbstractMatrix,basis::FermionBasis) = blockdiagonal(m,basis.symmetry)
blockdiagonal(::Type{T},m::AbstractMatrix,basis::FermionBasis) where T = blockdiagonal(T, m,basis.symmetry)

function blockdiagonal(m::AbstractMatrix,sym::AbelianFockSymmetry)
    blockinds = values(sym.qntoinds)
    BlockDiagonal([m[block,block] for block in blockinds])
end
function blockdiagonal(::Type{T}, m::AbstractMatrix,sym::AbelianFockSymmetry) where T
    blockinds = values(sym.qntoinds)
    BlockDiagonal([T(m[block,block]) for block in blockinds])
end

focktoind(fs,b::FermionBasis) = focktoind(fs,b.symmetry)
indtofock(ind,b::FermionBasis) = indtofock(ind,b.symmetry)

focktoind(fs,::NoSymmetry) = fs + 1
indtofock(ind,::NoSymmetry) = ind -1

function nextfockstate_with_same_number(v)
    #http://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    t = (v | (v - 1)) + 1;  
    t | (((div((t & -t), (v & -v))) >> 1) - 1)
end
function fockstates(M,n)
    v::Int = focknbr(ntuple(i->true,n))
    maxv = v*2^(M-n)
    states = Vector{Int}(undef,binomial(M,n))
    count = 1
    while v <= maxv
        states[count] = v
        v = nextfockstate_with_same_number(v)
        count+=1
    end
    states
end