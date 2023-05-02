blockinds(i::Integer,sym::AbelianFockSymmetry) = blockinds(i, values(sym.qntoblocksizes))
blockinds(inds::Dims,sym::AbelianFockSymmetry) = map(i->blockinds(i, sym),inds)
blockinds(sym::AbelianFockSymmetry) = sizestoinds(values(sym.qntoblocksizes))

qninds(qns::Tuple,sym::AbelianFockSymmetry) = map(qn->qninds(qn,sym), qns)
qninds(qn,sym::AbelianFockSymmetry) = sym.qntoinds[qn]
blockinds(inds::Dims,sizes) = map(n->blockinds(n,sizes),inds)
blockinds(i::Integer,sizes) = sizestoinds(sizes)[i]

sizestoinds(sizes) = accumulate((a,b)->last(a) .+ (1:b), sizes,init=0:0)::Vector{UnitRange{Int}}

struct QNIndex{T}
    qn::T
    ind::Int
end

abstract type AbstractBlockStructure end
abstract type AbstractFockBlockStructure end

struct Blocks{L} <: AbstractBlockStructure
    blocks::Dictionary{L,Int}
end
struct GenericFockBlocks{L,F} <: AbstractFockBlockStructure
    b::Blocks{L}
    qnfunc::F
    blockindtofock::Dictionary{QNIndex{L},Int}
    focktoblockind::Dictionary{Int,QNIndex{L}}
end
struct TrueFilter end
Iterators.Filter(::TrueFilter, itr) = itr 
GenericFockBlocks(func, M::Int; fockfilter = TrueFilter()) = GenericFockBlocks(func, Iterators.filter(fockfilter,0:2^M-1))

function GenericFockBlocks(func, fockstates)
    _blocktofocks = group(func, fockstates)
    blocktofocks = Dictionary([keys(_blocktofocks)...],_blocktofocks.values)
    focktoblockind = dictionary(vcat([map((i,fs) -> fs=>QNIndex(label,i), eachindex(focks),focks) for (label,focks) in pairs(blocktofocks)]...))
    blockindtofock = Dictionary(focktoblockind.values, keys(focktoblockind).values)
    blocks = Blocks(map(length,blocktofocks))
    GenericFockBlocks(blocks, func, blockindtofock, focktoblockind)
end

abstract type AbstractQuantumNumber end

qn(qnind::QNIndex) = qnind.qn
ind(qnind::QNIndex) = qnind.ind

struct Parity <:AbstractSymmetry end

struct Z2 <: AbstractQuantumNumber
    val::Bool
end
struct Z2Symmetry <: AbstractSymmetry
    focktoqnind::Dictionary{Int,QNIndex{Z2}}
    qnindtofock::Dictionary{QNIndex{Z2},Int}
end


struct ParityFockBlocks <: AbstractFockBlockStructure
    b::Blocks{Symbol}
    evenfocktoind::Dictionary{Int,Int}
    oddfocktoind::Dictionary{Int,Int}
end
ParityFockBlocks(M::Int; fockfilter = TrueFilter()) = ParityFockBlocks(Iterators.filter(fockfilter,0:2^M-1))
function ParityFockBlocks(fockstates)
    evencount = 0
    oddcount = 0
    evenfocktoind = Pair{Int,Int}[]
    oddfocktoind = Pair{Int,Int}[]
    for fs in fockstates
        parity = iseven(fermionnumber(fs))
        if parity
            evencount += 1
            push!(evenfocktoind, fs => evencount)
        else
            oddcount += 1
            push!(oddfocktoind, fs => oddcount)
        end
    end
    blocks = Blocks(Dictionary([:even,:odd], [length(evenfocktoind),length(oddfocktoind)]))
    return ParityFockBlocks(blocks, dictionary(evenfocktoind), dictionary(oddfocktoind))
end

function focktoqnind(fs, blockstructure::ParityFockBlocks)
    if iseven(fermionnumber(fs))
        QNIndex(qn, blockstructure.evenfocktoind[fs])
    else
        QNIndex(qn, blockstructure.oddfocktoind[fs])
    end
end

blockstructure(::NTuple{M}, ::Type{Z2}) where M = Z2Symmetry()
Z2Symmetry() = Z2Symmetry(enumerate_Z2fockstates(N)...)
qns(::Z2Symmetry) = (Z2(false), Z2(true))
Z2parity(fs::Integer) = Z2(!iseven(fermionnumber(fs)))
function enumerate_Z2fockstates(M; filter = fs -> true)
    evencount = 0
    oddcount = 0
    qninds = QNIndex{Z2}[]
    fockstates = Int[]
    for fs in 0:2^M - 1
        parity = iseven(fermionnumber(fs))
        filter(fs) || continue
        push!(fockstates, fs)
        if parity
            evencount += 1
            push!(qninds,QNIndex(Z2(false),evencount))
        else
            oddcount += 1
            push!(qninds,QNIndex(Z2(true),oddcount))
        end
    end
    focktoqninds = Dictionary(fockstates, qninds)
    return focktoqninds, Dictionary(focktoqninds.values, keys(focktoqninds).values)
end

struct U1Symmetry{N} <: AbstractSymmetry end
qns(::U1Symmetry{N}) where N = ntuple(n->U1(n-1),N)
struct U1 <: AbstractQuantumNumber
    val::Integer
end


# qnindtofock(qnind, sym::Z2Symmetry) = sym.qnindtofock[qnind]
# focktoqnind(fs, sym::Z2Symmetry) = sym.focktoqnind[fs]
# qnindtoind(qnind::QNIndex{Z2}, ::Z2Symmetry) where N = qnind.ind + (qnind.qn.val ? 2^(N-1) : 0)
# indtoqnind(ind,::Z2Symmetry)  = ind < 2^(N-1) ? QNIndex(Z2(false),ind) : QNIndex(Z2(true), ind - 2^(N-1))
# blocksize(::Z2,::Z2Symmetry) = 2^(N-1)
# Base.:(==)(a::Z2,b::Z2) = a.val == b.val
# qnsize(::Z2Symmetry)  = 2^N


focktoqnind(fs, sym::U1Symmetry) = indtoqnind(fs+1, sym)
qnindtoind(qnind::QNIndex{U1}, ::U1Symmetry{N}) where N = sum(binomial(N,m) for m in 0:qn(qnind).val-1; init=0) + ind(qnind)
function indtoqnind(ind,::U1Symmetry{N}) where N
    qn = count_ones(ind-1)
    return QNIndex(U1(qn), ind - sum(binomial(N,m) for m in 0:qn-1; init=0))
end
blocksize(n::U1,::U1Symmetry{N}) where N = binomial(N,n.val)
Base.:(==)(a::U1,b::U1) = a.val == b.val
qnsize(::U1Symmetry{N}) where N = 2^N


function blockstructure(fermionids::NTuple{M}, qn) where M
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

function fermion_sparse_matrix(fermion_number,sym::AbelianFockSymmetry)
    mat = spzeros(Int, size(sym), size(sym))
    _fill!(mat, fs -> removefermion(fermion_number,fs), sym)
    mat
end

function fermion_sparse_matrix(fermion_number, sym::Z2Symmetry,M) 
    qns = [(Z2(false),Z2(true)), (Z2(true),Z2(false))]
    mats = [spzeros(Int,2^(M-1),2^(M-1)) for k in eachindex(qns)]
    A = QArray(qns,mats,(sym,sym))
    _fill!(A, fs -> removefermion(fermion_number,fs))
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