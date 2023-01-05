blockinds(i::Integer,sym::AbelianFockSymmetry) = blockinds(i, values(sym.qntoblocksizes))
blockinds(inds::Dims,sym::AbelianFockSymmetry) = map(i->blockinds(i, sym),inds)
blockinds(sym::AbelianFockSymmetry) = sizestoinds(values(sym.qntoblocksizes))

qninds(qns::Tuple,sym::AbelianFockSymmetry) = map(qn->qninds(qn,sym), qns)
qninds(qn,sym::AbelianFockSymmetry) = sym.qntoinds[qn]
blockinds(inds::Dims,sizes) = map(n->blockinds(n,sizes),inds)
blockinds(i::Integer,sizes) = sizestoinds(sizes)[i]

sizestoinds(sizes) = accumulate((a,b)->last(a) .+ (1:b), sizes,init=0:0)::Vector{UnitRange{Int}}

function symmetry(fermionids::NTuple{M}, qn) where M
    qntooldinds = group(ind->qn(ind-1), 1:2^M)
    sortkeys!(qntooldinds)
    oldindfromnew = vcat(qntooldinds...)
    blocksizes = map(length,qntooldinds)
    newindfromold = map(first,sort(collect(enumerate(oldindfromnew)),by=last))
    indtofocklist = oldindfromnew .- 1
    indtofock(ind) = indtofocklist[ind]
    focktoind(f) = newindfromold[f+1]
    qntoinds = map(oldinds->map(oldind->newindfromold[oldind],oldinds), qntooldinds)
    qntofockstates = map(oldinds-> oldinds .-1 , qntooldinds)
    AbelianFockSymmetry(indtofock,focktoind,blocksizes,qntofockstates,qntoinds,qn)
end

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