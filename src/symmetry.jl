
blockinds(dims::Dims,sym::AbelianFockSymmetry) = blockinds(dims, values(sym.blocksizes))
qninds(qns::Dims,sym::AbelianFockSymmetry) = map(sizestoinds, sym.blocksizes[qns])'
blockinds(dims::Dims{N},sizes) where N = map(n->sizestoinds(sizes)[n],dims)
sizestoinds(sizes) = accumulate((a,b)->last(a) .+ (1:b), sizes,init=0:0)

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
