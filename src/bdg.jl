struct QuasiParticle{T,M,L}
    weights::Dictionary{Tuple{L,Symbol},T}
    basis::FermionBdGBasis{M,L}
end
function QuasiParticle(v::AbstractVector{T}, basis::FermionBdGBasis{M,L}) where {T,M,L}
    holelabels = map(k->(k,:h), keys(basis.position).values)
    particlelabels = map(k->(k,:p), keys(basis.position).values)
    weights = Dictionary(vcat(holelabels, particlelabels), collect(v))
    QuasiParticle{T,M,L}(weights, basis)
end
Base.getindex(qp::Union{QuasiParticle,MajoranaQuasiParticle}, i) = getindex(qp.weights,i)
Base.getindex(qp::Union{QuasiParticle,MajoranaQuasiParticle}, i...) = getindex(qp.weights,i)

struct MajoranaQuasiParticle{T,M,L}
    weights::Dictionary{Tuple{L,Symbol},T}
    basis::FermionBdGBasis{M,L}
end
function MajoranaQuasiParticle(v::AbstractVector{T}, basis::FermionBdGBasis{M,L}) where {T,M,L}
    xlabels = map(k->(k, :x), keys(basis.position).values)
    ylabels = map(k->(k, :y), keys(basis.position).values)
    weights = Dictionary(vcat(xlabels, ylabels), collect(v))
    MajoranaQuasiParticle{T,M,L}(weights, basis)
end

function majorana_transform(bdgham::AbstractMatrix)
    n = div(size(bdgham,1),2)
    i = Matrix(I/2,n,n)
    U = [i i; 1im*i -1im*i]
    U*bdgham*U'
end
function MajoranaQuasiParticle(quasiparticle::QuasiParticle)
    N = nbr_of_fermions(quasiparticle.basis)
    i = Matrix(I/sqrt(2),N,N)
    U = [i i; 1im*i -1im*i]
    MajoranaQuasiParticle(U*quasiparticle.weights.values, quasiparticle.basis)
end
# majorana_transform(quasiparticle::QuasiParticle) = MajoranaQuasiParticle(majorana_transform(quasiparticle.vector), quasiparticle.basis)

# function majorana_coefficients(a::QuasiParticle)
#     N = div(length(a),2)
#     return [(a[k] + a[k+N])/2 for k in 1:N], [(a[k] - a[k+N])/2 for k in 1:N] 
# end

labels(b::FermionBdGBasis) = keys(b.position).values
function _left_half_labels(basis::FermionBdGBasis)
    N = nbr_of_fermions(basis)
    labels(basis)[1:Int(ceil(N/2))]
end
function majorana_polarization(maj1::MajoranaQuasiParticle, maj2::MajoranaQuasiParticle, labels = _left_half_labels(maj1.basis))
    md1 = majorana_density(maj1, labels)^2
    md2 = majorana_density(maj2, labels)^2
    (md1-md2)/(md1+md2)
end

function majorana_density(maj::MajoranaQuasiParticle, labels = _left_half_labels(maj.basis))
    xs = map(l->maj[l,:x], labels)
    ys = map(l->maj[l,:y], labels)
    sum(abs2, Iterators.flatten((xs, ys)))
end