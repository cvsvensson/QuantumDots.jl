
struct QuasiParticle{T,M,L}
    vector::Vector{T}
    basis::FermionBdGBasis{M,L}
    function QuasiParticle(v::Vector{T}, basis::FermionBdGBasis{M,L}) where {T,M,L}
        new{T,M,L}(v, basis)
    end
end
struct MajoranaQuasiParticle{T,M,L}
    vector::Vector{T}
    basis::FermionBdGBasis{M,L}
    function MajoranaQuasiParticle(v::Vector{T}, basis::FermionBdGBasis{M,L}) where {T,M,L}
        new{T,M,L}(v, basis)
    end
end
function majorana_transform(bdgham::AbstractMatrix)
    n = div(size(bdgham,1),2)
    i = Matrix(I/2,n,n)
    U = [i i; 1im*i -1im*i]
    U*bdgham*U'
end
function majorana_transform(quasiparticle::AbstractVector)
    n = div(length(quasiparticle),2)
    i = Matrix(I/2,n,n)
    U = [i i; 1im*i -1im*i]
    U*quasiparticle
end
majorana_transform(quasiparticle::QuasiParticle) = MajoranaQuasiParticle(majorana_transform(quasiparticle.vector), quasiparticle.basis)

function majorana_coefficients(a::QuasiParticle)
    N = div(length(a),2)
    return [(a[k] + a[k+N])/2 for k in 1:N], [(a[k] - a[k+N])/2 for k in 1:N] 
end

function majorana_polarization(maj::MajoranaQuasiParticle)
    
end