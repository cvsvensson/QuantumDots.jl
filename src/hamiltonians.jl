struct HC end
Base.:+(m,::HC) = m+m'
const hc = HC()

hopping(t, f1, f2) = t*f1'f2 + hc
pairing(Δ, f1, f2) = Δ*f2 * f1 + hc
numberop(f) = f'f
coulomb(f1, f2) = numberop(f1) * numberop(f2)

su2_rotation(θ::Number) = @SMatrix [cos(θ/2) -sin(θ/2); sin(θ/2) cos(θ/2)]
su2_rotation((θ,ϕ)) = @SMatrix [cos(θ/2) -sin(θ/2)exp(-1im*ϕ); sin(θ/2)exp(1im*ϕ) cos(θ/2)]

function hopping_rotated(t,(c1up,c1dn),(c2up,c2dn), angles1, angles2)
    Ω = su2_rotation(angles1)'*su2_rotation(angles2)
    t*(Ω[1,1]*c1up'*c2up + Ω[2,1]*c1dn'*c2up + Ω[1,2]*c1up'*c2dn + Ω[2,2]*c1dn'*c2dn) + hc
end

function pairing_rotated(Δ,(c1up,c1dn),(c2up,c2dn),  angles1, angles2)
    Ω = transpose(su2_rotation(angles1))*[0 -1; 1 0]*su2_rotation(angles2)
    Δ*(Ω[1,1]*c1up*c2up + Ω[2,1]*c1dn*c2up + Ω[1,2]*c1up*c2dn + Ω[2,2]*c1dn*c2dn) + hc
end

_kitaev_2site(f1, f2; t, Δ, V) = -t * hopping(f1, f2) + 4V * coulomb(f1, f2) + Δ * pairing(f1, f2)
_kitaev_1site(f; μ) = -μ * numberop(f)

function kitaev_hamiltonian(basis::FermionBasis{N}; μ::Number, t::Number, Δ::Number, V::Number=0.0, bias::Number=0.0) where {N}
    dbias = bias * collect(range(-0.5, 0.5, length=N))
    _kitaev_hamiltonian(basis; μ=fill(μ, N), t=fill(t, N), Δ=fill(Δ, N), V=fill(V, N), bias=dbias)
end

function _kitaev_hamiltonian(c::FermionBasis{N}; μ, t, Δ, V, bias) where {N}
    h1s = (_kitaev_1site(c[j]; μ=μ[j] + bias[j]) for j in 1:N)
    h2s = (_kitaev_2site(c[j], c[j+1]; t = t[j], Δ = Δ[j], V = V[j]) for j in 1:N-1)
    hs = Iterators.flatten((h1s, h2s))
    sum(hs)
end



function _BD1_2site((c1up,c1dn),(c2up,c2dn); t, Δ1, V, θϕ1,θϕ2)
    hopping_rotated(t,(c1up,c1dn),(c2up,c2dn),θϕ1,θϕ2) +
    pairing_rotated(Δ1,(c1up,c1dn),(c2up,c2dn),θϕ1,θϕ2) +
    V* ((numberop(c1up)+numberop(c1dn))*(numberop(c2up)+numberop(c2dn)))
end
function _BD1_1site((cup,cdn); μ,h,Δ,U)
    (-μ - h)*numberop(cup) + (-μ + h)*numberop(cdn) +
    pairing(Δ, cup,cdn) + U*numberop(cup)*numberop(cdn)
end

_tovec(μ::Number,N) = fill(μ,N)
_tovec(μ::Vector,N) = (@assert length(μ)==N; μ)
_tovec((x,diff),N) = _tovec(x,N) .* (diff==:diff ? (0:N-1) : 1)
function BD1_hamiltonian(c::FermionBasis{M}; μ, h, t, Δ, Δ1, U, V, θ, ϕ) where M
    @assert length(cell(1,c)) == 2 "Each unit cell should have two fermions for this hamiltonian"
    N = div(M,2)
    θϕ = collect(zip(_tovec(θ,N),_tovec(ϕ,N)))
    _BD1_hamiltonian(c::FermionBasis{M}; μ = _tovec(μ,N), h = _tovec(h,N), t = _tovec(t,N), Δ = _tovec(Δ,N),Δ1 = _tovec(Δ1,N), U = _tovec(U,N), V = _tovec(V,N), θϕ)
end
function real_BD1_hamiltonian(c::FermionBasis{M}; μ, h, t, Δ, Δ1, U, V, dθ) where M
    @assert length(cell(1,c)) == 2 "Each unit cell should have two fermions for this hamiltonian"
    N = div(M,2)
    _BD1_hamiltonian(c::FermionBasis{M}; μ = _tovec(μ,N), h = _tovec(h,N), t = _tovec(t,N), Δ = _tovec(Δ,N),Δ1 = _tovec(Δ1,N), U = _tovec(U,N), V = _tovec(V,N), θϕ=_tovec((dθ,:diff),N))
end

function _BD1_hamiltonian(c::FermionBasis{M}; μ::Vector, h::Vector, t::Vector, Δ::Vector,Δ1::Vector, U::Vector, V::Vector, θϕ::Vector) where {M}
    @assert length(cell(1,c)) == 2 "Each unit cell should have two fermions for this hamiltonian"
    @assert length(μ) == div(M,2)
    N = div(M,2)
    h1s = (_BD1_1site(cell(j,c); μ = μ[j], h = h[j], Δ = Δ[j], U = U[j]) for j in 1:N)
    h2s = (_BD1_2site(cell(j,c), cell(j+1,c); t = t[j] ,Δ1 = Δ1[j],V = V[j],θϕ1=θϕ[j], θϕ2=θϕ[j+1]) for j in 1:N-1)
    sum(Iterators.flatten((h1s,h2s)))
end
