hopping(f1, f2) = f1'f2 + f2'f1
pairing(f1, f2) = f1'f2' + f2 * f1
numberop(f) = f'f
coulomb(f1, f2) = numberop(f1) * numberop(f2)

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



function _BD1_2site((c1up,c1dn),(c2up,c2dn); t,tϕ, Δϕ, Δasym, V, ϕ=0)
    t*(hopping(c1up,c2up) + hopping(c1dn,c2dn)) +
    tϕ*(exp(1im*ϕ)hopping(c1dn,c2up) - exp(-1im*ϕ)hopping(c1up,c2dn)) +
    V* (numberop(c1up)+numberop(c1dn))*(numberop(c2up)+numberop(c2dn)) +
    Δasym*(pairing(c1up,c2dn) - pairing(c1dn,c2up)) +
    Δϕ*(pairing(c1up,c2up)exp(1im*ϕ) + exp(-1im*ϕ)pairing(c1dn,c2dn))
end
function _BD1_1site((cup,cdn); μ,h,Δ,U)
    (-μ - h)*numberop(cup) + (-μ + h)*numberop(cdn) +
    Δ*pairing(cup,cdn) + U*numberop(cup)*numberop(cdn)
end

function BD1_hamiltonian(c::FermionBasis{M}; μ, h, Δ1, t, Δ, U, V, θ=0.0, ρ = π/2, ξ = π/2, bias=0.0) where M
    @assert length(cell(1,c)) == 2 "Each unit cell should have two fermions for this hamiltonian"
    N = div(M,2)
    dbias =  bias * collect(range(-0.5, 0.5, length=N))

    Δasym = Δ*cos(θ/2) # updn - dnup
    Δϕ = Δ*sin(θ/2) # upup*exp(iϕ) + exp(-iϕ)*dndn
    
    tsym = t*cos(θ/2) # dndn + upup
    tϕ = t*sin(θ/2) #dnup*exp(iϕ) - exp(-iϕ)*updn

    h1s = (_BD1_1site(cell(j,c); μ = μ + dbias[j],h,Δ,U) for j in 1:N)
    h2s = (_BD1_2site(cell(j,c), cell(j+1,c); t,α,Δ1,Δk,V) for j in 1:N-1)
    sum(Iterators.flatten((h1s,h2s)))
end
function BD1_hamiltonian_disorder(c::FermionBasis{M}; μs, h, Δ1, t, α, Δ, U, V, θ=0.0, bias=0.0) where M
    @assert length(cell(1,c)) == 2 "Each unit cell should have two fermions for this hamiltonian"
    N = div(M,2)
    dbias =  bias* collect(range(-0.5, 0.5, length=N))
    αnew = cos(θ/2)*α + sin(θ/2)*t
    tnew = cos(θ/2)*t - sin(θ/2)*α
    Δk = Δ1*sin(θ/2)
    Δ1 = Δ1*cos(θ/2)
    t = tnew
    α = αnew
    h1s = (_BD1_1site(cell(j,c); μ = μs[j]+dbias[j],h,Δ,U) for j in 1:N)
    h2s = (_BD1_2site(cell(j,c), cell(j+1,c); t,α,Δ1,Δk,V) for j in 1:N-1)
    sum(Iterators.flatten((h1s,h2s)))
end