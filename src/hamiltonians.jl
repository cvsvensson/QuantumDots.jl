struct HC end
Base.:+(m::AbstractArray, ::HC) = m+m'
const hc = HC()

hopping(t, f1, f2) = t*f1'f2 + hc
pairing(Δ, f1, f2) = Δ*f2 * f1 + hc
numberop(f) = f'f
hopping!(H, t, f1, f2) = (mul!(H,f1',f2,t,1); mul!(H,f2',f1,t',1))
pairing!(H,Δ, f1, f2) = (mul!(H,f2,f1,Δ,1); mul!(H,f1',f2',Δ',1))#@. H += Δ*f2 * f1 + Δ'*f1' * f2'
numberop!(H,μ,f) = mul!(H,f',f,μ,1) #(H .+= μ.*f'.*f; return H)
coulomb(f1, f2) = numberop(f1) * numberop(f2)
coulomb!(H,V,f1, f2) = mul!(H,f1'f1, f2'*f2,V,1) #TODO: Avoid allocations
function coulomb(f1::BdGFermion, f2::BdGFermion) 
    @warn "Returning zero as Coulomb term for BdGFermions. This message is not displayed again." maxlog=1
    0*numberop(f1)
end
coulomb!(H,V,f1::BdGFermion, f2::BdGFermion) = @warn "No Coulomb term added for BdGFermions. This message is not displayed again." maxlog=1
function su2_rotation((θ,ϕ))
    pf = mod(ϕ,π) == 0 ? real(exp(1im*ϕ)) : exp(1im*ϕ) #Gives runtime dispatch, but necessary for symbolic numbers to play nice
    # pf = exp(1im*ϕ)
    _su2_rotation(θ,pf)
end
function su2_rotation_no_symbolics((θ,ϕ))
    _su2_rotation(θ,exp(1im*ϕ))
end
_su2_rotation(θ,pf) = @SMatrix [cos(θ/2) -sin(θ/2)pf'; sin(θ/2)pf cos(θ/2)]

function hopping_rotated(t,(c1up,c1dn),(c2up,c2dn), angles1, angles2)
    Ω = su2_rotation(angles1)'*su2_rotation(angles2)
    t*(Ω[1,1]*c1up'*c2up + Ω[2,1]*c1dn'*c2up + Ω[1,2]*c1up'*c2dn + Ω[2,2]*c1dn'*c2dn) + hc
end
function hopping_rotated!(H,t,(c1up,c1dn),(c2up,c2dn), angles1, angles2)
    Ω = su2_rotation_no_symbolics(angles1)'*su2_rotation_no_symbolics(angles2)
    temp = zero(H)
    mul!(temp,c1up',c2up,t*Ω[1,1],1)
    mul!(temp,c1dn',c2up,t*Ω[2,1],1)
    mul!(temp,c1up',c2dn,t*Ω[1,2],1)
    mul!(temp,c1dn',c2dn,t*Ω[2,2],1)
    H .+= temp + hc
    # mul!(temp,c2up',c1up,t'*Ω[1,1]',1)
    # mul!(temp,c2up',c1dn,t'*Ω[2,1]',1)
    # mul!(temp,c2dn',c1up,t'*Ω[1,2]',1)
    # mul!(temp,c2dn',c1dn,t'*Ω[2,2]',1)
    # H .+= t*(Ω[1,1]*c1up'*c2up + Ω[2,1]*c1dn'*c2up + Ω[1,2]*c1up'*c2dn + Ω[2,2]*c1dn'*c2dn) + 
    # t'*(Ω[1,1]'*c2up'*c1up + Ω[2,1]'*c2up'*c1dn + Ω[1,2]'*c2dn'*c1up + Ω[2,2]'*c2dn'*c1dn)
    # return t*(Ω[1,1]*c1up'*c2up + Ω[2,1]*c1dn'*c2up + Ω[1,2]*c1up'*c2dn + Ω[2,2]*c1dn'*c2dn) + 
    # t'*(Ω[1,1]'*c2up'*c1up + Ω[2,1]'*c2up'*c1dn + Ω[1,2]'*c2dn'*c1up + Ω[2,2]'*c2dn'*c1dn)
    return H
end

function pairing_rotated(Δ,(c1up,c1dn),(c2up,c2dn),  angles1, angles2)
    Ω = transpose(su2_rotation(angles1))*[0 -1; 1 0]*su2_rotation(angles2)
    Δ*(Ω[1,1]*c1up*c2up + Ω[2,1]*c1dn*c2up + Ω[1,2]*c1up*c2dn + Ω[2,2]*c1dn*c2dn) + hc
end
function pairing_rotated!(H,Δ,(c1up,c1dn),(c2up,c2dn),  angles1, angles2)
    Ω = transpose(su2_rotation_no_symbolics(angles1))*[0 -1; 1 0]*su2_rotation_no_symbolics(angles2)
    temp = zero(H)
    mul!(temp,c1up,c2up,Δ*Ω[1,1],1)
    mul!(temp,c1dn,c2up,Δ*Ω[2,1],1)
    mul!(temp,c1up,c2dn,Δ*Ω[1,2],1)
    mul!(temp,c1dn,c2dn,Δ*Ω[2,2],1)
    H .+= temp + hc
    # mul!(H,c2up',c1up',Δ'*Ω[1,1]',1)
    # mul!(H,c2up',c1dn',Δ'*Ω[2,1]',1)
    # mul!(H,c2dn',c1up',Δ'*Ω[1,2]',1)
    # mul!(H,c2dn',c1dn',Δ'*Ω[2,2]',1)
    return H
    # @. H += Δ*(Ω[1,1]*c1up*c2up + Ω[2,1]*c1dn*c2up + Ω[1,2]*c1up*c2dn + Ω[2,2]*c1dn*c2dn) + 
    # Δ'*(Ω[1,1]'*c2up'*c1up' + Ω[2,1]'*c2up'*c1dn' + Ω[1,2]'*c2dn'*c1up' + Ω[2,2]'*c2dn'*c1dn')
end

_kitaev_1site(f; μ) = μ*numberop(f)
_kitaev_2site(f1, f2; t, Δ, V) = hopping(-t,f1, f2) + 4V * coulomb(f1, f2) + pairing(Δ,f1, f2)
_kitaev_1site!(H,f; μ) = numberop!(H,μ,f)
function _kitaev_2site!(H,f1, f2; t, Δ, V)
    hopping!(H, -t,f1, f2)
    coulomb!(H,4V,f1, f2)
    pairing!(H,Δ,f1, f2)
end
function kitaev_hamiltonian(basis::AbstractBasis; μ::Number, t::Number, Δ::Number, V::Number=0.0, bias::Number=0.0)
    N = nbr_of_fermions(basis)
    dbias = bias * collect(range(-0.5, 0.5, length=N))
    _kitaev_hamiltonian(basis; μ=fill(μ, N), t=fill(t, N-1), Δ=fill(Δ, N-1), V=fill(V, N-1), bias=dbias)
end

function _kitaev_hamiltonian(c::AbstractBasis; μ, t, Δ, V, bias)
    N = nbr_of_fermions(c)
    h1s = (_kitaev_1site(c[j]; μ=μ[j] + bias[j]) for j in 1:N)
    h2s = (_kitaev_2site(c[j], c[j+1]; t = t[j], Δ = Δ[j], V = V[j]) for j in 1:N-1)
    hs = Iterators.flatten((h1s, h2s))
    sum(hs)
end



function _BD1_2site((c1up,c1dn),(c2up,c2dn); t, Δ1, V, θϕ1,θϕ2)
    ms = @SVector [hopping_rotated(t,(c1up,c1dn),(c2up,c2dn),θϕ1,θϕ2),
    pairing_rotated(Δ1,(c1up,c1dn),(c2up,c2dn),θϕ1,θϕ2),
    iszero(V) ? missing : V*((numberop(c1up)+numberop(c1dn))*(numberop(c2up)+numberop(c2dn)))]
    sum(skipmissing(ms))
end
function _BD1_1site((cup,cdn); μ,h,Δ,U)
    (-μ - h)*numberop(cup) + (-μ + h)*numberop(cdn) +
    pairing(Δ, cup,cdn) + U*(numberop(cup)*numberop(cdn))
end

function _BD1_2site!(H,(c1up,c1dn),(c2up,c2dn); t, Δ1, V, θϕ1,θϕ2)
    hopping_rotated!(H,t,(c1up,c1dn),(c2up,c2dn),θϕ1,θϕ2)
    pairing_rotated!(H,Δ1,(c1up,c1dn),(c2up,c2dn),θϕ1,θϕ2)
    coulomb!(H,V,c1up,c2up)
    coulomb!(H,V,c1up,c2dn)
    coulomb!(H,V,c1dn,c2up)
    coulomb!(H,V,c1dn,c2dn)
    # H .+= V*((numberop(c1up)+numberop(c1dn))*(numberop(c2up)+numberop(c2dn)))
    return H
end
function _BD1_1site!(H,(cup,cdn); μ,h,Δ,U)
    numberop!(H,(-μ - h),cup)
    numberop!(H,(-μ + h),cdn)
    pairing!(H,Δ, cup,cdn)
    coulomb!(H,U,cup,cdn)
    return H
end

_tovec(μ::Number,N) = fill(μ,N)
_tovec(μ::Vector,N) = (@assert length(μ)==N; μ)
function _tovec((x,symb),N) 
    if symb == :diff
        return _tovec(x,N) .* (0:N-1) 
    elseif symb == :reflect
        @assert length(x) == Int(ceil(N/2)) "$x does not match half the sites of $N"
        if iseven(N)
            return (x...,reverse(x)...)
        else
            return (x...,reverse(x)[2:end]...)
        end
    end
    return _tovec(x,N)
end
function BD1_hamiltonian(c::AbstractBasis; μ, h, t, Δ, Δ1, U, V, θ, ϕ)
    M = nbr_of_fermions(c)
    @assert length(cell(1,c)) == 2 "Each unit cell should have two fermions for this hamiltonian"
    N = div(M,2)
    θϕ = collect(zip(_tovec(θ,N),_tovec(ϕ,N)))
    _BD1_hamiltonian(c; μ = _tovec(μ,N), h = _tovec(h,N), t = _tovec(t,N), Δ = _tovec(Δ,N),Δ1 = _tovec(Δ1,N), U = _tovec(U,N), V = _tovec(V,N), θϕ)
end
function BD1_hamiltonian!(H,c::AbstractBasis; μ, h, t, Δ, Δ1, U, V, θ, ϕ)
    fill!(H, zero(eltype(H)))
    M = nbr_of_fermions(c)
    @assert length(cell(1,c)) == 2 "Each unit cell should have two fermions for this hamiltonian"
    N = div(M,2)
    θϕ = collect(zip(_tovec(θ,N),_tovec(ϕ,N)))
    # H = zeros(promote_type(eltype((μ, h, t, Δ, Δ1, U, V, θ, ϕ))...),2^M,2^M)
    _BD1_hamiltonian!(H,c; μ = _tovec(μ,N), h = _tovec(h,N), t = _tovec(t,N), Δ = _tovec(Δ,N),Δ1 = _tovec(Δ1,N), U = _tovec(U,N), V = _tovec(V,N), θϕ)
    return H
end

function _BD1_hamiltonian(c::AbstractBasis; μ::Vector, h::Vector, t::Vector, Δ::Vector,Δ1::Vector, U::Vector, V::Vector, θϕ::Vector)
    M = nbr_of_fermions(c)
    @assert length(cell(1,c)) == 2 "Each unit cell should have two fermions for this hamiltonian"
    @assert length(μ) == div(M,2)
    N = div(M,2)
    h1s = (_BD1_1site(cell(j,c); μ = μ[j], h = h[j], Δ = Δ[j], U = U[j]) for j in 1:N)
    h2s = (_BD1_2site(cell(j,c), cell(j+1,c); t = t[j] ,Δ1 = Δ1[j],V = V[j],θϕ1=θϕ[j], θϕ2=θϕ[j+1]) for j in 1:N-1)
    sum(Iterators.flatten((h1s,h2s)))
end

function _BD1_hamiltonian!(H,c::AbstractBasis; μ::Vector, h::Vector, t::Vector, Δ::Vector,Δ1::Vector, U::Vector, V::Vector, θϕ::Vector)
    M = nbr_of_fermions(c)
    @assert length(cell(1,c)) == 2 "Each unit cell should have two fermions for this hamiltonian"
    @assert length(μ) == div(M,2)
    N = div(M,2)
    foreach(j->_BD1_1site!(H,cell(j,c); μ = μ[j], h = h[j], Δ = Δ[j], U = U[j]), 1:N)
    foreach(j->_BD1_2site!(H,cell(j,c), cell(j+1,c); t = t[j] ,Δ1 = Δ1[j],V = V[j],θϕ1=θϕ[j], θϕ2=θϕ[j+1]),1:N-1)
    return H
end


function TSL_hamiltonian(c::AbstractBasis; μL,μC,μR, h, t, Δ, U,tsoc)
    @assert nbr_of_fermions(c) == 6 "This basis has the wrong number of fermions for this hamiltonian $(nbr_of_fermions(c)) != 6"
    fermions = [(c[j,:↑],c[j,:↓]) for j in (:L,:C,:R)]
    TSL_hamiltonian(fermions...; μL,μC,μR, h, t, Δ, U,tsoc)
end
function TSL_hamiltonian((dLup,dLdn),(dCup,dCdn),(dRup,dRdn); μL,μC,μR, h, t, Δ, U,tsoc) 
    hopping(t,dLup,dCup) + hopping(t,dRup,dCup) +
    hopping(t,dLdn,dCdn) + hopping(t,dRdn,dCdn) +
    hopping(tsoc,dLup,dCdn) - hopping(tsoc,dLdn,dCup) -
    (hopping(tsoc,dRup,dCdn) - hopping(tsoc,dRdn,dCup)) +
    -pairing(Δ,dCup,dCdn) + 
    U*(numberop(dLup)numberop(dLdn) + numberop(dRup)*numberop(dRdn)) + 
    μL*(numberop(dLup) + numberop(dLdn)) +
    μC*(numberop(dCup) + numberop(dCdn)) +
    μR*(numberop(dRup) + numberop(dRdn)) +
    + h*(numberop(dLdn) + numberop(dRdn))
end


function TSL_generator(qn=NoSymmetry(); blocks = qn !== NoSymmetry(), dense = false, bdg=false)
    @variables μL, μC, μR, h, t, Δ, tsoc, U
    c = if !bdg
        FermionBasis((:L,:C,:R),(:↑,:↓); qn)
    elseif bdg && qn==NoSymmetry()
        FermionBdGBasis(Tuple(collect(Base.product((:L,:C,:R),(:↑,:↓)))) )
    end
    fdense = dense ? Matrix : identity
    fblock = blocks ? m->blockdiagonal(m,c) : identity
    f = fblock ∘ fdense
    H = TSL_hamiltonian(c;μL, μC, μR,h,t,Δ,tsoc,U) |> f
    _tsl,_tsl! = build_function(H,μL, μC, μR,h,t,Δ,tsoc,U,expression=Val{false})
    tsl(;μL, μC, μR,h,t,Δ,tsoc,U) = _tsl(μL, μC, μR,h,t,Δ,tsoc,U)
    tsl!(m;μL, μC, μR,h,t,Δ,tsoc,U) = (_tsl!(m,μL, μC, μR,h,t,Δ,tsoc,U); m)
    randparams = (;zip((:μL, :μC, :μR,:h,:t,:Δ,:tsoc,:U), rand(8))...)
    m = TSL_hamiltonian(c;randparams...) |> f
    return tsl, tsl!, m, c
end