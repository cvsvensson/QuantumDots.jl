struct QuasiParticle{T,M,L} <: AbstractBdGFermion
    weights::Dictionary{Tuple{L,Symbol},T}
    basis::FermionBdGBasis{M,L}
end
function QuasiParticle(v::AbstractVector{T}, basis::FermionBdGBasis{M,L}) where {T,M,L}
    holelabels = map(k -> (k, :h), keys(basis.position).values)
    particlelabels = map(k -> (k, :p), keys(basis.position).values)
    weights = Dictionary(vcat(holelabels, particlelabels), collect(v))
    QuasiParticle{T,M,L}(weights, basis)
end
Base.getindex(qp::QuasiParticle, i) = getindex(qp.weights, i)
Base.getindex(qp::QuasiParticle, i...) = getindex(qp.weights, i)

function majoranas(qp::QuasiParticle)
    return qp + qp', qp - qp'
end

Base.keys(b::FermionBdGBasis) = keys(b.position)
labels(b::FermionBdGBasis) = keys(b).values
Base.keys(qp::QuasiParticle) = keys(qp.weights)
labels(qp::QuasiParticle) = keys(qp).values
basis(qp::QuasiParticle) = qp.basis
function _left_half_labels(basis::FermionBdGBasis)
    N = nbr_of_fermions(basis)
    labels(basis)[1:Int(ceil(N / 2))]
end
function majorana_polarization(f::QuasiParticle, labels=_left_half_labels(basis(f)))
    md1, md2 = majorana_densities(f, labels)
    (md1 - md2) / (md1 + md2)
end

function majorana_wavefunctions(f::QuasiParticle, labels = labels(basis(f)))
    xylabels = [map(l -> (l, :x), labels); map(l -> (l, :y), labels)]
    xplus = map(l -> real(f[l,:h] + f[l, :p]), labels)
    xminus = map(l -> imag(f[l, :h] + f[l, :p]), labels)
    yplus = map(l -> imag(f[l, :h] - f[l, :p]), labels)
    yminus = map(l -> real(f[l, :h] - f[l, :p]), labels)
    return Dictionary(xylabels, [xplus; yplus]),
    Dictionary(xylabels, [xminus; yminus])
end
function majorana_densities(f::QuasiParticle, labels=_left_half_labels(basis(f)))
    γplus,γminus = majorana_wavefunctions(f,labels)
    sum(abs2, γplus), sum(abs2, γminus)
end

function majvisualize(qp::QuasiParticle)
    ls = labels(basis(qp))
    for (γ,title) in (((qp+qp')/2,"γ₊ = (χ + χ')/2"), ((qp-qp')/2,"γ₋ = (χ - χ')/2"))
        xlabels = map(l -> (l, :x), ls)
        ylabels = map(l -> (l, :y), ls)
        xweights = map(l -> γ[l, :h] + γ[l, :p], ls)
        yweights = map(l -> γ[l, :h] - γ[l, :p], ls)
        display(barplot(xlabels, abs2.(xweights); title, maximum=1, border=:ascii))
        display(barplot(ylabels, abs2.(yweights), maximum=1, border=:dashed))
    end
end
function visualize(qp::QuasiParticle)
    hlabels = map(l -> (l, :h), labels(qp.basis))
    plabels = map(l -> (l, :p), labels(qp.basis))
    hweights = map(l -> qp[l], hlabels)
    pweights = map(l -> qp[l], plabels)
    display(barplot(hlabels, abs2.(hweights), title="Quasiparticle", maximum=1, border=:ascii))
    display(barplot(plabels, abs2.(pweights), maximum=1, border=:dashed))
end

"""
    one_particle_density_matrix(χ::QuasiParticle{T})

    Gives the one_particle_density_matrix for the state with χ as it's ground state
"""
function one_particle_density_matrix(χ::QuasiParticle{T}) where T
    N = nbr_of_fermions(basis(χ))
    U = χ.weights.values
    U*transpose(U)
end
function one_particle_density_matrix(χs::AbstractVector{<:QuasiParticle})
    sum(one_particle_density_matrix,χs)
end

function one_particle_density_matrix(U::AbstractMatrix{T}) where T
    N = div(size(U,1),2)
    ρ = zeros(T,2N,2N)
    for i in 1:N
        ρ += U[:,i]*transpose(U[:,i])
    end
    return ρ
end
enforce_ph_symmetry(F::Eigen) = enforce_ph_symmetry(F.values, F.vectors)
quasiparticle_adjoint(v, N=div(length(v), 2)) = [conj(v[N+1:2N]); conj(v[1:N])]
energysort(e) = e #(sign(e), abs(e))
quasiparticle_adjoint_index(n, N) = 2N + 1 - n #n+N
function enforce_ph_symmetry(es, ops; cutoff=1e-12)
    p = sortperm(es, by=energysort)
    es = es[p]
    ops = ops[:, p]
    N = div(length(es), 2)
    ph = quasiparticle_adjoint
    for k in Iterators.take(eachindex(es), N)
        k2 = quasiparticle_adjoint_index(k, N)
        if es[k] > cutoff
            @warn isapprox(es[k], -es[k2], atol=cutoff) "$(es[k]) != $(-es[k2])"
        end
        op = ops[:, k]
        op2 = ph(op)
        if abs(dot(op2, op)) < cutoff #op is not a majorana
            ops[:, k2] = op2
        else
            majplus = ph(op) + op + (ph(ops[:, k2]) + ops[:, k2]) / 2
            normalize!(majplus)
            majminus = ph(op) - op + (ph(ops[:, k2]) - ops[:, k2]) / 2
            normalize!(majminus)
            o1 =  (majplus + 1 * majminus) / sqrt(2)
            o2 =  (majplus - 1 * majminus) / sqrt(2)
            if abs(dot(o1,op)) > abs(dot(o2,op))
                ops[:, k] = o1
                ops[:, k2] = o2
            else
                ops[:, k] = o2
                ops[:, k2] = o1
            end
        end
    end
    es, ops
end

function check_ph_symmetry(es, ops; cutoff = 1e-11)
    N = div(length(es), 2)
    p = sortperm(es, by=energysort)
    inds = Iterators.take(eachindex(es), N)
    all(abs(es[p[i]] + es[p[quasiparticle_adjoint_index(i, N)]]) < cutoff for i in inds) &&
        all(quasiparticle_adjoint(ops[:, p[i]]) ≈ ops[:, p[quasiparticle_adjoint_index(i, N)]] for i in inds) &&
        ops' * ops ≈ I
end


function one_particle_density_matrix(ρ::AbstractMatrix{T}, b::FermionBasis) where {T}
    N = nbr_of_fermions(b)
    hoppings = zeros(T, N, N)
    pairings = zeros(T, N, N)
    hoppings2 = zeros(T, N, N)
    pairings2 = zeros(T, N, N)
    for (n, (f1, f2)) in enumerate(Base.product(b.dict, b.dict))
        pairings[n] += tr(ρ * f1 * f2)
        pairings2[n] += tr(ρ * f1' * f2')# -conj(pairings[n])#
        hoppings[n] += tr(ρ * f1' * f2)
        hoppings2[n] += tr(ρ * f1 * f2')
    end
    return [hoppings pairings2; pairings hoppings2]
end

function Base.:*(f1::QuasiParticle, f2::QuasiParticle; kwargs...)
    b = basis(f1)
    @assert b == basis(f2)
    sum(*(BdGFermion(first(l1), b, w1, last(l1) == :h), BdGFermion(first(l2), b, w2, last(l2) == :h); kwargs...) for ((l1, w1), (l2, w2)) in Base.product(pairs(f1.weights), pairs(f2.weights)))
end
Base.adjoint(f::QuasiParticle) = QuasiParticle(Dictionary(keys(f.weights).values, quasiparticle_adjoint(f.weights.values, nbr_of_fermions(basis(f)))), basis(f))

function Base.:+(f1::QuasiParticle, f2::QuasiParticle)
    @assert basis(f1) == basis(f2)
    QuasiParticle(map(+, f1.weights, f2.weights), basis(f1))
end
function Base.:-(f1::QuasiParticle, f2::QuasiParticle)
    @assert basis(f1) == basis(f2)
    QuasiParticle(map(-, f1.weights, f2.weights), basis(f1))
end
Base.:*(x::Number, f::QuasiParticle) = QuasiParticle(map(Base.Fix1(*, x), f.weights), basis(f))
Base.:*(f::QuasiParticle, x::Number) = QuasiParticle(map(Base.Fix2(*, x), f.weights), basis(f))
Base.:/(f::QuasiParticle, x::Number) = QuasiParticle(map(Base.Fix2(/, x), f.weights), basis(f))

QuasiParticle(f::BdGFermion) = QuasiParticle(rep(f), f.basis)
Base.promote_rule(::Type{<:BdGFermion}, ::Type{QuasiParticle{T,M,S}}) where {T,M,S} = QuasiParticle{T,M,S}
Base.convert(::Type{<:QuasiParticle}, f::BdGFermion) = QuasiParticle(f)
Base.:+(f1::AbstractBdGFermion, f2::AbstractBdGFermion) = +(promote(f1,f2)...)
Base.:-(f1::AbstractBdGFermion, f2::AbstractBdGFermion) = -(promote(f1,f2)...)
Base.:+(f1::BdGFermion, f2::BdGFermion) = QuasiParticle(f1) + QuasiParticle(f2)
Base.:-(f1::BdGFermion, f2::BdGFermion) = QuasiParticle(f1) - QuasiParticle(f2)
rep(qp::QuasiParticle) = sum((lw) -> rep(BdGFermion(first(first(lw)), basis(qp), last(lw), last(first(lw)) == :h)), pairs(qp.weights))

function many_body_fermion(f::BdGFermion, basis::FermionBasis)
    if f.hole
        return f.amp*basis[f.id]
    else
        return f.amp*basis[f.id]'
    end
end
function many_body_fermion(qp::QuasiParticle, basis::FermionBasis)
    mbferm((l,w)) = last(l) == :h ? w*basis[first(l)] : w*basis[first(l)]'
    sum(mbferm, pairs(qp.weights))
end

function ground_state_parity(vals, vecs)
    p = sortperm(vals, by = energysort)
    N = div(length(vals),2)
    pinds = p[[1:N; quasiparticle_adjoint_index.(1:N,N)]]
    sign(det(vecs[:,pinds]))
end