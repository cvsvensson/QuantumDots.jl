
struct LazyLindbladDissipator{J,J2,T,L,H,C} <: AbstractDissipator
    op::J
    opsquare::J2
    rate::T
    lead::L
    hamiltonian::H
    cache::C
end
Base.eltype(d::LazyLindbladDissipator) = promote_type(map(eltype, d.op.in)...)
function LazyLindbladDissipator(_lead::NormalLead, diagham::DiagonalizedHamiltonian, rate)
    lead = NormalLead(_lead.T, _lead.μ, complex.(collect.(_lead.jump_in)), complex.(collect.(_lead.jump_out)))
    op = (; in=map(op -> complex(ratetransform(op, diagham, lead.T, lead.μ)), lead.jump_in),
        out=map(op -> complex(ratetransform(op, diagham, lead.T, -lead.μ)), lead.jump_out))
    opsquare = map(leadops -> map(x -> x' * x, leadops), op)
    cache = similar(Matrix(first(op.in)))
    LazyLindbladDissipator(op, opsquare, rate, lead, diagham, cache)
end
Base.adjoint(d::LazyLindbladDissipator) = LazyLindbladDissipator(map(Base.Fix1(map, adjoint), d.op), d.opsquare, d.rate, adjoint(d.lead), adjoint(d.hamiltonian), d.cache)

update_dissipator(d::LazyLindbladDissipator, ::Union{Nothing,SciMLBase.NullParameters}, cache=d.cache) = d
update_dissipator!(d::LazyLindbladDissipator, ::Union{Nothing,SciMLBase.NullParameters}, cache=d.cache) = d
function update_dissipator!(d::LazyLindbladDissipator, p, cache=d.cache)
    rate = get(p, :rate, d.rate)
    newlead = update_lead(d.lead, p)
    op = (; in=map((out, op) -> complex(ratetransform!(out, cache, op, d.hamiltonian, newlead.T, newlead.μ)), d.op.in, newlead.jump_in),
        out=map((out, op) -> complex(ratetransform!(out, cache, op, d.hamiltonian, newlead.T, -newlead.μ)), d.op.out, newlead.jump_out))
    opsquare = map((outops, leadops) -> map((out, x) -> mul!(out, x', x), outops, leadops), d.opsquare, op)
    LazyLindbladDissipator(op, opsquare, rate, newlead, d.hamiltonian, d.cache)
end
function update_dissipator(d::LazyLindbladDissipator, p, cache=d.cache)
    rate = get(p, :rate, d.rate)
    newlead = update_lead(d.lead, p)
    op = (; in=map((out, op) -> complex(ratetransform!(similar(out), cache, op, d.hamiltonian, newlead.T, newlead.μ)), d.op.in, newlead.jump_in),
        out=map((out, op) -> complex(ratetransform!(similar(out), cache, op, d.hamiltonian, newlead.T, -newlead.μ)), d.op.out, newlead.jump_out))
    opsquare = map((outops, leadops) -> map((out, x) -> mul!(similar(out), x', x), outops, leadops), d.opsquare, op)
    LazyLindbladDissipator(op, opsquare, rate, newlead, d.hamiltonian, d.cache)
end

(d::LazyLindbladDissipator)(rho) = d * rho
function (d::LazyLindbladDissipator)(rho, p, t)
    T = promote_type(eltype(d), eltype(rho))
    out = similar(rho, T)
    d(out, rho, p, t)
end
function (d::LazyLindbladDissipator)(out, rho, p, t)
    d = update_dissipator!(d, p)
    mul!(out, d, rho)
    return out
end

struct LazyLindbladSystem{DS,H,NH,C} <: AbstractOpenSystem
    dissipators::DS
    hamiltonian::H
    nonhermitian_hamiltonian::NH
    cache::C
end

function LazyLindbladSystem(ham, leads::AbstractDict; rates=Dict(k => 1 for (k, v) in pairs(leads)))
    _diagham = diagonalize(ham)
    T = complex(eltype(_diagham.vectors))
    # We convert a blockdiagonal hamiltonian to a matrix, because jump operators individually are not blockdiagonal
    diagham = DiagonalizedHamiltonian(collect(_diagham.values), complex(collect(_diagham.vectors)), Matrix{T}(_diagham.original))
    dissipators = Dict(k => LazyLindbladDissipator(leads[k], diagham, rates[k]) for k in keys(leads))
    cache = -1im * diagham.vectors * first(first(first(values(dissipators)).opsquare))
    nonhermitian_hamiltonian = _nonhermitian_hamiltonian(diagham.original, dissipators)
    LazyLindbladSystem(dissipators, diagham, nonhermitian_hamiltonian, cache)
end
function Base.adjoint(d::LazyLindbladSystem)
    newdissipators = Dict(k => adjoint(v) for (k, v) in pairs(d.dissipators))
    newham = -d.hamiltonian
    newnonhermitian = _nonhermitian_hamiltonian(newham.original, newdissipators)
    LazyLindbladSystem(newdissipators, newham, newnonhermitian, d.cache)
end
function _nonhermitian_hamiltonian(H, dissipators)
    out = zeros(complex(eltype(H)), size(H))
    _nonhermitian_hamiltonian!(out, H, dissipators)
end
function _nonhermitian_hamiltonian!(out, H, dissipators)
    fill!(out, zero(eltype(out)))
    out .+= H
    for d in values(dissipators)
        rate = d.rate
        L2s = Iterators.flatten((d.opsquare.in, d.opsquare.out))
        for L2 in L2s
            out .-= (1im * rate / 2) .* L2
        end
    end
    return out
end

function (d::LazyLindbladSystem)(rho, p, t)
    d = update_coefficients(d, p)
    d * rho
end
function (d::LazyLindbladSystem)(out, rho, p, t)
    d = update_coefficients!(d, p)
    mul!(out, d, rho)
    return out
end
(d::LazyLindbladSystem)(rho) = d * rho

update_lazy_lindblad_system(L::LazyLindbladSystem, ::Union{Nothing,SciMLBase.NullParameters}) = L
update_lazy_lindblad_system!(L::LazyLindbladSystem, ::Union{Nothing,SciMLBase.NullParameters}) = L
function update_lazy_lindblad_system(L::LazyLindbladSystem, p)
    _newdissipators = Dict(k => update_dissipator(L.dissipators[k], v) for (k, v) in pairs(p))
    newdissipators = merge(L.dissipators, _newdissipators)
    nonhermitian_hamiltonian = _nonhermitian_hamiltonian(L.hamiltonian.original, newdissipators)
    LazyLindbladSystem(newdissipators, L.hamiltonian, nonhermitian_hamiltonian, L.cache)
end
function update_lazy_lindblad_system!(L::LazyLindbladSystem, p, cache=L.cache)
    for (k, v) in pairs(p)
        L.dissipators[k] = update_dissipator!(L.dissipators[k], v, cache)
    end
    _nonhermitian_hamiltonian!(L.nonhermitian_hamiltonian, L.hamiltonian.original, L.dissipators)
    L
end


function LinearAlgebra.mul!(out, d::LazyLindbladDissipator, rho)
    fill!(out, zero(eltype(out)))
    Ls = Iterators.flatten((d.op.in, d.op.out))
    L2s = Iterators.flatten((d.opsquare.in, d.opsquare.out))
    rate = d.rate
    cache = d.cache
    for (L, L2) in zip(Ls, L2s)
        mul!(cache, L, rho)
        mul!(out, cache, L', rate, 1)
        mul!(out, L2, rho, -rate / 2, 1)
        mul!(out, rho, L2, -rate / 2, 1)
        # out .+= rate * (L * rho * L' .- 1 / 2 .* (L2 * rho .+ rho * L2))
    end
    return out
end
function Base.:*(d::LazyLindbladDissipator, rho::AbstractMatrix)
    T = promote_type(eltype(rho), eltype(d))
    out = zeros(T, size(rho))
    mul!(out, d, rho)
    return out
end

function LinearAlgebra.mul!(out, d::LazyLindbladSystem, rho)
    Heff = d.nonhermitian_hamiltonian
    mul!(out, Heff, rho, -1im, 0)
    mul!(out, rho, Heff', 1im, 1)
    # out = -1im .* (Heff * rho .- rho * Heff')
    cache = d.cache
    for d in values(d.dissipators)
        rate = d.rate
        Ls = Iterators.flatten((d.op.in, d.op.out))
        for L in Ls
            mul!(cache, L, rho)
            mul!(out, cache, L', rate, 1)
            #out .+= rate .* (L * rho * L')
        end
    end
    return out
end
function Base.:*(d::LazyLindbladSystem, rho::AbstractMatrix)
    T = promote_type(eltype(rho), eltype(d))
    out = zeros(T, size(rho))
    mul!(out, d, rho)
    return out
end
function add_diagonal!(m, x)
    for n in diagind(m)
        @inbounds m[n] += x
    end
    return m
end

identity_density_matrix(system::LazyLindbladSystem) = vec(Matrix{eltype(system)}(I, size(system.hamiltonian)...))
Base.eltype(system::LazyLindbladSystem) = promote_type(typeof(1im), eltype(system.hamiltonian))

function ODEProblem(system::LazyLindbladSystem, u0::AbstractVector, tspan, p=SciMLBase.NullParameters(), args...; kwargs...)
    SciMLBase.ODEProblem(LinearOperator(system, p; kwargs...), u0, tspan, p, args...; kwargs...)
end

internal_rep(m::AbstractMatrix, ::LazyLindbladSystem) = m
internal_rep(v::AbstractVector, ls::LazyLindbladSystem) = reshape(v, size(ls.hamiltonian)...)
tomatrix(rho::AbstractMatrix, ::LazyLindbladSystem) = rho
tomatrix(rho::AbstractVector, ls::LazyLindbladSystem) = reshape(rho, size(ls.hamiltonian)...)

## Stationary state and vector action
function LinearOperator(L::LazyLindbladSystem, p=SciMLBase.NullParameters(); normalizer=false, kwargs...)
    L = update_lazy_lindblad_system(L, p)
    normalizer || return vec_FunctionOperator(L; p, kwargs...)
    return FunctionOperatorWithNormalizer(L; p, kwargs...)
end

function vec_action(d::LazyLindbladSystem)
    sz = size(d.hamiltonian)
    _vec_action(u, p, t) = vec(d(reshape(u, sz...), p, t))
    _vec_action(v, u, p, t) = vec(d(reshape(v, sz...), reshape(u, sz...), p, t))
    return _vec_action
end

function vec_FunctionOperator(d::LazyLindbladSystem; p=SciMLBase.NullParameters(), kwargs...)
    T = eltype(d)
    v = Vector{T}(undef, prod(size(d.hamiltonian)))
    FunctionOperator(vec_action(d), v, v; islinear=true, op_adjoint=vec_action(d'), p, kwargs...)
end
function FunctionOperatorWithNormalizer(d::LazyLindbladSystem; p=SciMLBase.NullParameters(), kwargs...)
    sz = size(d.hamiltonian)
    function vec_action(u, p, t)
        um = reshape(u, sz...)
        vm = d(um, p, t)
        v = vec(vm)
        push!(v, tr(um))
        return v
    end
    function vec_action(v, u, p, t)
        vm = reshape(@view(v[1:end-1]), sz...)
        um = reshape(u, sz...)
        d(vm, um, p, t)
        v[end] = tr(um)
        return v
    end
    dadj = d'
    function vec_action_adj(u, p, t)
        vm = dadj(reshape(@view(u[1:end-1]), sz...), p, t)
        add_diagonal!(vm, u[end])
        vec(vm)
    end
    function vec_action_adj(v, u, p, t)
        um = reshape(@view(u[1:end-1]), sz...)
        vm = reshape(v, sz...)
        dadj(vm, um, p, t)
        add_diagonal!(vm, u[end])
        vec(vm)
    end
    T = eltype(d)
    v = Vector{T}(undef, prod(size(d.hamiltonian)))
    FunctionOperator(vec_action, v, similar(v, length(v) + 1); islinear=true, op_adjoint=vec_action_adj, p, kwargs...)
end


## SciML interface
update_coefficients(d::LazyLindbladDissipator, p, t=nothing) = update_dissipator(d, p)
update_coefficients!(d::LazyLindbladDissipator, p, t=nothing) = update_dissipator!(d, p)
update_coefficients(L::LazyLindbladSystem, p, t=nothing) = update_lazy_lindblad_system(L, p)
update_coefficients!(L::LazyLindbladSystem, p, t=nothing) = update_lazy_lindblad_system!(L, p)