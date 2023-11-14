
struct LazyLindbladDissipator{J,J2,T,L,H} <: AbstractDissipator
    op::J
    opsquare::J2
    rate::T
    lead::L
    hamiltonian::H
end
Base.eltype(d::LazyLindbladDissipator) = promote_type(map(eltype, d.op.in)...)
function LazyLindbladDissipator(lead::NormalLead, diagham, rate)
    op = (; in=map(op -> complex(ratetransform(op, diagham, lead.T, lead.μ)), lead.jump_in),
        out=map(op -> complex(ratetransform(op, diagham, lead.T, -lead.μ)), lead.jump_out))
    opsquare = map(leadops -> map(x -> x' * x, leadops), op)
    LazyLindbladDissipator(op, opsquare, rate, lead, diagham)
end
Base.adjoint(d::LazyLindbladDissipator) = LazyLindbladDissipator(map(Base.Fix1(map, adjoint), d.op), d.opsquare, d.rate, adjoint(d.lead), adjoint(d.hamiltonian))

update_coefficients(d::LazyLindbladDissipator, ::SciMLBase.NullParameters, t=nothing) = d
function update_coefficients(d::LazyLindbladDissipator, p, t=nothing)
    rate = get(p, :rate, d.rate)
    newlead = update_lead(d.lead, p)
    LazyLindbladDissipator(newlead, d.hamiltonian, rate)
end

(d::LazyLindbladDissipator)(rho) = d * rho
function (d::LazyLindbladDissipator)(rho, p, t)
    T = promote_type(eltype(d), eltype(rho))
    out = similar(rho, T)
    d(out, rho, p, t)
end
function (d::LazyLindbladDissipator)(out, rho, p, t)
    d = update_coefficients(d, p)
    mul!(out, d, rho)
    return out
end

struct LazyLindbladSystem{DS,H,C} <: AbstractOpenSystem
    dissipators::DS
    hamiltonian::H
    cache::C
end

function LazyLindbladSystem(ham, leads; rates=map(l -> 1, leads))
    _diagham = diagonalize(ham)
    T = complex(eltype(_diagham.original))
    diagham = DiagonalizedHamiltonian(_diagham.values, _diagham.vectors, Matrix{T}(_diagham.original))
    dissipators = map((lead, rate) -> LazyLindbladDissipator(lead, diagham, rate), leads, rates)
    cache = -1im * diagham.vectors * first(first(first(dissipators).opsquare))
    LazyLindbladSystem(dissipators, diagham, cache)
end
Base.adjoint(d::LazyLindbladSystem) = LazyLindbladSystem(map(adjoint, d.dissipators), -d.hamiltonian, d.cache)

function (d::LazyLindbladSystem)(rho, p, t)
    d = update_coefficients(d, p)
    d * rho
end
function (d::LazyLindbladSystem)(out, rho, p, t)
    d = update_coefficients(d, p)
    mul!(out, d, rho)
    return out
end
(d::LazyLindbladSystem)(rho) = d * rho

function update_lazy_lindblad_system(L::LazyLindbladSystem, p)
    _newdissipators = map(lp -> first(lp) => update_coefficients(L.dissipators[first(lp)], last(lp)), collect(pairs(p)))
    newdissipators = merge(L.dissipators, _newdissipators)
    LazyLindbladSystem(newdissipators, L.hamiltonian, L.cache)
end

update_coefficients(L::LazyLindbladSystem, p) = update_lazy_lindblad_system(L, p)
update_coefficients(L::LazyLindbladSystem, ::Union{Nothing,SciMLBase.NullParameters}) = L
update_coefficients(L::LazyLindbladDissipator, ::Union{Nothing,SciMLBase.NullParameters}) = L
update_coefficients!(L::LazyLindbladSystem, p) = update_lazy_lindblad_system(L, p)
update_coefficients!(L::LazyLindbladSystem, ::Union{Nothing,SciMLBase.NullParameters}) = L
update_coefficients!(L::LazyLindbladDissipator, ::Union{Nothing,SciMLBase.NullParameters}) = L

function LinearAlgebra.mul!(out, d::LazyLindbladDissipator, rho)
    fill!(out, zero(eltype(out)))
    for (L, L2, rate) in dissipator_op_list(d)
        out .+= rate .* (L * rho * L' .- 1 / 2 .* (L2 * rho .+ rho * L2))
    end
    return out
end
function Base.:*(d::LazyLindbladDissipator, rho)
    ops = dissipator_op_list(d)
    (L, L2, rate) = first(ops)
    out = rate .* (L * rho * L' .- 1 / 2 .* (L2 * rho .+ rho * L2))
    for (L, L2, rate) in ops[2:end]
        out .+= rate .* (L * rho * L' .- 1 / 2 .* (L2 * rho .+ rho * L2))
    end
    return out
end

function LinearAlgebra.mul!(out, d::LazyLindbladSystem, rho)
    H = original_hamiltonian(d.hamiltonian)
    dissipator_ops = dissipator_op_list(d)
    mul!(out, H, rho, -1im, 0)
    mul!(out, rho, H, 1im, 1)
    cache = d.cache
    for (L, L2, rate) in dissipator_ops
        mul!(cache, L, rho)
        mul!(out, cache, L', rate, 1)
        mul!(out, L2, rho, -rate / 2, 1)
        mul!(out, rho, L2, -rate / 2, 1)
    end
    return out
end
function Base.:*(d::LazyLindbladSystem, rho)
    H = d.hamiltonian.original
    dissipator_ops = dissipator_op_list(d)
    out = -1im .* (H * rho .- rho * H)
    for (L, L2, rate) in dissipator_ops
        out .+= rate .* (L * rho * L' .- 1 / 2 .* (L2 * rho .+ rho * L2))
    end
    return out
end
dissipator_op_list(d::LazyLindbladSystem) = mapreduce(dissipator_op_list, vcat, d.dissipators)
function dissipator_op_list(d::LazyLindbladDissipator)
    ops = vcat(collect(zip(d.op.in, d.opsquare.in)), collect(zip(d.op.out, d.opsquare.out)))
    map(o -> (o..., d.rate), ops)
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

function add_diagonal!(m, x)
    for n in diagind(m)
        @inbounds m[n] += x
    end
    return m
end

function LinearOperator(L::LazyLindbladSystem, p=SciMLBase.NullParameters(); normalizer=false, kwargs...)
    L = update_coefficients(L, p)
    normalizer || return vec_FunctionOperator(L; p, kwargs...)
    return FunctionOperatorWithNormalizer(L; p, kwargs...)
end

identity_density_matrix(system::LazyLindbladSystem) = vec(Matrix{eltype(system)}(I, size(system.hamiltonian)...))
Base.eltype(system::LazyLindbladSystem) = promote_type(typeof(1im), eltype(system.hamiltonian))


function ODEProblem(system::LazyLindbladSystem, u0::AbstractVector, tspan, p=SciMLBase.NullParameters(), args...; kwargs...)
    SciMLBase.ODEProblem(LinearOperator(system, p; kwargs...), u0, tspan, p, args...; kwargs...)
end
