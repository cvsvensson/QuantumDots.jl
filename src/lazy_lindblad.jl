
struct LazyLindbladDissipator{J,J2,T,L,E} <: AbstractDissipator
    op::J
    opsquare::J2
    rate::T
    lead::L
    energies::E
end
Base.eltype(d::LazyLindbladDissipator) = promote_type(map(eltype, d.op.in)...)
function LazyLindbladDissipator(lead, energies, rate)
    op = (; in=map(op -> complex(ratetransform(op, energies, lead.T, lead.μ)), lead.jump_in),
        out=map(op -> complex(ratetransform(op, energies, lead.T, -lead.μ)), lead.jump_out))
    opsquare = map(leadops -> map(x -> Hermitian(x' * x), leadops), op)
    LazyLindbladDissipator(op, opsquare, rate, lead, energies)
end
Base.adjoint(d::LazyLindbladDissipator) = LazyLindbladDissipator(map(Base.Fix1(map, adjoint), d.op), d.opsquare, d.rate, adjoint(d.lead), d.energies)

update(d::LazyLindbladDissipator, ::SciMLBase.NullParameters) = d
function update(d::LazyLindbladDissipator, p)
    rate = get(p, :rate, d.rate)
    newlead = update_lead(d.lead, p)
    LazyLindbladDissipator(newlead, d.energies, rate)
end

function (d::LazyLindbladDissipator)(rho, p, t)
    T = promote_type(eltype(d), eltype(rho))
    out = similar(rho, T)
    d(out, rho, p, t)
end
function (d::LazyLindbladDissipator)(out, rho, p, t)
    d = update(d, p)
    mul!(out, d, rho)
    return out
end

struct LazyLindbladSystem{DS,H,C} <: AbstractOpenSystem
    dissipators::DS
    hamiltonian::H
    cache::C
end

function LazyLindbladSystem(system::OpenSystem{<:DiagonalizedHamiltonian}; rates=map(l -> 1, system.leads))
    energies = eigenvaluevector(system)
    dissipators = map((lead, rate) -> LazyLindbladDissipator(lead, energies, rate), system.leads, rates)
    LazyLindbladSystem(dissipators, system.hamiltonian, Matrix(1im * eigenvalues(system.hamiltonian)))
end
Base.adjoint(d::LazyLindbladSystem) = LazyLindbladSystem(map(adjoint, d.dissipators), -d.hamiltonian, d.cache)

function (d::LazyLindbladSystem)(rho, p, t)
    d = update(d, p)
    d * rho
end
function (d::LazyLindbladSystem)(out, rho, p, t)
    d = update(d, p)
    mul!(out, d, rho)
    return out
end

function update_lazy_lindblad_system(L::LazyLindbladSystem, p)
    _newdissipators = map(lp -> first(lp) => update(L.dissipators[first(lp)], last(lp)), collect(pairs(p)))
    newdissipators = merge(L.dissipators, _newdissipators)
    LazyLindbladSystem(newdissipators, L.hamiltonian)
end
update(L::LazyLindbladSystem, p) = update_lazy_lindblad_system(L, p)
update(L::LazyLindbladSystem, ::Union{Nothing,SciMLBase.NullParameters}) = L
update(L::LazyLindbladDissipator, ::Union{Nothing,SciMLBase.NullParameters}) = L

function LinearAlgebra.mul!(out, d::LazyLindbladDissipator, rho)
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

function LinearAlgebra.mul!(out, d::LazyLindbladSystem, _rho)
    H = eigenvalues(d.hamiltonian)
    dissipator_ops = dissipator_op_list(d)
    rho = isreal(_rho) ? complex(_rho) : _rho #Need to have complex matrices for the mul! to be non-allocating
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
    H = eigenvalues(d.hamiltonian)
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
    ops_rate = map(o -> (o..., d.rate), ops)
end
function vec_action(d::LazyLindbladSystem)
    sz = size(d.hamiltonian)
    _vec_action(u, p, t) = vec(d(reshape(u, sz...), p, t))
    _vec_action(v, u, p, t) = vec(d(reshape(v, sz...), reshape(u, sz...), p, t))
    return _vec_action
end
function _FunctionOperator(d::LazyLindbladSystem, p)
    T = eltype(d)
    v = Vector{T}(undef, prod(size(d.hamiltonian)))
    FunctionOperator(vec_action(d), v, v; islinear=true, op_adjoint=vec_action(d'))
end
function FunctionOperatorWithNormalizer(d::LazyLindbladSystem, p)
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
    FunctionOperator(vec_action, v, similar(v, length(v) + 1); islinear=true, op_adjoint=vec_action_adj)
end

function add_diagonal!(m, x)
    for n in diagind(m)
        @inbounds m[n] += x
    end
    return m
end

function LinearOperator(L::LazyLindbladSystem, p=SciMLBase.NullParameters(); normalizer=false)
    L = update(L, p)
    normalizer || return _FunctionOperator(L, p)
    return FunctionOperatorWithNormalizer(L, p)
end

identity_density_matrix(system::LazyLindbladSystem) = vec(Matrix{eltype(system)}(I, size(system.hamiltonian)...))
Base.eltype(system::LazyLindbladSystem) = promote_type(typeof(1im), eltype(system.hamiltonian))