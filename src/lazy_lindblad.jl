
struct LazyLindbladDissipator{J,J2,T,L,E} <: AbstractDissipator
    op::J
    opsquare::J2
    rate::T
    lead::L
    energies::E
end
function LazyLindbladDissipator(lead, energies, rate)
    op = (; in=map(op -> ratetransform(op, energies, lead.T, lead.μ), lead.jump_in),
        out=map(op -> ratetransform(op, energies, lead.T, -lead.μ), lead.jump_out))
    opsquare = map(leadops -> map(x -> Hermitian(x' * x), leadops), op)
    LazyLindbladDissipator(op, opsquare, rate, lead, energies)
end
Base.adjoint(d::LazyLindbladDissipator) = LazyLindbladDissipator(map(Base.Fix1(map,adjoint),d.op), d.opsquare, d.rate, adjoint(d.lead), d.energies)

update(d::LazyLindbladDissipator, ::SciMLBase.NullParameters) = d
function update(d::LazyLindbladDissipator, p)
    rate = get(p, :rate, d.rate)
    newlead = update_lead(d.lead, p)
    LazyLindbladDissipator(newlead, d.energies, rate)
end

function (d::LazyLindbladDissipator)(rho, p, t)
    T = promote_type(eltype(L), eltype(rho))
    out = similar(L, T)
    fill!(out, zero(T))
    d(out, rho, p, t)
end
function (d::LazyLindbladDissipator)(out, rho, p, t)
    d = update(d, p)
    cache = similar(out)
    foreach((op, opsquare) -> dissipator_action!(out, rho, op, opsquare, d.rate, cache), d.op.in, d.opsquare.in)
    foreach((op, opsquare) -> dissipator_action!(out, rho, op, opsquare, d.rate, cache), d.op.out, d.opsquare.out)
    return out
end

function dissipator_action!(out, rho, L, L2, rate, cache)
    mul!(cache, L, rho)
    mul!(out, cache, L', rate, 1)
    mul!(out, L2, rho, -rate / 2, 1)
    mul!(out, rho, transpose(L2), -rate / 2, 1)
end
function commutator_action!(out, rho, H)
    mul!(out, H, rho, -1im, 1)
    mul!(out, rho, H, 1im, 1)
end


struct LazyLindbladSystem{DS,H} <: AbstractOpenSystem
    dissipators::DS
    hamiltonian::H
end

function LazyLindbladSystem(system::OpenSystem{<:DiagonalizedHamiltonian}; rates=map(l -> 1, system.leads))
    energies = eigenvaluevector(system)
    dissipators = map((lead, rate) -> LazyLindbladDissipator(lead, energies, rate), system.leads, rates)
    LazyLindbladSystem(dissipators, system.hamiltonian)
end
Base.adjoint(d::LazyLindbladSystem) = LazyLindbladSystem(map(adjoint,d.dissipators), -d.hamiltonian)

function (d::LazyLindbladSystem)(rho, p, t)
    L = eigenvectors(d.hamiltonian)
    T = promote_type(typeof(1im), eltype(L), eltype(rho))
    out = similar(L, T)
    fill!(out, zero(T))
    d(out, rho, p, t)
end
function (d::LazyLindbladSystem)(out, rho, p, t)
    d = update(d, p)
    commutator_action!(out, rho, eigenvalues(d.hamiltonian))
    map(d -> d(out, rho, SciMLBase.NullParameters(), t), d.dissipators)
    return out
end

function update_lazy_lindblad_system(L::LazyLindbladSystem, p)
    _newdissipators = map(lp -> first(lp) => update(L.dissipators[first(lp)], last(lp)), collect(pairs(p)))
    newdissipators = merge(L.dissipators, _newdissipators)
    LazyLindbladSystem(newdissipators, L.hamiltonian)
end
update(L::LazyLindbladSystem, p) = update_lazy_lindblad_system(L, p)
update(L::LazyLindbladSystem, ::Union{Nothing,SciMLBase.NullParameters}) = L

mul!(v, d::LazyLindbladSystem, u) = d(v, u, nothing, nothing)
mul(d::LazyLindbladSystem, u) = d(u, nothing, nothing)

function vec_action(d::LazyLindbladSystem)
    sz = size(d.hamiltonian)
    _vec_action(u, p, t) = vec(d(reshape(u, sz...), p, t))
    _vec_action(v, u, p, t) = vec(d(reshape(v, sz...), reshape(u, sz...), p, t))
    return _vec_action
end
function _FunctionOperator(d::LazyLindbladSystem, p)
    T = eltype(d)
    v = Vector{T}(undef,prod(size(d.hamiltonian)))
    FunctionOperator(vec_action(d), v, v; islinear=true, op_adjoint = vec_action(d'))
end
function FunctionOperatorWithNormalizer(d::LazyLindbladSystem, p)
    sz = size(d.hamiltonian)
    function vec_action(u, p, t)
        vm = (d(reshape(u, sz...), p, t))
        push!(vec(vm), tr(vm))
    end
    function vec_action(v, u, p, t)
        vm = reshape(@view(v[1:end-1]), sz...)
        um = reshape(u, sz...)
        d(vm, um, p, t)
        v[end] = tr(vm)
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
    v = Vector{T}(undef,prod(size(d.hamiltonian)))
    FunctionOperator(vec_action, v, similar(v, length(v) + 1); islinear=true, op_adjoint = vec_action_adj)
end

function add_diagonal!(m,x)
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