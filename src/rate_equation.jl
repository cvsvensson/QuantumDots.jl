struct Pauli <: AbstractOpenSolver end

density_of_states(lead::NormalLead) = 1 #FIXME: put in some physics here


struct PauliSystem{A,W,I,D} <: AbstractOpenSystem
    total_master_matrix::A
    total_rate_matrix::W
    total_current_operator::I
    dissipators::D
end
Base.Matrix(P::PauliSystem) = P.total_master_matrix

struct PauliDissipator{L,W,I,D,HD} <: AbstractDissipator
    lead::L
    Win::W
    Wout::W
    Iin::I
    Iout::I
    total_master_matrix::D
    H::HD
end
Base.Matrix(d::PauliDissipator) = d.total_master_matrix

"""
    PauliDissipator(ham, lead; change_lead_basis=true)

Constructs the Pauli dissipator for a given Hamiltonian and lead.

# Arguments
- `ham`: The Hamiltonian.
- `lead`: The leads.
- `change_lead_basis`: A boolean indicating whether to change the lead basis to the energy basis. Default is `true`.
"""
function PauliDissipator(ham::H, lead; change_lead_basis=true) where {H<:DiagonalizedHamiltonian}
    energies = ham.values
    lead = change_lead_basis ? changebasis(lead, ham) : lead
    # diaglead = changebasis(lead, ham) #map(lead -> changebasis(lead, ham), leads)
    Win, Wout = get_rates(energies, lead)
    D = Win + Wout
    Iin = vec(sum(Win, dims=1))
    Iout = -vec(sum(Wout, dims=1))
    D .-= Diagonal(Iin) .- Diagonal(Iout)
    PauliDissipator{typeof(lead),typeof(Win),typeof(Iin),typeof(D),H}(lead, Win, Wout, Iin, Iout, D, ham)
end

internal_rep(u::UniformScaling, sys::PauliSystem) = u[1, 1] * ones(size(sys.total_master_matrix, 2))
internal_rep(u::AbstractMatrix, ::PauliSystem) = diag(u)
internal_rep(u::AbstractVector, ::PauliSystem) = u
tomatrix(u::AbstractVector, ::PauliSystem) = tomatrix(u, Pauli())
tomatrix(u::AbstractVector, ::Pauli) = Diagonal(u)
function LinearOperator(L::PauliSystem, args...; normalizer=false)
    A = normalizer ? add_normalizer(L.total_master_matrix) : L.total_master_matrix
    MatrixOperator(A)
end
function identity_density_matrix(system::PauliSystem)
    A = system.total_master_matrix
    fill(one(eltype(A)), size(A, 2))
end

"""
    PauliSystem(ham, leads)

Constructs a PauliSystem object from a Hamiltonian and a set of leads.
"""
PauliSystem(ham, leads) = PauliSystem(diagonalize(ham), leads)
function PauliSystem(H::DiagonalizedHamiltonian, leads)
    ds = Dict(k => PauliDissipator(H, lead) for (k, lead) in pairs(leads))
    PauliSystem(ds)
end
function PauliSystem(ds)
    first_diss = first(values(ds))
    Win = zero(first_diss.Win)
    Wout = zero(first_diss.Wout)
    D = zero(first_diss.total_master_matrix)
    Iin = zero(first_diss.Iin)
    Iout = zero(first_diss.Iout)
    P = PauliSystem(D, (; in=Win, out=Wout), (; in=Iin, out=Iout), ds)
    update_total_operators!(P)
    return P
end
update_coefficients(L::PauliSystem, p, tmp=nothing) = update_pauli_system(L, p)
update_pauli_system(L::PauliSystem, ::Union{Nothing,SciMLBase.NullParameters}) = L
function update_pauli_system(L::PauliSystem, p)
    _newdissipators = Dict(k => update_coefficients(L.dissipators[k], v) for (k, v) in pairs(p))
    newdissipators = merge(L.dissipators, _newdissipators)
    PauliSystem(newdissipators)
end
function update_coefficients(d::PauliDissipator, p, tmp=nothing)
    PauliDissipator(d.H, update_lead(d.lead, p); change_lead_basis=false)
end


function zero_total_operators!(P::PauliSystem)
    foreach(x -> fill!(x, zero(eltype(x))), (P.total_rate_matrix.in,
        P.total_rate_matrix.out,
        P.total_current_operator.in,
        P.total_current_operator.out,
        P.total_master_matrix))
end
function update_total_operators!(P::PauliSystem)
    zero_total_operators!(P)
    for d in values(P.dissipators)
        P.total_rate_matrix.in .+= d.Win
        P.total_rate_matrix.out .+= d.Wout
        P.total_current_operator.in .+= d.Iin
        P.total_current_operator.out .+= d.Iout
        P.total_master_matrix .+= d.total_master_matrix
    end
end

function get_rates(E::AbstractVector, lead::NormalLead)
    dos = density_of_states(lead)
    T = promote_type(eltype(E), eltype(lead.μ), eltype(lead.T), eltype(first(lead.jump_in)))
    Win = zeros(T, size(first(lead.jump_in))...)
    Wout = zeros(T, size(first(lead.jump_in))...)
    update_rates!(Win, lead.jump_in, lead.T, lead.μ, E; dos)
    update_rates!(Wout, lead.jump_out, lead.T, -lead.μ, E; dos)
    return Win, Wout
end

function update_rates!(W, ops, T, μ, E::AbstractVector; dos)
    for I in CartesianIndices(W)
        n1, n2 = Tuple(I)
        δE = E[n1] - E[n2]
        pf = 2π * dos * fermidirac(δE, T, μ)
        for op in ops
            W[n1, n2] += pf * abs2(op[n1, n2])
        end
    end
    return W
end

function add_normalizer(m::AbstractMatrix{T}) where {T}
    [m; fill(one(T), 1, size(m, 2))]
end

get_currents(rho, eq::PauliSystem) = get_currents(internal_rep(rho, eq), eq)
function get_currents(rho::AbstractVector, P::PauliSystem) #rho is the diagonal density matrix
    Dict(k => dot(d.Iin, rho) + dot(d.Iout, rho) for (k, d) in pairs(P.dissipators))
end


function conductance_matrix(dμ::Number, sys::PauliSystem)
    perturbations = [Dict(k => (; μ=d.lead.μ + dμ)) for (k, d) in pairs(sys.dissipators)]
    function get_current(pert)
        newsys = update_coefficients(sys, pert)
        sol = solve(StationaryStateProblem(newsys))
        currents = get_currents(sol, newsys)
        [real(currents[k]) for k in keys(sys.dissipators)]
    end
    I0 = get_current(SciMLBase.NullParameters())
    stack([(get_current(pert) .- I0) / dμ for pert in perturbations])
end

function conductance_matrix(ad::AD.FiniteDifferencesBackend, ls::AbstractOpenSystem)
    keys_iter = keys(ls.dissipators)
    μs0 = [ls.dissipators[k].lead.μ for k in keys_iter]
    function get_current(μs)
        pert = Dict(k => (; μ=μs[n]) for (n, k) in enumerate(keys_iter))
        newsys = update_coefficients(ls, pert)
        sol = solve(StationaryStateProblem(newsys))
        currents = get_currents(sol, newsys)
        [real(currents[k]) for k in keys_iter]
    end
    AD.jacobian(ad, get_current, μs0)[1]
end


function conductance_matrix(backend::AD.AbstractBackend, sys::PauliSystem, rho)
    linsolve = init(StationaryStateProblem(sys))
    func = d -> [Matrix(d), d.Iin + d.Iout]
    key_iter = keys(sys.dissipators)
    mapreduce(hcat, key_iter) do k
        d = sys.dissipators[k]
        dD = chem_derivative(backend, func, d)
        sol = solveDiffProblem!(linsolve, rho, dD[1])
        rhodiff_currents = get_currents(sol, sys)
        dissdiff_current = dot(dD[2], rho)
        [real(rhodiff_currents[k2] + dissdiff_current * (k2 == k)) for k2 in key_iter]
    end
end

function ODEProblem(system::PauliSystem, u0, tspan, p=SciMLBase.NullParameters(), args...; kwargs...)
    internalu0 = internal_rep(u0, system)
    ODEProblem(LinearOperator(system, p; kwargs...), internalu0, tspan, p, args...; kwargs...)
end

Base.size(d::PauliDissipator) = size(d.total_master_matrix)
Base.size(d::PauliSystem) = size(d.total_master_matrix)
Base.eltype(d::PauliSystem) = eltype(d.total_master_matrix)
Base.eltype(d::PauliDissipator) = eltype(d.total_master_matrix)