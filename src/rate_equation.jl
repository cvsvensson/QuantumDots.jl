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

PauliSystem(ham, leads) = PauliSystem(diagonalize(ham), leads)
function PauliSystem(H::DiagonalizedHamiltonian, leads)
    ds = map(l -> PauliDissipator(H, l), leads)
    PauliSystem(ds)
end
function PauliSystem(ds)
    Win = zero(first(ds).Win)
    Wout = zero(first(ds).Wout)
    D = zero(first(ds).total_master_matrix)
    Iin = zero(first(ds).Iin)
    Iout = zero(first(ds).Iout)
    P = PauliSystem(D, (; in=Win, out=Wout), (; in=Iin, out=Iout), ds)
    update_total_operators!(P)
    return P
end
update_coefficients(L::PauliSystem, p, tmp=nothing) = update_pauli_system(L, p)
update_pauli_system(L::PauliSystem, ::SciMLBase.NullParameters) = L
function update_pauli_system(L::PauliSystem, p)
    _newdissipators = map(lp -> first(lp) => update_coefficients(L.dissipators[first(lp)], last(lp)), collect(pairs(p)))
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
    for d in P.dissipators
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
    map(d -> dot(d.Iin, rho) + dot(d.Iout, rho), P.dissipators)
end


function conductance_matrix(dμ::Number, sys::PauliSystem)
    perturbations = map(d -> (; μ=d.lead.μ + dμ), sys.dissipators)
    function get_current(pert)
        newsys = update_coefficients(sys, pert)
        sol = solve(StationaryStateProblem(newsys))
        real(collect(get_currents(sol, newsys)))
    end
    I0 = get_current(SciMLBase.NullParameters())
    stack(map(key -> (get_current(perturbations[[key]]) .- I0) / dμ, keys(perturbations)))
end

function conductance_matrix(ad::AD.FiniteDifferencesBackend, ls::AbstractOpenSystem)
    μs0 = [d.lead.μ for d in ls.dissipators]
    function get_current(μs)
        count = 0
        pert = map(d -> (; μ=μs[count+=1]), ls.dissipators)
        newsys = update_coefficients(ls, pert)
        sol = solve(StationaryStateProblem(newsys))
        real(collect(get_currents(sol, newsys)))
    end
    AD.jacobian(ad, get_current, μs0)[1]
end


function conductance_matrix(backend::AD.AbstractBackend, sys::PauliSystem, rho)
    dDs = [chem_derivative(backend, d -> [Matrix(d), d.Iin + d.Iout], d) for d in sys.dissipators]
    linsolve = init(StationaryStateProblem(sys))
    rhodiff = stack([collect(get_currents(solveDiffProblem!(linsolve, rho, dD[1]), sys)) for dD in dDs])
    dissdiff = Diagonal([dot(dD[2], rho) for dD in dDs])
    return dissdiff + rhodiff
end

function ODEProblem(system::PauliSystem, u0, tspan, p=SciMLBase.NullParameters(), args...; kwargs...)
    internalu0 = internal_rep(u0, system)
    ODEProblem(LinearOperator(system, p; kwargs...), internalu0, tspan, p, args...; kwargs...)
end

Base.size(d::PauliDissipator) = size(d.total_master_matrix)
Base.size(d::PauliSystem) = size(d.total_master_matrix)
Base.eltype(d::PauliSystem) = eltype(d.total_master_matrix)
Base.eltype(d::PauliDissipator) = eltype(d.total_master_matrix)