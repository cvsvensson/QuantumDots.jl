struct Pauli <: AbstractOpenSolver end

density_of_states(lead::NormalLead) = 1 #FIXME: put in some physics here
prepare_rate_equations(H::OpenSystem) = prepare_rate_equations(diagonalize(H))
function prepare_rate_equations(H::OpenSystem{<:DiagonalizedHamiltonian})
    sum(prepare_rate_equations(H, lead) for lead in leads(H))
end

struct RateEquation{W,A,I,L,S}
    rate_matrix::W
    master_matrix::A
    current_operator::I
    label::L
    system::S
end
struct PauliSystem{W,A,I,R,S} <: AbstractOpenSystem
    total_rate_matrix::W
    total_master_matrix::A
    total_current_operator::I
    rate_equations::R
    system::S
end

function LinearProblem(::Pauli, system::OpenSystem; kwargs...)
    system = prepare_rate_equations(system; kwargs...)
    LinearProblem(system; kwargs...)
end
function ReshaperProblem(prob, system::AbstractOpenSystem)
    int = Base.Fix2(internal_rep, system)
    ext = Base.Fix2(external_rep, system)
    ReshaperProblem(prob, int, ext)
end
internal_rep(u::AbstractMatrix, ::PauliSystem) = diag(u)
internal_rep(u::AbstractVector, ::PauliSystem) = u
internal_rep(u::ReshaperSolution{<:LinearSolution}, ::PauliSystem) = u.sol
external_rep(u::AbstractVector, ::PauliSystem) = Diagonal(u)
external_rep(u::AbstractMatrix, ::PauliSystem) = u

LinearOperator(system::PauliSystem; kwargs...) = LinearOperator(system.total_master_matrix; kwargs...)
LinearOperatorWithNormalizer(system::PauliSystem; kwargs...) = LinearOperator(add_normalizer(system.total_master_matrix); kwargs...)

Base.reshape(rho::AbstractVector, ::PauliSystem) = Diagonal(rho)
function identity_density_matrix(system::PauliSystem)
    A = system.total_master_matrix
    fill(one(eltype(A)), size(A, 2))
end


Base.:+(r1::RateEquation, r2::RateEquation) = PauliSystem(r1.rate_matrix .+ r2.rate_matrix, r1.master_matrix + r2.master_matrix, r1.current_operator .+ r2.current_operator, (r1, r2), r1.system)
Base.:+(r1::RateEquation, r2::PauliSystem) = PauliSystem(r1.rate_matrix .+ r2.total_rate_matrix, r1.master_matrix + r2.total_master_matrix, r1.current_operator .+ r2.total_current_operator, (r1, r2.rate_equations...), r1.system)
Base.:+(r1::PauliSystem, r2::RateEquation) = PauliSystem(r2.rate_matrix .+ r1.total_rate_matrix, r2.master_matrix + r1.total_master_matrix, r2.current_operator .+ r1.total_current_operator, (r1.rate_equations..., r2), r1.system)
Base.:+(r1::PauliSystem, r2::PauliSystem) = PauliSystem(r1.total_rate_matrix .+ r2.total_rate_matrix, r1.total_master_matrix + r2.total_master_matrix, r1.total_current_operator .+ r2.total_current_operator, (r1.rate_equations..., r2.rate_equations...), r1.system)

function prepare_rate_equations(system::AbstractOpenSystem, lead::NormalLead)
    W, A, I = _prepare_rate_equations(eigenvaluevector(system), lead)
    return RateEquation(W, A, I, lead.label, system)
end
function _prepare_rate_equations(E::AbstractVector, lead::NormalLead)
    Tin = lead.jump_in
    Tout = lead.jump_out
    Win = zeros(Float64, size(Tin))
    Wout = zeros(Float64, size(Tin))
    dos = density_of_states(lead)
    T = temperature(lead)
    μ = chemical_potential(lead)
    for I in CartesianIndices(Win)
        n1, n2 = Tuple(I)
        δE = E[n1] - E[n2]
        Win[n1, n2] = 2π * dos * abs2(Tin[n1, n2]) * QuantumDots.fermidirac(δE, T, μ)
        Wout[n1, n2] = 2π * dos * abs2(Tout[n1, n2]) * (1 - QuantumDots.fermidirac(-δE, T, μ))#*QuantumDots.fermidirac(E[n1]-E[n2],T,-μ)
    end
    D = (Win + Wout) - Diagonal(vec(sum(Win + Wout, dims=1)))
    Iin, Iout = (vec(sum(Win, dims=1)), vec(sum(-Wout, dims=1)))
    return (Win, Wout), D, (Iin, Iout)
end
function add_normalizer(m::AbstractMatrix{T}) where {T}
    [m; fill(one(T), size(m, 2))']
end

get_currents(eq::PauliSystem, alg=nothing; kwargs...) = get_currents(stationary_state(eq, alg), eq; kwargs...)
get_currents(rho, eq::PauliSystem) = get_currents(internal_rep(rho,eq), eq)
function get_currents(diagonal_density_matrix::AbstractVector, eq::PauliSystem)
    currents = [(; in=dot(eq.current_operator[1], diagonal_density_matrix), out=dot(eq.current_operator[2], diagonal_density_matrix)) for eq in eq.rate_equations]
    [merge(c, (; total=c.in + c.out, label=eq.label)) for (c, eq) in zip(currents, eq.rate_equations)]
end
