
density_of_states(lead::NormalLead) = 1 #FIXME: put in some physics here

prepare_rate_equations(H::OpenSystem) = prepare_rate_equations(diagonalize(H))
function prepare_rate_equations(H::OpenSystem{<:DiagonalizedHamiltonian})
    sum(prepare_rate_equations(diag(eigenvalues(H)), lead) for lead in leads(H))
end

struct RateEquation{W,A,I,L}
    rate_matrix::W
    master_matrix::A
    current_operator::I
    label::L
end
struct RateEquations{W,A,I,R}
    total_rate_matrix::W
    total_master_matrix::A
    total_current_operator::I
    rate_equations::R
end
Base.:+(r1::RateEquation, r2::RateEquation) = RateEquations(r1.rate_matrix .+ r2.rate_matrix, r1.master_matrix + r2.master_matrix, r1.current_operator + r2.current_operator, (r1, r2))
Base.:+(r1::RateEquation, r2::RateEquations) = RateEquations(r1.rate_matrix .+ r2.total_rate_matrix, r1.master_matrix + r2.total_master_matrix, r1.current_operator + r2.total_current_operator, (r1, r2.rate_equations...))
Base.:+(r1::RateEquations, r2::RateEquation) = RateEquations(r2.rate_matrix .+ r1.total_rate_matrix, r2.master_matrix + r1.total_master_matrix, r2.current_operator + r1.total_current_operator, (r1.rate_equations..., r2))
Base.:+(r1::RateEquations, r2::RateEquations) = RateEquations(r1.total_rate_matrix .+ r2.total_rate_matrix, r1.total_master_matrix + r2.total_master_matrix, r1.total_current_operator + r2.total_current_operator, (r1.rate_equations..., r2.rate_equations...))

function prepare_rate_equations(E::AbstractVector, lead::NormalLead)
    W, A, I = _prepare_rate_equations(E, lead)
    return RateEquation(W, A, I, lead.label)
end
function _prepare_rate_equations(E::AbstractVector, lead::NormalLead)
    Tin = lead.jump_in
    Tout = lead.jump_out
    Win = zeros(Float64, size(Tin))
    Wout = zeros(Float64, size(Tin))
    dos = density_of_states(lead)
    T = lead.temperature
    μ = lead.chemical_potential
    for I in CartesianIndices(Win)
        n1, n2 = Tuple(I)
        δE = E[n1] - E[n2]
        Win[n1, n2] = 2π * dos * abs2(Tin[n1, n2]) * QuantumDots.fermidirac(δE, T, μ)
        Wout[n1, n2] = 2π * dos * abs2(Tout[n1, n2]) * (1 - QuantumDots.fermidirac(-δE, T, μ))#*QuantumDots.fermidirac(E[n1]-E[n2],T,-μ)
    end
    D = (Win + Wout) - Diagonal(vec(sum(Win + Wout, dims=1)))
    I = vec(sum(Win - Wout, dims=1))
    return (Win,Wout), D, I
end
function add_normalizer(m::AbstractMatrix{T}) where {T}
    [m; fill(one(T), size(m, 2))']
end

function stationary_state(eq::RateEquations, alg = KrylovJL_LSMR(); kwargs...)
    M = add_normalizer(eq.total_master_matrix)
    v0 = zeros(eltype(M),size(M,2))
    push!(v0,one(eltype(M)))
    prob = LinearProblem(M, v0)
    sol = solve(prob, alg; kwargs...)
    return sol
end
get_currents(eq::RateEquations,alg = KrylovJL_LSMR(); kwargs...) = get_currents(stationary_state(eq,alg); kwargs...)
function get_currents(diagonal_density_matrix::AbstractVector, eq::RateEquations)
    labels = [eq.label for eq in eq.rate_equations]
    currents = [dot(eq.current_operator, diagonal_density_matrix) for eq in eq.rate_equations]
    if all(map(!ismissing, labels))
        return Dict(zip(labels,currents))
    else
        return currents
    end
end
