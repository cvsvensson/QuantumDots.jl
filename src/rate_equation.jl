struct Pauli <: AbstractOpenSolver end


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

update_coefficients(L::PauliSystem, ::Union{Nothing,SciMLBase.NullParameters}) = L
function update_coefficients(L::PauliSystem, p, tmp=nothing)
    _newdissipators = Dict(k => update_coefficients(L.dissipators[k], v) for (k, v) in pairs(p))
    newdissipators = merge(L.dissipators, _newdissipators)
    PauliSystem(newdissipators)
end
function update_coefficients(d::PauliDissipator, p, tmp=nothing)
    PauliDissipator(d.H, update_lead(d.lead, p); change_lead_basis=false)
end
function update_coefficients!(d::PauliDissipator{L,W,I,D,HD}, p) where {L,W,I,D,HD}
    lead = update_lead(d.lead, p)
    energies = d.H.values
    update_rates!(d.Win, lead.jump_in, lead.T, lead.μ, energies)
    update_rates!(d.Wout, lead.jump_out, lead.T, -lead.μ, energies)
    d.total_master_matrix .= d.Win .+ d.Wout
    sum!(d.Iin', d.Win)
    sum!(d.Iout', d.Wout)
    d.Iout .*= -1
    d.total_master_matrix .-= Diagonal(d.Iin) .- Diagonal(d.Iout)
    PauliDissipator{L,W,I,D,HD}(lead, d.Win, d.Wout, d.Iin, d.Iout, d.total_master_matrix, d.H)
end
update_coefficients!(L::PauliSystem, ::Union{Nothing,SciMLBase.NullParameters}) = L

function update_coefficients!(L::PauliSystem, p)
    for (k, v) in pairs(p)
        L.dissipators[k] = update_coefficients!(L.dissipators[k], v)
    end
    update_total_operators!(L)
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
    T = promote_type(eltype(E), eltype(lead.μ), eltype(lead.T), eltype(first(lead.jump_in)))
    Win = zeros(T, size(first(lead.jump_in))...)
    Wout = zeros(T, size(first(lead.jump_in))...)
    update_rates!(Win, lead.jump_in, lead.T, lead.μ, E)
    update_rates!(Wout, lead.jump_out, lead.T, -lead.μ, E)
    return Win, Wout
end

function update_rates!(W, ops, T, μ, E::AbstractVector)
    for I in CartesianIndices(W)
        n1, n2 = Tuple(I)
        δE = E[n1] - E[n2]
        pf = fermidirac(δE, T, μ)
        for op in ops
            W[n1, n2] = pf * abs2(op[n1, n2])
        end
    end
    return W
end

function add_normalizer(m::AbstractMatrix{T}) where {T}
    [m; fill(one(T), 1, size(m, 2))]
end

get_currents(rho, eq::PauliSystem) = get_currents(internal_rep(rho, eq), eq)
function get_currents(rho::AbstractVector, P::PauliSystem) #rho is the diagonal density matrix
    ks = collect(keys(P.dissipators))
    AxisKeys.KeyedArray([dot(P.dissipators[k].Iin, rho) + dot(P.dissipators[k].Iout, rho) for k in ks], ks)
end


function conductance_matrix(dμ::Number, sys::PauliSystem)
    perturbations = [Dict(k => (; μ=d.lead.μ + dμ)) for (k, d) in pairs(sys.dissipators)]
    function get_current(pert)
        newsys = update_coefficients(sys, pert)
        sol = solve(StationaryStateProblem(newsys))
        get_currents(sol, newsys)
    end
    I0 = get_current(SciMLBase.NullParameters())
    ks = AxisKeys.axiskeys(I0, 1)
    N = length(I0)
    T = real(eltype(I0))
    G = AxisKeys.wrapdims(AxisKeys.KeyedArray(Matrix{T}(undef, N, N), (ks, ks)), :∂Iᵢ, :∂μⱼ)

    for dict in perturbations
        diff_currents = (get_current(dict) .- I0) ./ dμ
        key = first(only(dict))
        G(:, key) .= real.(diff_currents)
    end
    return G
end

function conductance_matrix(ad::AD.FiniteDifferencesBackend, sys::PauliSystem)
    keys_iter = collect(keys(sys.dissipators))
    μs0 = [sys.dissipators[k].lead.μ for k in keys_iter]
    function get_current(μs)
        pert = Dict(k => (; μ=μs[n]) for (n, k) in enumerate(keys_iter))
        newsys = update_coefficients(sys, pert)
        sol = solve(StationaryStateProblem(newsys))
        currents = get_currents(sol, newsys)
        [real(currents(k)) for k in keys_iter]
    end
    J = AD.jacobian(ad, get_current, μs0)[1]
    AxisKeys.wrapdims(AxisKeys.KeyedArray(J, (keys_iter, keys_iter)), :∂Iᵢ, :∂μⱼ)
end


function conductance_matrix(backend::AD.AbstractBackend, sys::PauliSystem, rho)
    linsolve = init(StationaryStateProblem(sys))
    func = d -> [Matrix(d), d.Iin + d.Iout]
    key_iter = collect(keys(sys.dissipators))
    N = length(key_iter)
    T = real(eltype(sys))
    G = AxisKeys.wrapdims(AxisKeys.KeyedArray(Matrix{T}(undef, N, N), (key_iter, key_iter)), :∂Iᵢ, :∂μⱼ)

    for k in key_iter
        d = sys.dissipators[k]
        dD = chem_derivative(backend, func, d)
        sol = solveDiffProblem!(linsolve, rho, dD[1])
        rhodiff_currents = get_currents(sol, sys)
        dissdiff_current = dot(dD[2], rho)
        rhodiff_currents[AxisKeys.Key(k)] += dissdiff_current
        G(:, k) .= real.(rhodiff_currents)
    end
    G

end

function ODEProblem(system::PauliSystem, u0, tspan, p=SciMLBase.NullParameters(), args...; kwargs...)
    internalu0 = internal_rep(u0, system)
    ODEProblem(LinearOperator(system, p; kwargs...), internalu0, tspan, p, args...; kwargs...)
end

Base.size(d::PauliDissipator) = size(d.total_master_matrix)
Base.size(d::PauliSystem) = size(d.total_master_matrix)
Base.eltype(d::PauliSystem) = eltype(d.total_master_matrix)
Base.eltype(d::PauliDissipator) = eltype(d.total_master_matrix)

@testitem "Pauli updating" begin
    using LinearAlgebra
    import QuantumDots: update_coefficients, update_coefficients!
    H = Hermitian(rand(ComplexF64, 4, 4))
    leads = Dict([:lead => NormalLead(rand(ComplexF64, 4, 4); T=0.1, μ=0.5)])
    P1 = PauliSystem(H, leads)
    P2 = update_coefficients(P1, Dict(:lead => Dict(:T => 0.2, :μ => 0.3)))
    P3 = update_coefficients(P1, (; lead=(; T=0.2, μ=0.3)))
    properties = [:total_master_matrix, :total_rate_matrix, :total_current_operator]
    @test all(getproperty(P2, s) == getproperty(P3, s) for s in properties)
    @test !any(getproperty(P1, s) == getproperty(P3, s) for s in properties)

    update_coefficients!(P3, (; lead=(; T=0.3, μ=0.4)))
    P4 = update_coefficients(P1, (; lead=(; T=0.3, μ=0.4)))
    @test all(getproperty(P3, s) == getproperty(P4, s) for s in properties)
    @test !any(getproperty(P1, s) == getproperty(P4, s) for s in properties)

    @test P3.dissipators[:lead].lead.T == 0.3
    @test P4.dissipators[:lead].lead.T == 0.3
end
