struct Pauli <: AbstractOpenSolver end

density_of_states(lead::NormalLead) = 1 #FIXME: put in some physics here
(::Pauli)(H::OpenSystem) = Pauli()(diagonalize(H))
function (::Pauli)(H::OpenSystem{<:DiagonalizedHamiltonian})
    ds = map(l -> PauliDissipator(eigenvalues(H), l), H.leads)
    PauliSystem(ds)
end

struct PauliSystem{A,W,I,D} <: AbstractOpenSystem
    total_master_matrix::A
    total_rate_matrix::W
    total_current_operator::I
    dissipators::D
end
Base.Matrix(P::PauliSystem) = P.total_master_matrix

struct PauliDissipator{L,W,I,D,E} <: AbstractDissipator
    lead::L
    Win::W
    Wout::W
    Iin::I
    Iout::I
    total_master_matrix::D
    energies::E
end
Base.Matrix(d::PauliDissipator) = d.total_master_matrix

function PauliDissipator(energies::E, lead::L) where {L,E}
    Win, Wout = get_rates(energies, lead)
    D = Win + Wout
    Iin = vec(sum(Win, dims=1))
    Iout = -vec(sum(Wout, dims=1))
    D .-= Diagonal(Iin) .- Diagonal(Iout)
    PauliDissipator{L,typeof(Win),typeof(Iin),typeof(D),E}(lead, Win, Wout, Iin, Iout, D, energies)
end

internal_rep(u::UniformScaling, sys::PauliSystem) = u[1, 1] * ones(size(sys.total_master_matrix, 2))
internal_rep(u::AbstractMatrix, ::PauliSystem) = diag(u)
internal_rep(u::AbstractVector, ::PauliSystem) = u
tomatrix(u::AbstractVector, ::PauliSystem) = tomatrix(u, Pauli())
tomatrix(u::AbstractVector, ::Pauli) = Diagonal(u)
LinearOperator(L::PauliSystem{<:AbstractMatrix}, args...; normalizer=false) = MatrixOperator(L; normalizer)

function identity_density_matrix(system::PauliSystem)
    A = system.total_master_matrix
    fill(one(eltype(A)), size(A, 2))
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
update(L::PauliSystem, p, tmp=nothing) = update_pauli_system(L, p)
function update_pauli_system(L::PauliSystem, ::SciMLBase.NullParameters)
    L
end
function update_pauli_system(L::PauliSystem, p)
    _newdissipators = map(lp -> first(lp) => update(L.dissipators[first(lp)], last(lp)), collect(pairs(p)))
    newdissipators = merge(L.dissipators, _newdissipators)
    PauliSystem(newdissipators)
end
function update(d::PauliDissipator, p, tmp=nothing)
    PauliDissipator(d.energies, update_lead(d.lead, p))
end

function MatrixOperator(P::PauliSystem; normalizer)
    A = normalizer ? add_normalizer(P.total_master_matrix) : P.total_master_matrix
    MatrixOperator(A)
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



function conductance_matrix(sys::PauliSystem, args...)
    rho = solve(StationaryStateProblem(sys))
    conductance_matrix(rho, sys::PauliSystem, args...)
end

function conductance_matrix(rho, sys::PauliSystem, dμ)
    perturbations = map(d -> (; μ=d.lead.μ + dμ), sys.dissipators)
    function get_current(pert)
        newsys = update(sys, pert)
        sol = solve(StationaryStateProblem(newsys))
        collect(get_currents(sol, newsys))
    end
    I0 = get_current(SciMLBase.NullParameters())
    stack(map(key -> (get_current(perturbations[[key]]) .- I0) / dμ, keys(perturbations)))
end