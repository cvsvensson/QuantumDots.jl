struct Pauli <: AbstractOpenSolver end

density_of_states(lead::NormalLead) = 1 #FIXME: put in some physics here
pauli_system(H::OpenSystem) = pauli_system(diagonalize(H))
function pauli_system(H::OpenSystem{<:DiagonalizedHamiltonian})
    ds = map(l->PauliDissipator(eigenvaluevector(H),l), H.leads)
    PauliSystem(ds)
end

struct PauliSystem{A,W,I,D} <: AbstractOpenSystem
    total_master_matrix::A
    total_rate_matrix::W
    total_current_operator::I
    dissipators::D
end

struct PauliDissipator{L,W,I,D,T,E}
    lead::L
    Win::W
    Wout::W
    Iin::I
    Iout::I
    total_master_matrix::D
    props::LArray{T,1, Vector{T}, (:T, :μ)}
    energies::E
end
function PauliDissipator(energies::E,lead::L) where {L,E}
    props = _default_pauli_dissipator_params(lead)
    Win, Wout = get_rates(energies, lead)
    D = zero(Win)
    Iin = zeros(eltype(Win), size(Win,1))
    Iout = zero(Iin)
    update_currents!(Iin,Iout,Win,Wout)
    update_master_matrix!(D,Win,Wout,Iin,Iout)
    PauliDissipator{L,typeof(Win),typeof(Iin),typeof(D),eltype(props),E}(lead, Win, Wout, Iin, Iout, D, props, energies)
end
function _default_pauli_dissipator_params(l::NormalLead)
    T, μ = promote(temperature(l), chemical_potential(l))
    type = eltype(T)
    props = @LVector type (:T, :μ)
    props.T = T
    props.μ = μ
    return props
end

internal_rep(u::UniformScaling, sys::PauliSystem) = u[1,1]*ones(size(sys.total_master_matrix, 2))
internal_rep(u::AbstractMatrix, ::PauliSystem) = diag(u)
internal_rep(u::AbstractVector, ::PauliSystem) = u
tomatrix(u::AbstractVector, ::PauliSystem) = tomatrix(u, Pauli())
tomatrix(u::AbstractVector, ::Pauli) = Diagonal(u)
LinearOperator(L::PauliSystem{<:AbstractMatrix}; normalizer = false) = MatrixOperator(L; normalizer)
# LinearOperator(system::PauliSystem; kwargs...) = LinearOperator(system.total_master_matrix; kwargs...)
# LinearOperatorWithNormalizer(system::PauliSystem; kwargs...) = LinearOperator(add_normalizer(system.total_master_matrix); kwargs...)

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
    P = PauliSystem(D,(;in = Win,out = Wout),(;in = Iin, out = Iout),ds)
    update_total_operators!(P)
    return P
end

# LinearOperator(P::PauliSystem{<:AbstractMatrix}; normalizer = false) = MatrixOperator(P; normalizer)
function MatrixOperator(_P::PauliSystem; normalizer)
    P = deepcopy(_P)
    update_func! = pauli_updater!(P; normalizer)
    A = normalizer ? add_normalizer(P.total_master_matrix) : P.total_master_matrix
    MatrixOperator(A; update_func!)
end
function zero_total_operators!(P::PauliSystem)
    foreach(x->fill!(x, zero(eltype(x))), (P.total_rate_matrix.in, 
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

function pauli_updater!(L; normalizer)
    normalizer || return 
    function update_func!(A, u, p, t)
        updated = false
        for (label, props) in pairs(p)
            update_dissipator!(L, label, props)
            updated = true
        end
        if updated
            update_total_operator!(L)
            A .= L.total_master_matrix
        end
        return nothing
    end

    return function update_func_normalizer!(A, u, p, t)
        updated = false
        for (label, props) in pairs(p)
            update_dissipator!(L, label, props)
            updated = true
        end
        if updated
            update_total_operator!(L)
            A[1:end-1,:] .= L.total_master_matrix
        end
        return nothing
    end
end

function PauliDissipator(system::AbstractOpenSystem, lead::NormalLead)
    PauliDissipator(eigenvaluevector(system), lead)
    # return RateEquation(W, A, I, lead.label, system)
end
function get_rates(E::AbstractVector, lead::NormalLead)
    dos = density_of_states(lead)
    Win = zero(lead.jump_in)
    Wout = zero(lead.jump_out)
    update_rates!(Win, lead.jump_in, lead.T, lead.μ, E; dos)
    update_rates!(Wout, lead.jump_out, lead.T, -lead.μ, E; dos)
    return Win, Wout
end

function update_rates!(W, op, T, μ, E::AbstractVector; dos)
    for I in CartesianIndices(W)
        n1, n2 = Tuple(I)
        δE = E[n1] - E[n2]
        W[n1, n2] = 2π * dos * abs2(op[n1, n2]) * fermidirac(δE, T, μ)
        # Wout[n1, n2] = 2π * dos * abs2(Tout[n1, n2]) * (1 - fermidirac(-δE, T, μ))#*QuantumDots.fermidirac(E[n1]-E[n2],T,-μ)
    end
    return W
end

update_master_matrix!(d::PauliDissipator) = update_master_matrix!(d.total_master_matrix, d.Win, d.Wout, d.Iin, d.Iout)

function update_master_matrix!(D, Win, Wout, Iin, Iout)
    D .= Win .+ Wout
    # Diagonal(Iin - Iout)
    # D .-= Diagonal(vec(sum(D, dims=1)))
    for (i,di) in enumerate(diagind(D))
        D[di] -= Iin[i] - Iout[i]
    end
    return nothing
end
update_currents!(d::PauliDissipator) = update_currents!(d.Iin,d.Iout,d.Win,d.Wout)
function update_currents!(Iin,Iout,Win,Wout)
    for j in eachindex(Iin)
        Iin[j] = sum(@view Win[:,j])
        Iout[j] = -sum(@view Wout[:,j])
    end
    return nothing
end

function update_dissipator(d::PauliDissipator, p = ())
    if haskey(p, :μ) || haskey(p, :T)
        μ = get(p, :μ, d.props.μ)
        T = get(p, :T, d.props.T)
        d.props.μ = μ
        d.props.T = T
        update_rate_matrix!(d.Win, d.lead.jump_in, d.energies, T,μ)
        update_rate_matrix!(d.Wout, d.lead.jump_out, d.energies, T,-μ)
        update_currents!(d)
        update_master_matrix!(d)
    end 
    return nothing
end

function add_normalizer(m::AbstractMatrix{T}) where {T}
    [m; fill(one(T), size(m, 2))']
end

get_currents(eq::PauliSystem, alg=nothing; kwargs...) = get_currents(solve(StationaryStateProblem(eq), alg), eq; kwargs...)
get_currents(rho, eq::PauliSystem) = get_currents(internal_rep(rho, eq), eq)
function get_currents(rho::AbstractVector, P::PauliSystem) #rho is the diagonal density matrix
    map(d-> (; in=dot(d.Iin, rho), out=dot(d.Iout, rho)), P.dissipators)
end
