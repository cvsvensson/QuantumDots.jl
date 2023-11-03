import AbstractDifferentiation as AD

chem_derivative(backend, args...) = chem_derivative(backend, d -> Matrix(d), args...)
function chem_derivative(backend, f::Function, d)
    func = μ -> f(update(d, (; μ), nothing))
    AD.derivative(backend, func, d.lead.μ)[1]
end

function chem_derivative(backend, f::Function, d, _p)
    p = _dissipator_params(d, _p)
    func = μ -> f(update(d, (; μ, T=p.T, rate=p.rate), nothing))
    AD.derivative(backend, func, p.μ)[1]
end

function conductance_matrix(backend, rho, current_op, ls::AbstractOpenSystem)
    dDs = [chem_derivative(backend, d) for d in ls.dissipators]
    linsolve = init(StationaryStateProblem(ls))
    sols = [QuantumDots.solveDiffProblem!(linsolve, rho, dD) for dD in dDs]
    rhodiff = stack([collect(measure(sol, current_op, ls)) for sol in sols])
    dissdiff = Diagonal([dot(current_op, tomatrix(dD * rho, ls)) for dD in dDs])
    return dissdiff + rhodiff
end

function conductance_matrix(backend, rho, sys::PauliSystem)
    dDs = [chem_derivative(backend, d -> [Matrix(d), d.Iin + d.Iout], d) for d in sys.dissipators]
    linsolve = init(StationaryStateProblem(sys))
    rhodiff = stack([collect(get_currents(solveDiffProblem!(linsolve, rho, dD[1]), sys)) for dD in dDs])
    dissdiff = Diagonal([dot(dD[2], rho) for dD in dDs])
    return dissdiff + rhodiff
end


function conductance_matrix(dμ::Number, current_op, ls::AbstractOpenSystem)
    perturbations = map(d -> (; μ=d.lead.μ + dμ), ls.dissipators)
    function get_current(pert)
        newls = update(ls, pert)
        sol = solve(StationaryStateProblem(newls))
        collect(measure(sol, current_op, newls))
    end
    I0 = get_current(SciMLBase.NullParameters())
    stack(map(key -> (get_current(perturbations[[key]]) .- I0) / dμ, keys(perturbations)))
end


function conductance_matrix(ad::AD.FiniteDifferencesBackend, current_op, ls::AbstractOpenSystem)
    # perturbations = map(d -> (; μ=d.lead.μ + dμ), ls.dissipators)
    μs0 = [d.lead.μ for d in ls.dissipators]
    function get_current(μs)
        count = 0
        pert = map(d -> (; μ=μs[count+=1]), ls.dissipators)
        newls = update(ls, pert)
        sol = solve(StationaryStateProblem(newls))
        real(collect(measure(sol, current_op, newls)))
    end
    AD.jacobian(ad, get_current, μs0)[1]
end