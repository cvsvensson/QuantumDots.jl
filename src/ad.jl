
chem_derivative(backend, d::LindbladDissipator, args...) = chem_derivative(backend, d -> Matrix(d), d, args...)
chem_derivative(backend, d::PauliDissipator, args...) = chem_derivative(backend, d -> Matrix(d), d, args...)

function chem_derivative(backend, f::Function, d)
    func = μ -> f(update(d, (; μ), nothing))
    AD.derivative(backend, func, d.lead.μ)[1]
end

function chem_derivative(backend, f::Function, d, _p)
    p = _dissipator_params(d, _p)
    func = μ -> f(update(d, (; μ, T=p.T, rate=p.rate), nothing))
    AD.derivative(backend, func, p.μ)[1]
end

function conductance_matrix(backend, ls::AbstractOpenSystem, rho, current_op)
    dDs = [chem_derivative(backend, d) for d in ls.dissipators]
    linsolve = init(StationaryStateProblem(ls))
    rhodiff = stack([collect(measure(QuantumDots.solveDiffProblem!(linsolve, rho, dD), current_op, ls)) for dD in dDs])
    dissdiff = Diagonal([dot(current_op, tomatrix(dD * rho, ls)) for dD in dDs])
    return dissdiff + rhodiff
end


function conductance_matrix(dμ::Number, ls::AbstractOpenSystem, current_op)
    perturbations = map(d -> (; μ=d.lead.μ + dμ), ls.dissipators)
    function get_current(pert)
        newls = update(ls, pert)
        sol = solve(StationaryStateProblem(newls))
        collect(measure(sol, current_op, newls))
    end
    I0 = get_current(SciMLBase.NullParameters())
    stack(map(key -> (get_current(perturbations[[key]]) .- I0) / dμ, keys(perturbations)))
end


function conductance_matrix(ad::AD.FiniteDifferencesBackend, ls::AbstractOpenSystem, current_op)
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