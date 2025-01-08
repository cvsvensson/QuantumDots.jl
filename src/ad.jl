
chem_derivative(backend, d::LindbladDissipator, args...) = chem_derivative(backend, d -> Matrix(d), d, args...)
chem_derivative(backend, d::PauliDissipator, args...) = chem_derivative(backend, d -> Matrix(d), d, args...)

function chem_derivative(backend, f::Function, d)
    func = μ -> f(update_coefficients(d, (; μ), nothing))
    AD.derivative(backend, func, d.lead.μ)[1]
end

function chem_derivative(backend, f::Function, d, _p)
    p = _dissipator_params(d, _p)
    func = μ -> f(update_coefficients(d, (; μ, T=p.T, rate=p.rate), nothing))
    AD.derivative(backend, func, p.μ)[1]
end

function conductance_matrix(backend, ls::AbstractOpenSystem, rho, current_op)
    linsolve = init(StationaryStateProblem(ls))
    key_iter = keys(ls.dissipators)
    mapreduce(hcat, key_iter) do k
        d = ls.dissipators[k]
        dD = chem_derivative(backend, d)
        sol = solveDiffProblem!(linsolve, rho, dD)
        rhodiff_currents = measure(sol, current_op, ls)
        dissdiff_current = dot(current_op, tomatrix(dD * rho, ls))
        [real(rhodiff_currents[k2] + dissdiff_current * (k2 == k)) for k2 in key_iter]
    end
end


function conductance_matrix(dμ::Number, ls::AbstractOpenSystem, current_op)
    perturbations = [Dict(k => (; μ=d.lead.μ + dμ)) for (k, d) in pairs(ls.dissipators)]
    function get_current(pert)
        newls = update_coefficients(ls, pert)
        sol = solve(StationaryStateProblem(newls))
        real(collect(values(measure(sol, current_op, newls))))
    end
    I0 = get_current(SciMLBase.NullParameters())
    stack(map(pert -> (get_current(pert) .- I0) / dμ, perturbations))
end


function conductance_matrix(ad::AD.FiniteDifferencesBackend, ls::AbstractOpenSystem, current_op)
    keys_iter = keys(ls.dissipators)
    μs0 = [ls.dissipators[k].lead.μ for k in keys_iter]
    function get_current(μs)
        pert = Dict(k => (; μ=μs[n]) for (n, k) in enumerate(keys_iter))
        newls = update_coefficients(ls, pert)
        sol = solve(StationaryStateProblem(newls))
        currents = measure(sol, current_op, newls)
        [real(currents[k]) for k in keys_iter]
    end
    AD.jacobian(ad, get_current, μs0)[1]
end