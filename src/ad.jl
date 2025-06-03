
chem_derivative(backend, d::LindbladDissipator, args...) = chem_derivative(backend, d -> Matrix(d), d, args...)
chem_derivative(backend, d::PauliDissipator, args...) = chem_derivative(backend, d -> Matrix(d), d, args...)

function chem_derivative(backend, f::Function, d)
    func = μ -> f(__update_coefficients(d, (; μ), nothing))
    AD.derivative(backend, func, d.lead.μ)[1]
end

function chem_derivative(backend, f::Function, d, _p)
    p = _dissipator_params(d, _p)
    func = μ -> f(__update_coefficients(d, (; μ, T=p.T, rate=p.rate), nothing))
    AD.derivative(backend, func, p.μ)[1]
end

function conductance_matrix(backend, ls::AbstractOpenSystem, rho, current_op)
    linsolve = init(StationaryStateProblem(ls))
    key_iter = collect(keys(ls.dissipators))
    N = length(key_iter)
    T = real(eltype(ls))
    # G = AxisKeys.KeyedArray(Matrix{T}(undef, N, N), (key_iter, key_iter))
    G = AxisKeys.wrapdims(AxisKeys.KeyedArray(Matrix{T}(undef, N, N), (key_iter, key_iter)), :∂Iᵢ, :∂μⱼ)

    for k in key_iter
        d = ls.dissipators[k]
        dD = chem_derivative(backend, d)
        sol = solveDiffProblem!(linsolve, rho, dD)
        rhodiff_currents = measure(sol, current_op, ls)
        dissdiff_current = dot(current_op, tomatrix(dD * rho, ls))
        rhodiff_currents[AxisKeys.Key(k)] += dissdiff_current
        G(:, k) .= real.(rhodiff_currents)
    end
    G
end


function conductance_matrix(dμ::Number, ls::AbstractOpenSystem, current_op)
    perturbations = [Dict(k => (; μ=d.lead.μ + dμ)) for (k, d) in pairs(ls.dissipators)]
    function get_current(pert)
        newls = __update_coefficients(ls, pert)
        sol = solve(StationaryStateProblem(newls))
        measure(sol, current_op, newls)
    end
    I0 = get_current(SciMLBase.NullParameters())
    N = length(I0)
    T = real(eltype(I0))
    ks = AxisKeys.axiskeys(I0, 1)
    G = AxisKeys.wrapdims(AxisKeys.KeyedArray(Matrix{T}(undef, N, N), (ks, ks)), :∂Iᵢ, :∂μⱼ)
    for dict in perturbations
        diff_currents = (get_current(dict) .- I0) ./ dμ
        key = first(only(dict))
        G(:, key) .= real.(diff_currents)
    end
    return G

end


function conductance_matrix(ad::AD.FiniteDifferencesBackend, ls::AbstractOpenSystem, current_op)
    keys_iter = collect(keys(ls.dissipators))
    μs0 = [ls.dissipators[k].lead.μ for k in keys_iter]
    function get_current(μs)
        pert = Dict(k => (; μ=μs[n]) for (n, k) in enumerate(keys_iter))
        newls = __update_coefficients(ls, pert)
        sol = solve(StationaryStateProblem(newls))
        currents = measure(sol, current_op, newls)
        [real(currents(k)) for k in keys_iter]
        #real()
    end
    J = AD.jacobian(ad, get_current, μs0)[1]
    AxisKeys.wrapdims(AxisKeys.KeyedArray(J, (keys_iter, keys_iter)), :∂Iᵢ, :∂μⱼ)
end