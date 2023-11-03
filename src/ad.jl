import AbstractDifferentiation as AD

chem_derivative(args...) = chem_derivative(backend, d -> Matrix(d), args...)
function chem_derivative(backend, f::Function, d)
    func = μ -> f(update(d, (; μ), nothing))
    AD.derivative(backend, func, d.lead.μ)
end

function chem_derivative(backend, f::Function, d, _p)
    p = _dissipator_params(d, _p)
    func = μ -> f(update(d, (; μ, T=p.T, rate=p.rate), nothing))
    AD.derivative(backend, func, p.μ)
end

function conductance_matrix(backend, rho, current_op, ls::AbstractOpenSystem)
    dDs = [chem_derivative(backend, d) for d in ls.dissipators]
    linsolve = init(StationaryStateProblem(ls))
    rhodiff = stack([collect(measure(QuantumDots.solveDiffProblem!(linsolve, rho, dD), current_op, ls)) for dD in dDs])
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