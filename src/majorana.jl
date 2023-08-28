function majorana_coefficients(ψ1, ψ2, c::FermionBasis)
    a = map(f -> ψ1' * f * ψ2, c.dict)
    b = map(f -> ψ1' * f' * ψ2, c.dict)
    w = a .+ b
    z = 1im .* (a .- b)
    return w, z
end

function majorana_polarization(w, z, region)
    total_md = sum(w[l]^2 + z[l]^2 for l in region)
    N = abs(total_md)
    D = sum(abs2(w[l]) + abs2(z[l]) for l in region)
    return (; mp=N / D, mpu=N)
end
