module ExamplePMM
using QuantumDots
using LinearAlgebra
using Plots

function pmm_hamiltonian(particle_ops, delta, tun, eps1, eps2 = eps1)
    d = particle_ops
    ham_dots = eps1*d[1]'*d[1] + eps2*d[2]'*d[2]
    ham_tun = tun*d[1]'*d[2] + tun*d[2]'*d[1]
    ham_sc = delta*d[1]'*d[2]' + delta*d[2]*d[1]
    ham = ham_dots + ham_tun + ham_sc
    return ham
end

function groundindices(basis, vecs, energies)
    parityop = parityoperator(basis)
    parities = [v'parityop*v for v in vecs]
    evenindices = findall(parity -> parity ≈ 1, parities)
    oddindices = setdiff(1:length(energies), evenindices)
    return evenindices[1]::Int, oddindices[1]::Int
end

function majoranapolarization(majoranas, oddstate, evenstate)
    w, z = map(majorana -> oddstate'*majorana*evenstate, majoranas)
    return (w^2 - z^2)/(w^2 + z^2)
end

function plot_gapandmp()
    N = 2
    delta = 1
    tun = delta
    d = FermionBasis(1:N)
    points = 100
    eps_vec = range(-delta, delta, points)
    gaps = zeros(Float64, points)
    mps = zeros(Float64, points)
    maj_plus = d[1] + d[1]'
    maj_minus = d[1] - d[1]'
    majoranas = (maj_plus, maj_minus)
    for i = 1:points
        ham = pmm_hamiltonian(d, delta, tun, eps_vec[i])
        energies, vecs = eigen!(Matrix(ham))
        even, odd = groundindices(d, eachcol(vecs), energies)
        gaps[i] = abs(energies[even] - energies[odd])
        mps[i] = majoranapolarization(majoranas, vecs[:,odd], vecs[:,even])
    end
    display(plot(eps_vec, [gaps, mps], label=["Gap" "MP"]))
    xlabel!("Dot energies")
end
end
