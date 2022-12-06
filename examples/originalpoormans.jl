module ExamplePMM
using QuantumDots
using LinearAlgebra
using Plots

function pmm_hamiltonian(particle_ops, delta, tun, eps1, eps2)
    d = particle_ops
    ham_dots = eps1*d[1]'*d[1] + eps2*d[2]'*d[2]
    ham_tun = tun*d[1]'*d[2] + tun*d[2]'*d[1]
    ham_sc = delta*d[1]'*d[2]' + delta*d[2]*d[1]
    ham = ham_dots + ham_tun + ham_sc
    return ham
end

function pmm_hamiltonian(particle_ops, delta, tun, eps)
    pmm_hamiltonian(particle_ops, delta, tun, eps, eps)
end

function groundindices(basis, vecs, energies)
    parityop = basis*QuantumDots.ParityOperator()*basis
    parities = [QuantumDots.measure(parityop, vec) for vec in vecs]
    evenindices = findall(parity -> parity ≈ 1, parities)
    oddindices = setdiff(1:length(energies), evenindices)
    return evenindices[1], oddindices[1]
end

function majoranapolarization(basis, majoranas, oddstate, evenstate)
    majoranaswbasis = map(maj -> basis*maj*basis, majoranas)
    w, z = map(majwbasis -> QuantumDots.dot(oddstate, majwbasis, evenstate), majoranaswbasis)
    return (w^2 - z^2)/(w^2 + z^2)
end

function plot_gapandmp()
    N = 2
    delta = 1
    tun = delta
    basis = FermionBasis(N)
    points = 100
    eps_vec = range(-delta, delta, points)
    gaps = zeros(points)
    mps = zeros(points)
    d = particles(basis)
    maj_plus = d[1] + d[1]'
    maj_minus = d[1] - d[1]'
    majoranas = (maj_plus, maj_minus)
    for i = 1:points
        ham = pmm_hamiltonian(d, delta, tun, eps_vec[i])
        hamwithbasis = basis*ham*basis
        mat = Matrix(hamwithbasis)
        energies, vecs = eigen(mat)
        even, odd = groundindices(basis, eachcol(vecs), energies)
        gaps[i] = abs(energies[even] - energies[odd])
        mps[i] = majoranapolarization(basis, majoranas, vecs[:,odd], vecs[:,even])
    end
    display(plot(eps_vec, [gaps, mps], label=["Gap" "MP"]))
    xlabel!("Dot energies")
end
end
