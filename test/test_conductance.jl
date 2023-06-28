n = 16
N = 4
c = FermionBasis(ntuple(identity,N))

ham(c,μ,t,Δ) = μ * numberoperator(c) + sum(t*c[i+1]'*c[i] + Δ*c[i+1]c[i] + QuantumDots.hc for i in 1:N-1)
H = ham(c,0,1,1)
ms = [numberoperator(c) ]
leads = [QuantumDots.NormalLead(.1,0.0; in = c[1]'), QuantumDots.NormalLead(.1,0.0; in = c[N]')]
sys = QuantumDots.OpenSystem(H,leads)
dsys = QuantumDots.diagonalize(sys)
Hd = QuantumDots.diagonalize(H)

function currents(V)
    newleads = [QuantumDots.NormalLead(leads[1]; μ = V), leads[2]]
    sys = QuantumDots.OpenSystem(H,newleads)
    l,m = QuantumDots.prepare_lindblad(sys,ms)
    sol = QuantumDots.stationary_state(l)
    real(dot(vec(m[1]),l.dissipators[1].in*vec(sol)) - 
    dot(vec(m[1]),l.dissipators[1].out*vec(sol)))
    # real(QuantumDots.measure(sol,m1[1],l1)[1].in)
end

function get_lsys(V)
    newleads = [QuantumDots.NormalLead(leads[1]; μ = V), leads[2]]
    sys = QuantumDots.OpenSystem(H,newleads)
    QuantumDots.prepare_lindblad(sys,ms)
end
function get_lop(V) 
    l = get_lsys(V)[1]
    QuantumDots.lindblad_with_normalizer(l.lindblad, l.vectorizer)
end

function conductance2(V)
    l,m = get_lsys(V)
    A = QuantumDots.lindblad_with_normalizer(l.lindblad, l.vectorizer)
    dAdV = ForwardDiff.derivative(get_lop,V)
    n = size(l.lindblad,2)
    x = zeros(eltype(A),n)
    push!(x,one(eltype(A)))
    u0 = complex(l.vectorizer.idvec ./ sqrt(n))
    prob = LinearProblem(A, x; u0)
    linsolve = init(prob)
    x0 = solve!(linsolve)
    linsolve.b = -dAdV*x0
    sol2 = solve!(linsolve)
    sol = reshape(sol2,l.vectorizer)
    QuantumDots.measure(m[1], sol, l)
end
function conductance3(V;dV = .01)
    l1,m1 = get_lsys(V)
    sol1 = QuantumDots.stationary_state(l1)
    l2,m2 = get_lsys(V+dV)
    sol2 = QuantumDots.stationary_state(l2)
    QuantumDots.measure(m1[1], (sol2 - sol1)/dV, l1)
end