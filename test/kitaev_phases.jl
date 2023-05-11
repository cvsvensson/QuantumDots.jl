labels = Tuple(1:3)
b = QuantumDots.FermionBdGBasis(labels)
b2 = QuantumDots.FermionBasis(labels)


t1 = Δ1 = 1
t2 = Δ2 = 1
p1 = 1#exp(1im*0)
p2 = 1#exp(0*1im*(π/1.5))
μ = 0 .* [1,-3,5]
# p1 = 1/p2
poor_mans_ham = Matrix(QuantumDots._kitaev_hamiltonian(b; μ , t=[t1,t2],Δ=[Δ1*p1,Δ2*p2],V=[0,0],bias=[0,0,0]))
poor_mans_ham2 = Matrix(QuantumDots._kitaev_hamiltonian(b2; μ,t=[t1,t2],Δ=[Δ1*p1,Δ2*p2],V=[0,0],bias=[0,0,0]))
es, ops = QuantumDots.enforce_ph_symmetry(eigen(poor_mans_ham; sortby = e->(sign(e),abs(e))))
es2, states = eigen(poor_mans_ham2)

@test norm(sort(es,by=abs)[1:2]) < 1e-12
qps = map(op -> QuantumDots.QuasiParticle(op,b), eachcol(ops));
foreach(QuantumDots.visualize, majs[3:4])
QuantumDots.majorana_density.(majs[3:4])
QuantumDots.majorana_polarization(majs[3:4]...)

ρ1 = QuantumDots.one_particle_density_matrix(ops)
# ρ12 = QuantumDots.one_particle_density_matrix2(ops)
ρ12 = QuantumDots.one_particle_density_matrix(qps[1:3])
ρ13 = QuantumDots.one_particle_density_matrix(qps[[1,2,4]])
ρ2 = QuantumDots.one_particle_density_matrix(states[:,1]*states[:,1]',b2)
ρ22 = QuantumDots.one_particle_density_matrix(states[:,2]*states[:,2]',b2)

@test norm(ρ12 - ρ2) < 1e-12

function mp(θ)
    t1 = Δ1 = 10
    t2 = Δ2 = 1
    p1 = exp(1im*0)
    p2 = exp(1im*(θ/2))
    p1 = 1/p2
    poor_mans_ham = Matrix(QuantumDots._kitaev_hamiltonian(b; μ= [0,0,0],t=[t1,t2],Δ=[Δ1*p1,Δ2*p2],V=[0,0],bias=[0,0,0]))
    es, ops = eigen(poor_mans_ham^2)
    @test norm(es[1:2]) < 1e-12
    qps = map(op -> QuantumDots.QuasiParticle(op,b), eachcol(ops));
    QuantumDots.majorana_polarization(majs[1:2]...), es
end

θs = range(0,2pi; length = 100)
mps = [first(mp(θ)) for θ in θs]
ess = [last(mp(θ)) for θ in θs]

##
QuantumDots.lineplot(θs ./ π,abs.(mps),xlabel = "θ/π",ylabel = "MP", width = 150) QuantumDots.lineplot(θs ./ π,map(es->es[3] - es[2],ess),xlabel = "θ/π",ylabel = "Exc", width = 150)
QuantumDots.lineplot(θs ./ π,map(es->es[4] - es[2],ess),xlabel = "θ/π",ylabel = "Exc", width = 150)