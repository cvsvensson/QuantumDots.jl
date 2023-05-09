labels = Tuple(1:3)
b = QuantumDots.FermionBdGBasis(labels)

t1 = Δ1 = 1
t2 = Δ2 = 2
p1 = exp(1im*0)
p2 = exp(0*1im*(π/1.5))
# p1 = 1/p2
poor_mans_ham = Matrix(QuantumDots._kitaev_hamiltonian(b; μ= [0,0,0],t=[t1,t2],Δ=[Δ1*p1,Δ2*p2],V=[0,0],bias=[0,0,0]))
es, ops = eigen(poor_mans_ham^2)
@test norm(es[1:2]) < 1e-12
qps = map(op -> QuantumDots.QuasiParticle(op,b), eachcol(ops));
majs = QuantumDots.MajoranaQuasiParticle.(qps);
foreach(QuantumDots.visualize, majs[1:2])
QuantumDots.majorana_density.(majs[1:2])
QuantumDots.majorana_polarization(majs[1:2]...)


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
    majs = QuantumDots.MajoranaQuasiParticle.(qps);
    QuantumDots.majorana_polarization(majs[1:2]...), es
end

θs = range(0,2pi; length = 100)
mps = [first(mp(θ)) for θ in θs]
ess = [last(mp(θ)) for θ in θs]

##
QuantumDots.lineplot(θs ./ π,abs.(mps),xlabel = "θ/π",ylabel = "MP", width = 150) QuantumDots.lineplot(θs ./ π,map(es->es[3] - es[2],ess),xlabel = "θ/π",ylabel = "Exc", width = 150)
QuantumDots.lineplot(θs ./ π,map(es->es[4] - es[2],ess),xlabel = "θ/π",ylabel = "Exc", width = 150)