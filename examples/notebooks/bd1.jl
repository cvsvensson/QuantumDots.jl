### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ a9a6cdf0-76db-11ed-0076-6dfc37097407
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate("")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using QuantumDots, LinearAlgebra, Random, BlackBoxOptim, PlutoUI, Plots, Printf, KrylovKit, SparseArrays, Folds, FiniteDifferences, LinearMaps, ProfileCanvas 
	using .Threads: nthreads, threadid
	BLAS.set_num_threads(1)
end

# ╔═╡ bc605fb9-16f9-44a4-bb58-c471e42e9754
Pkg.status()

# ╔═╡ 33f235a6-0220-489f-97c8-40d8493048ca
Threads.nthreads()

# ╔═╡ 127abb1f-7850-402f-a3a9-d4a24ba20246
gapratio(es) = real(diff(es)[1][1]/(diff(sort(vcat(es...)))[2]))

# ╔═╡ 1fb9255b-da89-4358-b1e3-91adb691b377
let
	Random.seed!(1234)
	BLAS.set_num_threads(1)
	cost_function(es,mpu::Number) = 10^3*gapratio(es)^2 + (1-abs(mpu))^2
end

# ╔═╡ 12eaa1c6-1c66-4ebc-9577-31569a706fa0
function MPu(coeffs)
    n = div(size(coeffs, 1), 2)
    cs = abs.(coeffs[1:n, :,:] .^ 2)
	c2 = sum(cs,dims=2)[:,1,:]
    sum(c2[:, 1] .- c2[:, 2]) / sum(c2[:, 1] .+ c2[:, 2])
end

# ╔═╡ 9ad954b1-a9c1-4c0c-a6a8-44d62d085f71
cost_function(gapratio,mpu::Number) = 10^3*(gapratio)^2 + (1-abs(mpu))^2

# ╔═╡ 3fb9c1b3-8fd8-40e8-a269-3fa547e35dbd
begin
	hopping(c,id1,id2) = c[id1]'c[id2] + c[id2]'c[id1]
	sc(c,id1,id2) = c[id1]'c[id2]' + c[id2]c[id1]
	numberop(c,id) = c[id]'c[id]
	coulomb(c,id1,id2) = numberop(c,id1)*numberop(c,id2)
	
	function _BD1_ham_2site(basis,j;t, α, Δk, Δ1, V)
	    c = particles(basis)
	    -t/2*(hopping(c,(j,:↑),(j+1,:↑)) + hopping(c,(j,:↓),(j+1,:↓))) +
	    -α/2 *(hopping(c,(j,:↑),(j+1,:↓)) - hopping(c,(j,:↓),(j+1,:↑))) +
	    V* (numberop(c,(j,:↑))+numberop(c,(j,:↓)))*(numberop(c,(j+1,:↑))+numberop(c,(j+1,:↓))) +
	    Δ1*(sc(c,(j,:↑),(j+1,:↓)) + sc(c,(j,:↓),(j+1,:↑))) +
	    Δk*(sc(c,(j,:↓),(j+1,:↓)) - sc(c,(j,:↑),(j+1,:↑)))
	end
	function _BD1_ham_1site(basis,j;μ,h,Δ,U)
	    c = particles(basis)
	    (-μ - h)*numberop(c,(j,:↑)) + (-μ + h)*numberop(c,(j,:↓)) +
	    Δ*sc(c,(j,:↑),(j,:↓)) + U*numberop(c,(j,:↑))*numberop(c,(j,:↓))
	end
	function BD1_hamqd(basis; μ, h, Δ1, t, α, Δ, U, V, θ, bias)
	    N = div(length(particles(basis)),2)
	    #dbias =  bias*((1:N) .- ceil(N/2))/N
	    dbias = bias*2((1:N) .- ((N+1)/2))/N
		αnew = cos(θ/2)*α + sin(θ/2)*t
	    tnew = cos(θ/2)*t - sin(θ/2)*α
	    Δk = Δ1*sin(θ/2)
	    Δ1 = Δ1*cos(θ/2)
	    t = tnew
	    α = αnew
	    H = QuantumDots.FockOperatorSum(Float64[],[],basis,basis)
	    for j in 1:(N-1)
	        H += _BD1_ham_2site(basis,j;t,α,Δ1,Δk,V)
	    end
	    for j in 1:N
	        H += _BD1_ham_1site(basis,j;μ = μ+dbias[j],h,Δ,U)
	    end
	    return H
	end
	function BD1_hamqd_dis(basis; μs, h, Δ1, t, α, Δ, U, V, θ, bias)
	    N = div(length(particles(basis)),2)
	    dbias = bias*2((1:N) .- ((N+1)/2))/N
	    αnew = cos(θ/2)*α + sin(θ/2)*t
	    tnew = cos(θ/2)*t - sin(θ/2)*α
	    Δk = Δ1*sin(θ/2)
	    Δ1 = Δ1*cos(θ/2)
	    t = tnew
	    α = αnew
	    H = QuantumDots.FockOperatorSum(Float64[],[],basis,basis)
	    for j in 1:(N-1)
	        H += _BD1_ham_2site(basis,j;t,α,Δ1,Δk,V)
	    end
	    for j in 1:N
	        H += _BD1_ham_1site(basis,j;μ = μs[j]+dbias[j],h,Δ,U)
	    end
	    return H
	end
end

# ╔═╡ e13e0187-464c-48de-a08b-5e15c97323af
Ns = 2:4

# ╔═╡ c8417411-9ca6-45ef-a913-fc08acaf47b1
begin
	basis(N) = FermionParityBasis(FermionBasis(N, (:↑,:↓)))
	cs(N) = particles(basis(N))
	μsyms(N) = ntuple(i->Symbol(:μ,i),N)
	randparams(N) = rand(9+N)
end

# ╔═╡ 1b94162b-e895-4827-8815-dcc5c0ede0e6
parameternames(N) = [:t,:Δ,:V,:θ,:h,:U,:α,:bias,:Δ1,μsyms(N)...]

# ╔═╡ c16cf91f-3627-4174-85bc-9dd66119a858
begin
	densehamiltonian(N) = (t,Δ,V,θ,h,U,α,bias,Δ1,μs...) -> QuantumDots.BlockDiagonal(BD1_hamqd_dis(basis(N);μs,t,Δ,V,θ,h,U,α,Δ1,bias)).blocks
	sparsehamiltonian(N) = (t,Δ,V,θ,h,U,α,bias,Δ1,μs...) -> QuantumDots.spBlockDiagonal(BD1_hamqd_dis(basis(N);μs,t,Δ,V,θ,h,U,α,Δ1,bias)).blocks
	randblocks = Dict(N=>densehamiltonian(N)(randparams(N)...) for N in Ns)
	randblockssparse = Dict(N=>sparsehamiltonian(N)(randparams(N)...) for N in Ns)
	densehams = Dict(N=> [[deepcopy(first(randblocks[N])),
		deepcopy(last(randblocks[N]))] for _ in 1:nthreads()] for N in Ns)
	sparsehams = Dict(N=> [[deepcopy(first(randblockssparse[N])),
		deepcopy(last(randblockssparse[N]))] for _ in 1:nthreads()] for N in Ns)
    _oddhams! = Dict(N=>QuantumDots.generate_fastham(first ∘ densehamiltonian(N),parameternames(N)...)[2] for N in Ns)
    _evenhams! = Dict(N=>QuantumDots.generate_fastham(last ∘ densehamiltonian(N),parameternames(N)...)[2] for N in Ns)
    _oddhamssparse! = Dict(N=>QuantumDots.generate_fastham(first ∘ sparsehamiltonian(N),parameternames(N)...)[2] for N in Ns)
    _evenhamssparse! = Dict(N=>QuantumDots.generate_fastham(last ∘ sparsehamiltonian(N),parameternames(N)...)[2] for N in Ns)
	
	# _oddham, _oddham! = QuantumDots.generate_fastham(first ∘ generator,:t,:Δ,:V,:θ,:h,:U,:α,:bias,:Δ1,μsyms...)
 #    _oddhamsp, _oddhamsp! = QuantumDots.generate_fastham(first ∘ generatorsp,:t,:Δ,:V,:θ,:h,:U,:α,:bias,:Δ1,μsyms...)
 #    _evenham, _evenham! = QuantumDots.generate_fastham(last ∘ generator,:t,:Δ,:V,:θ,:h,:U,:α,:bias,:Δ1,μsyms...)
 #    _evenhamsp, _evenhamsp! = QuantumDots.generate_fastham(last ∘ generatorsp,:t,:Δ,:V,:θ,:h,:U,:α,:bias,:Δ1,μsyms...)
	densehamiltonian! = Dict(N=> params -> (n = threadid();_oddhams![N](densehams[N][n][1],params); _evenhams![N](densehams[N][n][2], params); densehams[N][n]) for N in Ns)
	sparsehamiltonian! = Dict(N=> params -> (n = threadid();_oddhamssparse![N](sparsehams[N][n][1],params); _evenhamssparse![N](sparsehams[N][n][2],params); sparsehams[N][n]) for N in Ns)
	
 #    function fasthamall!(params)
	# 	#_fastham!(hams[Threads.threadid()],params)
	# 	_oddham!(hams[Threads.threadid()][1],params)
	# 	_evenham!(hams[Threads.threadid()][2],params)
	# 	hams[Threads.threadid()]
	# end
	# function fasthamallsp!(params)
	# 	#_fastham!(hams[Threads.threadid()],params)
	# 	_oddhamsp!(hamssp[Threads.threadid()][1],params)
	# 	_evenhamsp!(hamssp[Threads.threadid()][2],params)
	# 	hamssp[Threads.threadid()]
	# end
end

# ╔═╡ ce1150f1-5ac0-4a99-b7e6-7bab919076d2
t=1.0

# ╔═╡ 1fcdf3e2-97a9-4b10-b23a-ef8e294e4b62
v0s = Dict(N=>rand(div(4^N,2)) for N in Ns)

# ╔═╡ d2dbda16-5c9a-46b6-8027-0dac4b78731c
md"""
tol = 10^-$tolexp $(@bind tolexp Slider(3:14,default=6)) 
"""

# ╔═╡ dac50315-9d43-401e-af05-759ce33d3a07
tol = 10.0^(-tolexp)

# ╔═╡ b32d769f-d149-472a-9e3a-10005c2f7846
begin
	exactsolver = Dict(:solver=>:Exact)
	krylovsolvers = Dict(N=>Dict(:solver=> :Krylov, :v0=>v0s[N], :tol => tol) for N in Ns)
end

# ╔═╡ c4da0cc9-1fc1-4ba1-9263-f9bd2707e32d
md"""
Matrix structure: $(@bind matrix_structure Select([:Dense, :Sparse],default=:Sparse))
"""

# ╔═╡ 1c332f07-a2d3-455e-8879-c1a0618b0dc9
hamiltonian! = if matrix_structure == :Sparse
		densehamiltonian!
elseif matrix_structure == :Dense
	sparsehamiltonian!
end

# ╔═╡ ab5b3104-b0d8-43d8-a808-85599da7e4fa
# ╠═╡ disabled = true
#=╠═╡
md"""
Solver: $(@bind solver_selection Select((matrix_structure == :Dense ? [:Exact] : [:Exact, :Krylov]),default=:Krylov)) 
"""
  ╠═╡ =#

# ╔═╡ 3f430364-b749-45e2-a92c-58252e056193
function LinearAlgebra.eigen((Hodd,Heven),solver)
	if solver[:solver] == :Krylov
			evenvals, evenvecs = eigsolve(Hermitian(Heven),solver[:v0],2,:SR;tol = solver[:tol],issymmetric=true,ishermitian=true)
		oddvals, oddvecs = eigsolve(Hermitian(Hodd),solver[:v0],2,:SR;tol = solver[:tol],issymmetric=true,ishermitian=true)
	elseif solver[:solver] == :Exact
		evenvals, _evenvecs = eigen(Heven)
		oddvals, _oddvecs = eigen(Hodd)
		evenvecs = eachcol(_evenvecs)
		oddvecs = eachcol(_oddvecs)
	end
	return oddvals,evenvals,oddvecs,evenvecs
end

# ╔═╡ bbb6932e-1bb4-43dd-9f71-02d1edd9e380
LinearAlgebra.eigen(H) = eigen(H,solver)

# ╔═╡ fb4e6268-d918-42a1-b8be-2cc1dfcf285e
function paramscan!(gapratios,mpus,calc_data,enumeratediter)
    # fulld = copy(d)
    #p = Progress(length(iter))
    # function calc_advance(params)
    #     d = calc_data(params)
    # end
    #data = Folds.map(calc_data,iter)
	Threads.@threads :static for (n,p) in enumeratediter
		gp,mpu = calc_data(p)
		gapratios[n] = gp
		mpus[n] = mpu
	end
	# gapratios .= first.(data)
	# mpus .= last.(data)
    return gapratios, mpus
end

# ╔═╡ 5daa28c7-7770-4375-a7a6-b656556e5da1
allμs(μ,N) = ntuple(i->μ,N)

# ╔═╡ 574d9dbf-bab1-4563-b2a7-898e6129d508
function heatandcontour(x,y,gapratios,mpus,sweetspot; showtitle=true)
    #sweetspot = (row[:sweet_spot][1]/row[:Δ],row[:sweet_spot][2])
    plotss() = scatter!(sweetspot,legend=false,c=:Red)
    #mpu = map(z->abs(z[1]),row[:param_scan]["MPu"])
    #e = map(gapratio,row[:param_scan]["es"])
    heatmap(x,y,50gapratios',xlabel="Δ1/Δ",ylabel="μ",c=:redsblues,clims = 1 .* (-1,1),cbar=false)
    
    ticks = range(y[1],y[end],length=6)
    ticklabels = [ @sprintf("%5.1f",x) for x in ticks ]
    plot!(yticks=(ticks,ticklabels))
    contour!(x,y,mpus',xlabel="Δ1/Δ",ylabel="μ",levels=range(.7,1,length=10))
    plotss()
end

# ╔═╡ 0c0698fd-a9e0-4934-b290-7477306d3d50
function heatandcontour2(x,y,gapratios,mpus,sweetspot,(sweet_gap,sweet_mpu))
    #sweetspot = (row[:sweet_spot][1]/row[:Δ],row[:sweet_spot][2])
    plotss() = scatter!(sweetspot,legend=false,c=:Red)
    #mpu = map(z->abs(z[1]),row[:param_scan]["MPu"])
    #e = map(gapratio,row[:param_scan]["es"])
    heatmap(x,y,abs.(mpus'),xlabel="Δ1/Δ",ylabel="μ",c=:inferno,clims = 1 .* (0,1),cbar=true, title = @sprintf("gap: %5.3f, MPu: %5.3f", sweet_gap,sweet_mpu))
    
    ticks = range(y[1],y[end],length=6)
    ticklabels = [ @sprintf("%1.1f",x) for x in ticks ]
    plot!(yticks=(ticks,ticklabels))
    contour!(x,y,gapratios',xlabel="Δ1/Δ",ylabel="μ",lw = 2,levels=[1e-32],c = :green)
	
    plotss()
end

# ╔═╡ b969ebc0-90de-4ef1-89a4-b4298863e0f6
md"""
non-local cond: $(@bind run_non_local_conductance CheckBox(default=false))

gate optimization: $(@bind run_gate_optimization CheckBox(default=false))

jacobians: $(@bind run_jacobians CheckBox(default=true))
"""

# ╔═╡ 0674b1e4-4767-4b2e-99dc-abc2a5533af1
md"""
N: $(@bind N Slider(Ns,default=Ns[1],show_value=true))
"""

# ╔═╡ dae7470c-51a2-4f8f-bab3-76bf8f552544
begin #Operators
	majps = Dict(N=>[sparse(basis(N)*(cs(N)[i,s]' + cs(N)[i,s])*basis(N))[1:div(4^N,2),div(4^N,2)+1:end] for (i,s) in Base.product(1:N,(:↑,:↓))] for N in N)
	majms = Dict(N=>[sparse(basis(N)*(cs(N)[i,s]' - cs(N)[i,s])*basis(N))[1:div(4^N,2),div(4^N,2)+1:end] for (i,s) in Base.product(1:N,(:↑,:↓))] for N in Ns)
	parity = Dict(N=>sparse(basis(N)*ParityOperator()*basis(N)) for N in Ns)
end

# ╔═╡ a081699f-df96-4d10-88e6-a0c7ea2dd67f
function measure_data((oddvals,evenvals,oddvecs,evenvecs))
	N = Int((log2(length(first(oddvecs))) + 1)/2)
	ws = [first(oddvecs)'*op*first(evenvecs) for op in majps[N]]
	vs = [first(oddvecs)'*op*first(evenvecs) for op in majms[N]]
	majcoeffs = [ws;;; vs]
	mpu = MPu(majcoeffs)
	gapratio([oddvals[1:2],evenvals[1:2]])::Float64, mpu::Float64
end

# ╔═╡ cce1e924-891d-4735-b8ba-3d3408a9afdd
function solve(H,solver = exactsolver)
 	eigsol = eigen(H,solver)
	return measure_data(eigsol)
end

# ╔═╡ f44f8dca-81f1-4cab-9b36-0d0cfdc4e381
md"""
Δ1start: $(@bind Δ1start Slider(range(-8t,0,length=21),default=-2t,show_value=true))
Δ1end: $(@bind Δ1end Slider(range(0,8t,length=21),default=2t,show_value=true))

μstart: $(@bind μstart Slider(range(-4t,0,length=11),default=-2t,show_value=true))
μend: $(@bind μend Slider(range(0,4t,length=11),default=2t,show_value=true))

resolution: $(@bind resolution Slider(10:100,default=40,show_value=true))
"""

# ╔═╡ a1af8874-b3e6-4f36-aac2-80a978a23128
begin
	μrange = (μstart,μend)
	Δ1range = (Δ1start,Δ1end)
	Δ1s = collect(range(Δ1range...,length=resolution))
	dμs = collect(range(μrange...,length=resolution))
end

# ╔═╡ b25de208-0911-49e4-ae35-97af5b3449f8
md"""
## Parameters

α: $(@bind α Slider(range(0,t,length=21),default=0,show_value=true))
V: $(@bind V Slider(range(0,2t,length=21),default=0,show_value=true))

h: $(@bind h Slider(range(0,20t,length=81),default=40t,show_value=true))
θ: $(@bind θ Slider(range(0,pi,length=21),default=pi/2,show_value=true))

Δ: $(@bind Δ Slider(range(0.1,4t,length=21),default=2t,show_value=true))

U: $(@bind U Slider(range(0,20t,length=21),default=10t,show_value=true))

μbias: $(@bind μbias Slider(range(-2t,2t,length=21),default=0,show_value=true))
MaxTime: $(@bind MaxTime Slider(range(0,10,length=101),default=0.2,show_value=true))
"""

# ╔═╡ 85e124b0-c48d-4bf3-8e07-f04bcd9892f7
begin
	[t,Δ,V,θ,h,U,α,μbias]
	dmusolve(N,(Δ1,μ)) = solve(hamiltonian![N]([t,Δ,V,θ,h,U,α,μbias,Δ1,allμs(μ,N)...]), krylovsolvers[N])
end

# ╔═╡ 5a0933d6-e098-4aa9-af4f-30676ef518f9
dmusolve(N,(1.0,1.0))

# ╔═╡ 4a310f82-0ccc-4b5e-a684-71d4f037e83c
begin
	μs = -h .+ dμs
	iter = Base.product(Δ1s,μs);
	enumeratediter = collect(enumerate(iter))
end

# ╔═╡ f758130e-3fab-4f08-8e66-a98ccd9e5365
paramscan_gapratios = zeros(Float64,size(iter));

# ╔═╡ f3434bcb-7670-4fd6-9623-9227143b4296
paramscan_mpus = zeros(Float64,size(iter));

# ╔═╡ 3578201e-24fd-45c6-9947-94fa62f2e618
@time gapratios, mpus = paramscan!(paramscan_gapratios, paramscan_mpus, p->dmusolve(N,p), enumeratediter);

# ╔═╡ 4cfbf14a-67d0-489d-ab5f-602c1a6712ad
begin
	res = bboptimize(Δ1μ-> cost_function(dmusolve(N,Δ1μ)[1:2]...), map(sc->sum(sc)/2,[Δ1range,(μs[1],μs[end])]); SearchRange = [Δ1range,(μs[1],μs[end])], NumDimensions = 2, MaxTime)
	sweet_spot = best_candidate(res)
	sweet_Δ1, sweet_μ = sweet_spot
	sweet_gap,sweet_mpu = dmusolve(N,sweet_spot)
end

# ╔═╡ 08ddf3f6-157f-4b99-ac83-b6b3cd5c8e0a
heatandcontour2(Δ1s ./ Δ,μs,gapratios,mpus,(sweet_Δ1/Δ,sweet_μ),(sweet_gap,sweet_mpu))

# ╔═╡ 4a639756-3051-4702-a210-d8933ae331aa
begin
	energyjacobian(params) = jacobian(central_fdm(5, 1), first ∘ solve ∘ densehamiltonian![N], params)[1] 
	mpujacobian(params) = jacobian(central_fdm(5, 1), last ∘ solve ∘ densehamiltonian![N], params)[1] 
end

# ╔═╡ 1cad4200-5efd-4966-940c-1f6da8ba3c27
sweet_spot_params = [t,Δ,V,θ,h,U,α,μbias,sweet_Δ1,allμs(sweet_μ,N)...]

# ╔═╡ a348c828-4289-48ec-9dc5-c629a8502565
sweet_spot_energy_jacobian = energyjacobian(sweet_spot_params)

# ╔═╡ ce68f40c-14cb-4df3-a489-09ea49291109
sweet_spot_mpu_jacobian = mpujacobian(sweet_spot_params)

# ╔═╡ ba2b48d2-d22d-42fe-b18f-46fe1babdb13
begin
	barplotxticks = (eachindex(parameternames(N)),parameternames(N))
	barplote = bar(sweet_spot_energy_jacobian',xticks = barplotxticks, title= string("|δgapratio|=",@sprintf("%5.3f",norm(sweet_spot_energy_jacobian))))
	barplotmpu = bar(sweet_spot_mpu_jacobian',xticks = barplotxticks, title= string("|δMPu|=",@sprintf("%5.3f",norm(sweet_spot_mpu_jacobian))))
	plot(barplote,barplotmpu,size=(800,400),plot_title = "Jacobian")
end

# ╔═╡ 321d1522-f687-47c0-8929-a7fed3fe6121
begin
	barplotxticksgate = (eachindex(parameternames(N)[end-N+1:end]),parameternames(N)[end-N+1:end])
	barplotegate = bar(sweet_spot_energy_jacobian[:,end-N+1:end]',xticks = barplotxticksgate, title= string("|δgapratio|=",@sprintf("%5.3f",norm(sweet_spot_energy_jacobian[end-N+1:end]))))
	barplotmpugate = bar(sweet_spot_mpu_jacobian[:,end-N+1:end]',xticks = barplotxticksgate, title= string("|δMPu|=",@sprintf("%5.3f",norm(sweet_spot_mpu_jacobian[end-N+1:end]))))
	plot(barplotegate,barplotmpugate,size=(800,400),plot_title = "Jacobian, gates")
end

# ╔═╡ 6afdb8d7-5929-427f-a0a6-19dac33014b0
md"""
xparam: $(@bind xparam Select(parameternames(N),default = :μ1))
xparamresolution: $(@bind xparamresolution Slider(10:100,default=40,show_value=true))

xparamstart: $(@bind xparamstart Slider(range(-t,0,length=11),default=-t,show_value=true))
xparamend: $(@bind xparamend Slider(range(0,t,length=11),default=t,show_value=true))
"""

# ╔═╡ e1550793-c4e6-4e6f-b4b0-4ba3a38c17bf
xparamindex = findfirst(p->p==xparam,parameternames(N))

# ╔═╡ 6800302a-3cfb-474e-a4b1-4539bf96f975
function sweet_spot_perturbation(N,p0,xparamindex,xs)
	dp = [n == xparamindex ? 1.0 : 0.0 for n in 1:length(p0)]  
	data = map(x->solve(densehamiltonian![N](p0 .+ dp*x),exactsolver), xs)
	gapratios = first.(data)
	mpus = last.(data)
	return gapratios, mpus
end

# ╔═╡ fb6a7184-a697-4027-9b3a-7b75f6e74e13
begin
	xparamvalues = range(xparamstart,xparamend,length=xparamresolution)
	sweet_spot_perturbation_data = sweet_spot_perturbation(N,sweet_spot_params,
		xparamindex,xparamvalues)
end

# ╔═╡ 9e504dd7-01dd-4f08-9577-92182c0c9487
begin
	pgapratio = plot(xparamvalues,(sweet_spot_perturbation_data[1]),xlabel=xparam,ylabel = "gapratio")
	mputicks = range(minimum(abs.(sweet_spot_perturbation_data[2])),1,length=10)
	mputicklabels = [ @sprintf("%1.3f",x) for x in mputicks]
	pmpu = plot(xparamvalues,abs.(sweet_spot_perturbation_data[2]),xlabel=xparam,ylabel = "MPu", yticks = (mputicks,mputicklabels))
	plot(pgapratio,pmpu)
end

# ╔═╡ d13930ab-d534-4090-8fb3-6a3f28a4ba01
begin #Gate Optimization
	if run_gate_optimization
		_gatesolve(μs) =  solve(hamiltonian![N]([t,Δ,V,θ,h,U,α,μbias,sweet_Δ1,μs...]))
		res_gate = bboptimize(μs-> cost_function(_gatesolve(μs)[1:2]...), fill(sweet_μ,N); SearchRange = fill((μs[1],μs[end]),N), NumDimensions = 2, MaxTime)
		sweet_spot_gate = best_candidate(res_gate)
		sweet_gap_gate,sweet_mpu_gate = _gatesolve(sweet_spot_gate)
	else
		sweet_spot_gate = sweet_spot
		sweet_gap_gate, sweet_mpu_gate = sweet_gap, sweet_mpu
	end
end

# ╔═╡ bad6d49c-d916-409d-ae43-9edb4af50578
sweet_gap,sweet_mpu

# ╔═╡ 8d98aa45-0916-4617-851a-5bdcd0d34c15
sweet_gap_gate,sweet_mpu_gate

# ╔═╡ 6f1a3db1-564d-4b6b-870a-e1eda8ea09c1
sweet_gap_gate/sweet_gap, abs(1-sweet_mpu_gate) /abs(1-sweet_mpu)

# ╔═╡ 870aeef3-6127-438a-9138-904f29be99c0
(μs[1],μs[end])

# ╔═╡ 980acaa2-c82e-46e2-ad21-702fc1ee8743
# ╠═╡ disabled = true
#=╠═╡
@bind params PlutoUI.combine() do Child
	md"""
	## Parameters
	
	α: $(Child(Slider(range(0,t,length=10),default=0,show_value=true)))
	
	h: $(Child(Slider(range(0,50t,length=50),default=40t,show_value=true)))
	
	θ: $(Child(Slider(range(0,pi,length=10),default=pi/2,show_value=true)))
	
	Δ: $(Child(Slider(range(0,4t,length=10),default=2t,show_value=true)))
	
	V: $(Child(Slider(range(0,2t,length=10),default=0,show_value=true)))
	
	U: $(Child(Slider(range(0,20t,length=10),default=10t,show_value=true)))
	
	μbias: $(Child(Slider(range(0,2t,length=10),default=0,show_value=true)))
	"""
end
  ╠═╡ =#

# ╔═╡ 07129582-c051-4c73-be4b-cd6c0719f648
begin
	function lindbladian(Heff,Ls)
	    id = one(Heff)
	    -1im*(-Heff⊗id + id⊗Heff) + sum(superjump,Ls, init = 0*id⊗id)
	end
	
	superjump(L) = L⊗L - 1/2*kronsum((L'*L),L'*L)
	current(ρ,op,sj) = tr(op * reshape(sj*ρ,size(op)))
	Base.one(T::LinearMap) = LinearMap(I,size(T,1))
	commutator(T1,T2) = -kron(T1,T2) .+ kron(T2,T1)#-T1⊗T2 + T2⊗T1
	function transform_jump_op(H,L,T,μ)
	    id = one(H)
	    comm = (commutator(H,id))
	    reshape(sqrt(fermidirac(comm,T,μ))*vec(L),size(L))
	end
	# function transform_jump_op_out(H,L,T,μ)
	#     id = one(H)
	#     comm = Matrix(commutator(H,id))
	#     reshape(sqrt(I - fermidirac(comm,T,μ))*vec(L),size(L))
	# end
	fermidirac(E::Diagonal,T,μ) = (I + exp(E/T)exp(-μ/T))^(-1)
end

# ╔═╡ 90b665b7-0f08-4005-8147-52632c362697
leftleadops, rightleadops = let c = cs(N), b = basis(N)
	([sparse(b*c[1,:↑]'*b), sparse(b*c[1,:↓]'*b), sparse(b*c[1,:↑]*b), sparse(b*c[1,:↓]*b)],
	[sparse(b*c[N,:↑]'*b), sparse(b*c[N,:↓]'*b), sparse(b*c[N,:↑]*b), sparse(b*c[N,:↓]*b)])
end

# ╔═╡ 6ceae5f7-36f5-411d-b19f-bdfd3d2a4bfe
total_particle_number_op = sparse(basis(N)*sum(numberop(cs(N),(i,:↓))+numberop(cs(N),(i,:↑)) for i in 1:N)*basis(N))

# ╔═╡ 4ee83133-32ca-4a36-9cc2-82d3de77e092
function conductance(p0,μL,μR,leftleadops,rightleadops,measureop,TL,TR)
    #μ = -Vg
    #μL = 0*Vbias/2
    #μR = -Vbias/2
	# c = particles(basis)
	Hs = densehamiltonian![N](p0)
	#Hfull = cat(H...;dims=[1,2])
	oeigvals, oeigvecs = eigen!(Hs[1])
	eeigvals, eeigvecs = eigen!(Hs[2])
	eigvals = vcat(oeigvals,eeigvals)
	S::SparseMatrixCSC{Float64, Int64} = cat(sparse(oeigvecs),sparse(eeigvecs); dims=[1,2])
    D::Diagonal{Float64,Vector{Float64}} = Diagonal(eigvals)
	
    left_jumps_in = leftleadops[1:2]
    left_jumps_out = leftleadops[3:4]
    right_jumps_in = rightleadops[1:2]
    right_jumps_out = rightleadops[3:4]
    #vals,S = eigen(H)
	# S = eigvecs
    left_jumps_in2 = map(L->transform_jump_op(D,S'*L*S,TL,μL), left_jumps_in)
    left_jumps_out2 = map(L->transform_jump_op(D,S'*L*S,TL,-μL), left_jumps_out)
    right_jumps_in2 = map(L->transform_jump_op(D,S'*L*S,TR,μR), right_jumps_in)
    right_jumps_out2 = map(L->transform_jump_op(D,S'*L*S,TR,-μR), right_jumps_out)

    superlind = Matrix(lindbladian(D, vcat(left_jumps_in2,left_jumps_out2,right_jumps_in2,right_jumps_out2)))
    #lindvals,lindvecs = eigsolve(superlind,rand(size(superlind,1)),2,EigSorter(abs;rev=false); tol = 1e-14)
    lindvals,_lindvecs = eigen(superlind,sortby=abs)
	lindvecs = eachcol(_lindvecs)
    particle_number = S'*(measureop)*S
    curr1 = real.(map(L->current(first(lindvecs)/(tr(reshape(first(lindvecs),size(S)))), particle_number, superjump(L)), hcat(left_jumps_in2,left_jumps_out2,right_jumps_in2,right_jumps_out2)))
    curr1
end

# ╔═╡ d7df5962-9a2c-4049-b5c6-d647e94599ee
function conductance2(p0,μL,μR,leftleadops,rightleadops,measureop,TL,TR)
	Hs = densehamiltonian![N](p0)
	oeigvals, oeigvecs = eigen!(Hs[1])
	eeigvals, eeigvecs = eigen!(Hs[2])
	eigvals = vcat(oeigvals,eeigvals)
	S::SparseMatrixCSC{Float64, Int64} = cat(sparse(oeigvecs),sparse(eeigvecs); dims=[1,2])
    D::Diagonal{Float64,Vector{Float64}} = Diagonal(eigvals)
	
    left_jumps_in = leftleadops[1:2]
    left_jumps_out = leftleadops[3:4]
    right_jumps_in = rightleadops[1:2]
    right_jumps_out = rightleadops[3:4]

    left_jumps_in2 = map(L->transform_jump_op(D,S'*L*S,TL,μL), left_jumps_in)
    left_jumps_out2 = map(L->transform_jump_op(D,S'*L*S,TL,-μL), left_jumps_out)
    right_jumps_in2 = map(L->transform_jump_op(D,S'*L*S,TR,μR), right_jumps_in)
    right_jumps_out2 = map(L->transform_jump_op(D,S'*L*S,TR,-μR), right_jumps_out)

    superlind = (lindbladian(D, vcat(left_jumps_in2,left_jumps_out2,right_jumps_in2,right_jumps_out2)))
    lindvals,lindvecs = eigsolve(superlind,rand(size(superlind,1)),2,EigSorter(abs;rev=false); tol = 1e-9)
    particle_number = S'*(measureop)*S
    curr1 = real.(map(L->current(first(lindvecs)/(tr(reshape(first(lindvecs),size(S)))), particle_number, superjump(L)), hcat(left_jumps_in2,left_jumps_out2,right_jumps_in2,right_jumps_out2)))
    curr1
end

# ╔═╡ 62f7287d-1515-4d04-a6d4-edec10a67290
#@code_warntype conductance2([t,Δ,V,θ,h,U,α,μbias,sweet_Δ1,allμs(0.0,N)...],0.0, 0.0, leftleadops, rightleadops, total_particle_number_op,TL,TR)

# ╔═╡ 2863eaed-0747-4586-b4fb-1326abcf263c
#@time conductance([t,Δ,V,θ,h,U,α,μbias,sweet_Δ1,allμs(0.0,N)...],0.0, 0.0, leftleadops, rightleadops, total_particle_number_op,TL,TR)

# ╔═╡ e5208402-f264-4c58-bcc7-dc871a46bf1c
@time eigen!(rand(div(4^4,2),div(4^4,2)));

# ╔═╡ 02a227fe-1307-4999-97d0-25916d1c0460
#@profview conductance([t,Δ,V,θ,h,U,α,μbias,sweet_Δ1,allμs(0.0,N)...],0.0, 0.0, leftleadops, rightleadops, total_particle_number_op,TL,TR)

# ╔═╡ 1f4a0090-1ae1-4f48-a769-a0629b2859c4
md"""
Vresolution: $(@bind Vresolution Slider(10:40,default=10,show_value=true))

Vbiasstart: $(@bind Vbiasstart Slider(range(-4t,0,length=11),default=-2t,show_value=true))
Vbiasend: $(@bind Vbiasend Slider(range(0,4t,length=11),default=2t,show_value=true))

TL: $(@bind TL Slider(range(0.01,t,length=11),default=0.5t,show_value=true))
TR: $(@bind TR Slider(range(0.01,t,length=11),default=0.5t,show_value=true))
"""

# ╔═╡ 9a6f40e1-14fd-4e55-84f2-53a951103621
Vbiasrange = range(Vbiasstart,Vbiasend,length=Vbiasresolution)

# ╔═╡ 8276c9a1-3cd5-4ba3-a54d-d7fe81e12577
nonlocalconductance(μs, μR) = sum(conductance([t,Δ,V,θ,h,U,α,μbias,sweet_Δ1,μs...],0.0, μR, leftleadops, rightleadops, total_particle_number_op,TL,TR)[:,1:2])

# ╔═╡ 9098b9d0-3954-4869-b4a0-16235982d0e8
nonlocaldiffconductance(μs,μR) = central_fdm(2,1,factor = 1e12)(μR->nonlocalconductance(μs,μR),μR)

# ╔═╡ 0ee04999-10ac-4fbb-92c6-017d68cc7eca
nonlocaldiffconductance(allμs(sweet_μ,N),0.0)

# ╔═╡ a300acc0-c0fb-4633-859f-4d0a8c52e993
nonlocalconductance(allμs(sweet_μ,N),0.0)

# ╔═╡ ce9049ed-a57f-42db-88c6-941d82583fe2
begin
	Vrange = range(μs[1],μs[end],length=Vresolution)
	if run_non_local_conductance
		nlcdata = Folds.map(lr->nonlocaldiffconductance([lr[1],fill(0,N-2)...,lr[2]],0.0),Base.product(Vrange,Vrange))
	else
		nlcdata = zeros(Vresolution,Vresolution)
	end
end

# ╔═╡ 68fcee3c-d837-4e4a-9c1f-014fe0f02b02
heatmap(Vrange,Vrange,nlcdata, clims = (-1,1) .* maximum(abs.(nlcdata)) ./ 5,c = :redsblues)

# ╔═╡ Cell order:
# ╠═a9a6cdf0-76db-11ed-0076-6dfc37097407
# ╠═bc605fb9-16f9-44a4-bb58-c471e42e9754
# ╠═33f235a6-0220-489f-97c8-40d8493048ca
# ╟─1fb9255b-da89-4358-b1e3-91adb691b377
# ╠═127abb1f-7850-402f-a3a9-d4a24ba20246
# ╟─12eaa1c6-1c66-4ebc-9577-31569a706fa0
# ╟─9ad954b1-a9c1-4c0c-a6a8-44d62d085f71
# ╟─3fb9c1b3-8fd8-40e8-a269-3fa547e35dbd
# ╠═e13e0187-464c-48de-a08b-5e15c97323af
# ╠═c8417411-9ca6-45ef-a913-fc08acaf47b1
# ╠═dae7470c-51a2-4f8f-bab3-76bf8f552544
# ╠═c16cf91f-3627-4174-85bc-9dd66119a858
# ╟─1b94162b-e895-4827-8815-dcc5c0ede0e6
# ╟─ce1150f1-5ac0-4a99-b7e6-7bab919076d2
# ╟─1fcdf3e2-97a9-4b10-b23a-ef8e294e4b62
# ╟─b32d769f-d149-472a-9e3a-10005c2f7846
# ╟─1c332f07-a2d3-455e-8879-c1a0618b0dc9
# ╟─dac50315-9d43-401e-af05-759ce33d3a07
# ╟─ab5b3104-b0d8-43d8-a808-85599da7e4fa
# ╟─d2dbda16-5c9a-46b6-8027-0dac4b78731c
# ╟─c4da0cc9-1fc1-4ba1-9263-f9bd2707e32d
# ╠═cce1e924-891d-4735-b8ba-3d3408a9afdd
# ╟─3f430364-b749-45e2-a92c-58252e056193
# ╠═bbb6932e-1bb4-43dd-9f71-02d1edd9e380
# ╠═a081699f-df96-4d10-88e6-a0c7ea2dd67f
# ╠═fb4e6268-d918-42a1-b8be-2cc1dfcf285e
# ╠═f758130e-3fab-4f08-8e66-a98ccd9e5365
# ╠═f3434bcb-7670-4fd6-9623-9227143b4296
# ╟─a1af8874-b3e6-4f36-aac2-80a978a23128
# ╠═5a0933d6-e098-4aa9-af4f-30676ef518f9
# ╠═5daa28c7-7770-4375-a7a6-b656556e5da1
# ╠═85e124b0-c48d-4bf3-8e07-f04bcd9892f7
# ╠═4a310f82-0ccc-4b5e-a684-71d4f037e83c
# ╠═574d9dbf-bab1-4563-b2a7-898e6129d508
# ╠═0c0698fd-a9e0-4934-b290-7477306d3d50
# ╟─b969ebc0-90de-4ef1-89a4-b4298863e0f6
# ╟─0674b1e4-4767-4b2e-99dc-abc2a5533af1
# ╟─f44f8dca-81f1-4cab-9b36-0d0cfdc4e381
# ╟─b25de208-0911-49e4-ae35-97af5b3449f8
# ╠═08ddf3f6-157f-4b99-ac83-b6b3cd5c8e0a
# ╟─ba2b48d2-d22d-42fe-b18f-46fe1babdb13
# ╠═321d1522-f687-47c0-8929-a7fed3fe6121
# ╠═3578201e-24fd-45c6-9947-94fa62f2e618
# ╠═4cfbf14a-67d0-489d-ab5f-602c1a6712ad
# ╟─9e504dd7-01dd-4f08-9577-92182c0c9487
# ╠═4a639756-3051-4702-a210-d8933ae331aa
# ╟─a348c828-4289-48ec-9dc5-c629a8502565
# ╟─ce68f40c-14cb-4df3-a489-09ea49291109
# ╠═1cad4200-5efd-4966-940c-1f6da8ba3c27
# ╟─6afdb8d7-5929-427f-a0a6-19dac33014b0
# ╠═e1550793-c4e6-4e6f-b4b0-4ba3a38c17bf
# ╠═6800302a-3cfb-474e-a4b1-4539bf96f975
# ╠═fb6a7184-a697-4027-9b3a-7b75f6e74e13
# ╠═d13930ab-d534-4090-8fb3-6a3f28a4ba01
# ╠═bad6d49c-d916-409d-ae43-9edb4af50578
# ╠═8d98aa45-0916-4617-851a-5bdcd0d34c15
# ╠═6f1a3db1-564d-4b6b-870a-e1eda8ea09c1
# ╠═870aeef3-6127-438a-9138-904f29be99c0
# ╟─980acaa2-c82e-46e2-ad21-702fc1ee8743
# ╠═07129582-c051-4c73-be4b-cd6c0719f648
# ╟─90b665b7-0f08-4005-8147-52632c362697
# ╟─6ceae5f7-36f5-411d-b19f-bdfd3d2a4bfe
# ╠═4ee83133-32ca-4a36-9cc2-82d3de77e092
# ╠═d7df5962-9a2c-4049-b5c6-d647e94599ee
# ╠═9a6f40e1-14fd-4e55-84f2-53a951103621
# ╠═62f7287d-1515-4d04-a6d4-edec10a67290
# ╠═2863eaed-0747-4586-b4fb-1326abcf263c
# ╠═e5208402-f264-4c58-bcc7-dc871a46bf1c
# ╠═02a227fe-1307-4999-97d0-25916d1c0460
# ╠═8276c9a1-3cd5-4ba3-a54d-d7fe81e12577
# ╠═9098b9d0-3954-4869-b4a0-16235982d0e8
# ╠═a300acc0-c0fb-4633-859f-4d0a8c52e993
# ╠═0ee04999-10ac-4fbb-92c6-017d68cc7eca
# ╠═1f4a0090-1ae1-4f48-a769-a0629b2859c4
# ╠═ce9049ed-a57f-42db-88c6-941d82583fe2
# ╠═68fcee3c-d837-4e4a-9c1f-014fe0f02b02
