### A Pluto.jl notebook ###
# v0.19.17

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
	using QuantumDots, LinearAlgebra, Random, BlackBoxOptim, PlutoUI, Plots, Printf, KrylovKit, SparseArrays, Folds, FiniteDifferences
	BLAS.set_num_threads(1)
end

# ╔═╡ bc605fb9-16f9-44a4-bb58-c471e42e9754
Pkg.status()

# ╔═╡ 33f235a6-0220-489f-97c8-40d8493048ca
Threads.nthreads()

# ╔═╡ 127abb1f-7850-402f-a3a9-d4a24ba20246
gapratio(es) = real(diff(es)[1][1])#/(diff(sort(vcat(es...)))[2]))

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

# ╔═╡ ce1150f1-5ac0-4a99-b7e6-7bab919076d2
t=1.0

# ╔═╡ c4da0cc9-1fc1-4ba1-9263-f9bd2707e32d
md"""
Matrix structure: $(@bind matrix_structure Select([:Dense, :Sparse],default=:Sparse))
"""

# ╔═╡ ab5b3104-b0d8-43d8-a808-85599da7e4fa
md"""
Solver: $(@bind solver_selection Select((matrix_structure == :Dense ? [:Exact] : [:Exact, :Krylov]),default=:Krylov)) 
"""

# ╔═╡ d2dbda16-5c9a-46b6-8027-0dac4b78731c
if solver_selection == :Krylov
md"""
tol = 10^-$tolexp $(@bind tolexp Slider(3:14,default=6)) 
"""
else
tolexp = 1e-6
end

# ╔═╡ dac50315-9d43-401e-af05-759ce33d3a07
tol = 10.0^(-tolexp)

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

# ╔═╡ fb4e6268-d918-42a1-b8be-2cc1dfcf285e
function paramscan(calc_data,iter)
    # fulld = copy(d)
    #p = Progress(length(iter))
    function calc_advance(params)
        #next!(p) 
        #yield()
        calc_data(params...)
    end
    data = map(calc_advance,iter)
	gapratios = first.(data)
	mpus = last.(data)
    return gapratios, mpus
end

# ╔═╡ 2b53130c-3ab3-4f43-8772-54a5b0b7b7f7
#fastham!(Δ1,μs...) = fasthamall!([t,Δ,V,θ,h,U,α,μbias,Δ1,μs...])

# ╔═╡ 564f1d21-8ea8-48e1-b30e-cee2936ad32d
#fasthamsp!(Δ1,μs...) = fasthamallsp!([t,Δ,V,θ,h,U,α,μbias,Δ1,μs...])

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
function heatandcontour2(x,y,gapratios,mpus,sweetspot; showtitle=true)
    #sweetspot = (row[:sweet_spot][1]/row[:Δ],row[:sweet_spot][2])
    plotss() = scatter!(sweetspot,legend=false,c=:Red)
    #mpu = map(z->abs(z[1]),row[:param_scan]["MPu"])
    #e = map(gapratio,row[:param_scan]["es"])
    heatmap(x,y,abs.(mpus'),xlabel="Δ1/Δ",ylabel="μ",c=:inferno,clims = 1 .* (0,1),cbar=true)
    
    ticks = range(y[1],y[end],length=6)
    ticklabels = [ @sprintf("%1.1f",x) for x in ticks ]
    plot!(yticks=(ticks,ticklabels))
    contour!(x,y,gapratios',xlabel="Δ1/Δ",ylabel="μ",lw = 2,levels=[1e-100],c = :green)
    plotss()
end

# ╔═╡ 0674b1e4-4767-4b2e-99dc-abc2a5533af1
md"""
N: $(@bind N Slider(2:4,default=2,show_value=true))
"""

# ╔═╡ dae7470c-51a2-4f8f-bab3-76bf8f552544
begin
	basis = FermionParityBasis(FermionBasis(N, (:↑,:↓)))
	a = particles(basis)
	majps = [sparse(basis*(a[i,s]' + a[i,s])*basis)[1:div(4^N,2),div(4^N,2)+1:end] for (i,s) in Base.product(1:N,(:↑,:↓))]
	majms = [sparse(basis*(a[i,s]' - a[i,s])*basis)[1:div(4^N,2),div(4^N,2)+1:end] for (i,s) in Base.product(1:N,(:↑,:↓))]
	parity = sparse(basis*ParityOperator()*basis)
	μsyms = ntuple(i->Symbol(:μ,i),N)
	randparams = rand(9+N)
end

# ╔═╡ c16cf91f-3627-4174-85bc-9dd66119a858
begin
	generator(t,Δ,V,θ,h,U,α,bias,Δ1,μs...) = QuantumDots.BlockDiagonal(BD1_hamqd_dis(basis;μs,t,Δ,V,θ,h,U,α,Δ1,bias)).blocks
	generatorsp(t,Δ,V,θ,h,U,α,bias,Δ1,μs...) = QuantumDots.spBlockDiagonal(BD1_hamqd_dis(basis;μs,t,Δ,V,θ,h,U,α,Δ1,bias)).blocks
	randblocks = generator(randparams...)
	randblockssp = generatorsp(randparams...)
	hams = [(deepcopy(first(randblocks)),
		deepcopy(last(randblocks))) for _ in Threads.nthreads()]
	hamssp = [(deepcopy(first(randblockssp)),
		deepcopy(last(randblockssp))) for _ in Threads.nthreads()]
    _oddham, _oddham! = QuantumDots.generate_fastham(first ∘ generator,:t,:Δ,:V,:θ,:h,:U,:α,:bias,:Δ1,μsyms...)
    _oddhamsp, _oddhamsp! = QuantumDots.generate_fastham(first ∘ generatorsp,:t,:Δ,:V,:θ,:h,:U,:α,:bias,:Δ1,μsyms...)
    _evenham, _evenham! = QuantumDots.generate_fastham(last ∘ generator,:t,:Δ,:V,:θ,:h,:U,:α,:bias,:Δ1,μsyms...)
    _evenhamsp, _evenhamsp! = QuantumDots.generate_fastham(last ∘ generatorsp,:t,:Δ,:V,:θ,:h,:U,:α,:bias,:Δ1,μsyms...)
    function fasthamall!(params)
		#_fastham!(hams[Threads.threadid()],params)
		_oddham!(hams[Threads.threadid()][1],params)
		_evenham!(hams[Threads.threadid()][2],params)
		hams[Threads.threadid()]
	end
	function fasthamallsp!(params)
		#_fastham!(hams[Threads.threadid()],params)
		_oddhamsp!(hamssp[Threads.threadid()][1],params)
		_evenhamsp!(hamssp[Threads.threadid()][2],params)
		hamssp[Threads.threadid()]
	end
end

# ╔═╡ 1c332f07-a2d3-455e-8879-c1a0618b0dc9
hamiltonian! = if matrix_structure == :Sparse
		fasthamallsp!
elseif matrix_structure == :Dense
	fasthamall!
end

# ╔═╡ 1b94162b-e895-4827-8815-dcc5c0ede0e6
parameternames = [:t,:Δ,:V,:θ,:h,:U,:α,:bias,:Δ1,μsyms...]

# ╔═╡ a081699f-df96-4d10-88e6-a0c7ea2dd67f
function measure_data((oddvals,evenvals,oddvecs,evenvecs))
	ws = [first(oddvecs)'*op*first(evenvecs) for op in majps]
	vs = [first(oddvecs)'*op*first(evenvecs) for op in majms]
	majcoeffs = [ws;;; vs]
	mpu = MPu(majcoeffs)
	gapratio([oddvals[1:2],evenvals[1:2]])::Float64, mpu::Float64
end

# ╔═╡ 1fcdf3e2-97a9-4b10-b23a-ef8e294e4b62
v0 = rand(div(4^N,2))

# ╔═╡ b32d769f-d149-472a-9e3a-10005c2f7846
begin
	exactsolver = Dict(:solver=>:Exact)
	krylovsolver = Dict(:solver=> solver_selection, :v0=>v0, :tol => tol)
end

# ╔═╡ b96e7112-bec6-45e5-b100-35c730d6249a
solver = if solver_selection == :Krylov
		krylovsolver
elseif solver_selection == :Exact
	exactsolver
end

# ╔═╡ 3983336d-5563-4d0f-82ce-82501072cb8a
solver

# ╔═╡ cfb4c4cd-5513-41b1-834d-1f3b2b3a53ea
solve(H) = solve(H,solver)

# ╔═╡ bbb6932e-1bb4-43dd-9f71-02d1edd9e380
LinearAlgebra.eigen(H) =eigen(H,solver)

# ╔═╡ cce1e924-891d-4735-b8ba-3d3408a9afdd
function solve(H,solver)
 	eigsol = eigen(H,solver)
	return measure_data(eigsol)
end

# ╔═╡ 5daa28c7-7770-4375-a7a6-b656556e5da1
allμs(μ) = ntuple(i->μ,N)

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
dmusolve(Δ1,μ) = solve(hamiltonian!([t,Δ,V,θ,h,U,α,μbias,Δ1,allμs(μ)...]))

# ╔═╡ 5a0933d6-e098-4aa9-af4f-30676ef518f9
dmusolve(1.0,1.0)

# ╔═╡ 4a310f82-0ccc-4b5e-a684-71d4f037e83c
begin
		μs = -h .+ dμs
		iter = Base.product(Δ1s,μs);
end

# ╔═╡ 3578201e-24fd-45c6-9947-94fa62f2e618
@time gapratios, mpus = paramscan(dmusolve,iter);

# ╔═╡ 4cfbf14a-67d0-489d-ab5f-602c1a6712ad
begin
	res = bboptimize(Δ1μ-> cost_function(dmusolve(Δ1μ...)[1:2]...), map(sc->sum(sc)/2,[Δ1range,(μs[1],μs[end])]); SearchRange = [Δ1range,(μs[1],μs[end])], NumDimensions = 2, MaxTime)
	sweet_spot = best_candidate(res)
	sweet_gap,sweet_mpu = dmusolve(sweet_spot...)
end

# ╔═╡ 2ba2ece6-7a2e-4e4e-88a7-c0554b056de5
sweet_Δ1, sweet_μ = sweet_spot

# ╔═╡ 3651aee7-8f85-4ccf-a8ca-663c07afe380
begin
	#heatandcontour(Δ1s ./ Δ,μs,gapratios,mpus,(sweet_spot[1]/Δ,sweet_spot[2]))
	((sweet_gap,sweet_mpu))
	heatandcontour2(Δ1s ./ Δ,μs,gapratios,mpus,(sweet_Δ1/Δ,sweet_μ))
	title!(@sprintf("gap: %5.3f, MPu: %5.3f", sweet_gap,sweet_mpu))
end

# ╔═╡ 4a639756-3051-4702-a210-d8933ae331aa
energyjacobian(params) = jacobian(central_fdm(5, 1), first ∘ solve ∘ fasthamall!, params)[1] 

# ╔═╡ 40650975-7a7a-4eaa-86b1-824b2040f0d8
mpujacobian(params) = jacobian(central_fdm(5, 1), last ∘ solve ∘ fasthamall!, params)[1] 

# ╔═╡ 1cad4200-5efd-4966-940c-1f6da8ba3c27
sweet_spot_params = [t,Δ,V,θ,h,U,α,μbias,sweet_Δ1,allμs(sweet_μ)...]

# ╔═╡ a348c828-4289-48ec-9dc5-c629a8502565
sweet_spot_energy_jacobian = energyjacobian(sweet_spot_params)

# ╔═╡ ce68f40c-14cb-4df3-a489-09ea49291109
sweet_spot_mpu_jacobian = mpujacobian(sweet_spot_params)

# ╔═╡ ba2b48d2-d22d-42fe-b18f-46fe1babdb13
begin
	barplotxticks = xticks = (eachindex(parameternames),parameternames)
	barplote = bar(sweet_spot_energy_jacobian',xticks = barplotxticks, title= string("|δgapratio|=",@sprintf("%5.3f",norm(sweet_spot_energy_jacobian))))
	barplotmpu = bar(sweet_spot_mpu_jacobian',xticks = barplotxticks, title= string("|δMPu|=",@sprintf("%5.3f",norm(sweet_spot_mpu_jacobian))))
	plot(barplote,barplotmpu,size=(800,400),plot_title = "Jacobian")
end

# ╔═╡ 396c3061-00ad-4195-85cd-284204ea2242
dot(sweet_spot_mpu_jacobian,sweet_spot_energy_jacobian)

# ╔═╡ 6afdb8d7-5929-427f-a0a6-19dac33014b0
md"""
xparam: $(@bind xparam Select(parameternames,default = :μ1))
xparamresolution: $(@bind xparamresolution Slider(10:100,default=40,show_value=true))

xparamstart: $(@bind xparamstart Slider(range(-t,0,length=11),default=-t,show_value=true))
xparamend: $(@bind xparamend Slider(range(0,t,length=11),default=t,show_value=true))
"""

# ╔═╡ e1550793-c4e6-4e6f-b4b0-4ba3a38c17bf
xparamindex = findfirst(p->p==xparam,parameternames)

# ╔═╡ 6800302a-3cfb-474e-a4b1-4539bf96f975
function sweet_spot_perturbation(p0,xparamindex,xs)
	dp = [n == xparamindex ? 1.0 : 0.0 for n in 1:length(p0)]  
	data = map(x->solve(fasthamall!(p0 .+ dp*x),exactsolver), xs)
	gapratios = first.(data)
	mpus = last.(data)
	return gapratios, mpus
end

# ╔═╡ fb6a7184-a697-4027-9b3a-7b75f6e74e13
begin
	xparamvalues = range(xparamstart,xparamend,length=xparamresolution)
	sweet_spot_perturbation_data = sweet_spot_perturbation(sweet_spot_params,
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

# ╔═╡ 32566151-0960-4550-886f-e2588abba626
diff(sweet_spot_perturbation_data[1]) ./ diff(xparamvalues)

# ╔═╡ 087ebb84-3c33-43c1-a01d-a9f6fd11c2d2
plot(diff(diff(sweet_spot_perturbation_data[1])) ./ (diff(xparamvalues) .^2)[2:end])

# ╔═╡ 9451426f-221b-4118-8ad4-e3d61c3652e4
plot(diff(diff(diff(sweet_spot_perturbation_data[1]))) ./ (diff(xparamvalues) .^3)[3:end])

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

# ╔═╡ Cell order:
# ╠═a9a6cdf0-76db-11ed-0076-6dfc37097407
# ╠═bc605fb9-16f9-44a4-bb58-c471e42e9754
# ╠═33f235a6-0220-489f-97c8-40d8493048ca
# ╟─1fb9255b-da89-4358-b1e3-91adb691b377
# ╠═127abb1f-7850-402f-a3a9-d4a24ba20246
# ╟─12eaa1c6-1c66-4ebc-9577-31569a706fa0
# ╟─9ad954b1-a9c1-4c0c-a6a8-44d62d085f71
# ╟─3fb9c1b3-8fd8-40e8-a269-3fa547e35dbd
# ╟─dae7470c-51a2-4f8f-bab3-76bf8f552544
# ╟─c16cf91f-3627-4174-85bc-9dd66119a858
# ╠═1b94162b-e895-4827-8815-dcc5c0ede0e6
# ╟─ce1150f1-5ac0-4a99-b7e6-7bab919076d2
# ╠═1fcdf3e2-97a9-4b10-b23a-ef8e294e4b62
# ╟─b32d769f-d149-472a-9e3a-10005c2f7846
# ╟─b96e7112-bec6-45e5-b100-35c730d6249a
# ╟─1c332f07-a2d3-455e-8879-c1a0618b0dc9
# ╟─3983336d-5563-4d0f-82ce-82501072cb8a
# ╟─dac50315-9d43-401e-af05-759ce33d3a07
# ╟─ab5b3104-b0d8-43d8-a808-85599da7e4fa
# ╟─d2dbda16-5c9a-46b6-8027-0dac4b78731c
# ╟─c4da0cc9-1fc1-4ba1-9263-f9bd2707e32d
# ╠═cfb4c4cd-5513-41b1-834d-1f3b2b3a53ea
# ╟─cce1e924-891d-4735-b8ba-3d3408a9afdd
# ╠═3f430364-b749-45e2-a92c-58252e056193
# ╠═bbb6932e-1bb4-43dd-9f71-02d1edd9e380
# ╟─a081699f-df96-4d10-88e6-a0c7ea2dd67f
# ╟─fb4e6268-d918-42a1-b8be-2cc1dfcf285e
# ╟─a1af8874-b3e6-4f36-aac2-80a978a23128
# ╠═2b53130c-3ab3-4f43-8772-54a5b0b7b7f7
# ╠═564f1d21-8ea8-48e1-b30e-cee2936ad32d
# ╠═5a0933d6-e098-4aa9-af4f-30676ef518f9
# ╠═5daa28c7-7770-4375-a7a6-b656556e5da1
# ╠═85e124b0-c48d-4bf3-8e07-f04bcd9892f7
# ╟─4a310f82-0ccc-4b5e-a684-71d4f037e83c
# ╟─574d9dbf-bab1-4563-b2a7-898e6129d508
# ╠═0c0698fd-a9e0-4934-b290-7477306d3d50
# ╟─0674b1e4-4767-4b2e-99dc-abc2a5533af1
# ╟─f44f8dca-81f1-4cab-9b36-0d0cfdc4e381
# ╟─b25de208-0911-49e4-ae35-97af5b3449f8
# ╟─3651aee7-8f85-4ccf-a8ca-663c07afe380
# ╟─ba2b48d2-d22d-42fe-b18f-46fe1babdb13
# ╠═396c3061-00ad-4195-85cd-284204ea2242
# ╠═3578201e-24fd-45c6-9947-94fa62f2e618
# ╠═2ba2ece6-7a2e-4e4e-88a7-c0554b056de5
# ╠═4cfbf14a-67d0-489d-ab5f-602c1a6712ad
# ╠═9e504dd7-01dd-4f08-9577-92182c0c9487
# ╠═32566151-0960-4550-886f-e2588abba626
# ╠═087ebb84-3c33-43c1-a01d-a9f6fd11c2d2
# ╠═9451426f-221b-4118-8ad4-e3d61c3652e4
# ╠═4a639756-3051-4702-a210-d8933ae331aa
# ╠═40650975-7a7a-4eaa-86b1-824b2040f0d8
# ╠═a348c828-4289-48ec-9dc5-c629a8502565
# ╠═ce68f40c-14cb-4df3-a489-09ea49291109
# ╠═1cad4200-5efd-4966-940c-1f6da8ba3c27
# ╟─6afdb8d7-5929-427f-a0a6-19dac33014b0
# ╠═e1550793-c4e6-4e6f-b4b0-4ba3a38c17bf
# ╠═6800302a-3cfb-474e-a4b1-4539bf96f975
# ╠═fb6a7184-a697-4027-9b3a-7b75f6e74e13
# ╟─980acaa2-c82e-46e2-ad21-702fc1ee8743
