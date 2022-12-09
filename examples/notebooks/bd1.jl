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
	using QuantumDots, LinearAlgebra, Random, ProgressMeter, BlackBoxOptim, Folds, PlutoUI, Plots, Printf, KrylovKit, SparseArrays
end

# ╔═╡ 127abb1f-7850-402f-a3a9-d4a24ba20246
gapratio(es) = real(diff(es)[1][1]/(abs(es[1][2] -es[2][1]) + abs(es[1][2] -es[2][1]))/2)

# ╔═╡ 1fb9255b-da89-4358-b1e3-91adb691b377
let
	Random.seed!(1234)
	BLAS.set_num_threads(1)
	cost_function(es,mpu::Number) = 10^3*gapratio(es)^2 + (1-abs(mpu))^2
end

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
	function BD1_hamqd(basis; μ, h, Δ1, t, α, Δ, U, V, θ=0.0, bias=0.0)
	    N = div(length(particles(basis)),2)
	    dbias =  bias*((1:N) .- ceil(N/2))/N
	    αnew = cos(θ/2)*α + sin(θ/2)*t
	    tnew = cos(θ/2)*t - sin(θ/2)*α
	    Δk = Δ1*sin(θ/2)
	    Δ1 = Δ1*cos(θ/2)
	    t = tnew
	    α = αnew
	    H = QuantumDots.FockOperatorSum(Float64[],[],basis,basis)
	    for j in 1:(N-1)
	        H += _BD1_ham_2site(basis,j;t,α,Δ1,Δk,V)
	        # ampo .+= _BD1_ham_2site(j;t,α,Δ1,Δk,V)
	    end
	    for j in 1:N
	        H += _BD1_ham_1site(basis,j;μ = μ+dbias[j],h,Δ,U)
	    end
	    return H
	end
	function BD1_hamqd_dis(basis; μs, h, Δ1, t, α, Δ, U, V, θ=0.0, bias=0.0)
	    N = div(length(particles(basis)),2)
	    dbias =  bias*((1:N) .- ceil(N/2))/N
	    αnew = cos(θ/2)*α + sin(θ/2)*t
	    tnew = cos(θ/2)*t - sin(θ/2)*α
	    Δk = Δ1*sin(θ/2)
	    Δ1 = Δ1*cos(θ/2)
	    t = tnew
	    α = αnew
	    H = QuantumDots.FockOperatorSum(Float64[],[],basis,basis)
	    for j in 1:(N-1)
	        H += _BD1_ham_2site(basis,j;t,α,Δ1,Δk,V)
	        # ampo .+= _BD1_ham_2site(j;t,α,Δ1,Δk,V)
	    end
	    for j in 1:N
	        H += _BD1_ham_1site(basis,j;μ = μs[j]+dbias[j],h,Δ,U)
	    end
	    return H
	end
end

# ╔═╡ ce1150f1-5ac0-4a99-b7e6-7bab919076d2
t=1.0

# ╔═╡ 12eaa1c6-1c66-4ebc-9577-31569a706fa0
function MPu(coeffs)
    n = Int(floor(size(coeffs, 1) / 2))
    c2 = real.(coeffs[1:n, :] .^ 2)
    sum(c2[:, 1] .+ c2[:, 2])
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
    data = Folds.map(calc_advance,iter)
	gapratios = first.(data)
	mpus = last.(data)
    return gapratios, mpus
end

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

# ╔═╡ 9ad954b1-a9c1-4c0c-a6a8-44d62d085f71
cost_function(gapratio,mpu::Number) = 10^3*(gapratio)^2 + (1-abs(mpu))^2

# ╔═╡ 0c8bbcfa-8dad-4471-8fb0-b3fcdf9ea294
begin
		#md"""
		#$(@bind sweet_Δ1 (NumberField(1:1)))
		#$(@bind sweet_μ (NumberField(1:1)))
		#"""
	sweet_spot = [1.0,1.0]
end

# ╔═╡ 0674b1e4-4767-4b2e-99dc-abc2a5533af1
md"""
N: $(@bind N Slider(2:4,default=2,show_value=true))
"""

# ╔═╡ c16cf91f-3627-4174-85bc-9dd66119a858
begin
	basis = FermionBasis(N, (:↑,:↓))
	a = particles(basis)
	majps = [sparse(basis*(a[i,s]' + a[i,s])*basis) for (i,s) in Base.product(1:N,(:↑,:↓))]
	majms = [1im*sparse(basis*(a[i,s]' - a[i,s])*basis) for (i,s) in Base.product(1:N,(:↑,:↓))]
	parity = sparse(basis*ParityOperator()*basis)
	hams = [(spzeros(Float64,4^N,4^N),spzeros(Float64,4^N,4^N)) for _ in 1:Threads.nthreads()]
    μsyms = ntuple(i->Symbol(:μ,i),N)
    #generator(Δ,V,θ,h,U,α,bias,Δ1,μs...) = Matrix(BD1_hamqd_dis(basis;μs,t,Δ,V,θ,h,U,α,Δ1,bias))
	generator(Δ,V,θ,h,U,α,bias,Δ1,μs...) = QuantumDots.spBlockDiagonal(BD1_hamqd_dis(basis;μs,t,Δ,V,θ,h,U,α,Δ1,bias))
    _, _oddham! = QuantumDots.generate_fastham(first ∘ generator,:Δ,:V,:θ,:h,:U,:α,:bias,:Δ1,μsyms...)
    _, _evenham! = QuantumDots.generate_fastham(last ∘ generator,:Δ,:V,:θ,:h,:U,:α,:bias,:Δ1,μsyms...)
    function fasthamall!(params)
		#_fastham!(hams[Threads.threadid()],params)
		_oddham!(hams[Threads.threadid()][1],params)
		_evenham!(hams[Threads.threadid()][2],params)
		hams[Threads.threadid()]
	end
end

# ╔═╡ b96e7112-bec6-45e5-b100-35c730d6249a
begin
	v0 = rand(4^N)
	tol = 1e-8
end

# ╔═╡ 3f430364-b749-45e2-a92c-58252e056193
begin
	function solve(H)
	    vals, vecs = eigen!(Hermitian(H))
	    eveninds = findall(v->(v'*parity*v)>.99, eachcol(vecs))
	    oddinds = setdiff(1:size(H,1),eveninds)
	    es = [vals[eveninds], vals[oddinds]]
	    gsodd = @view vecs[:,first(oddinds)]
	    gseven =  @view vecs[:,first(eveninds)]
	    ws = [gsodd'*op*gseven for op in majps]
	    vs = [gsodd'*op*gseven for op in majms]
	    majcoeffs = [ws;; vs]
	    mpu = MPu(majcoeffs)
		gapratio(es), mpu
	    #Dict("es"=> es, "MPu"=> mpu, "majcoeffs"=> majcoeffs)
	end
	function solve(Hodd,Heven)
	    evenvals, evenvecs = eigsolve(Hermitian(Hodd),v0,1,:LM,tol,issymmetric=true,ishermitian=true)
	    oddvals, oddvecs = eigsolve(Hermitian(Heven),v0,1,:LM,tol,issymmetric=true,ishermitian=true)
	    ws = [oddvecs[1]'*op*evenvecs[1] for op in majps]
	    vs = [oddvecs[1]'*op*evenvecs[1] for op in majms]
	    majcoeffs = [ws;; vs]
	    mpu = MPu(majcoeffs)
		gapratio(es), mpu
	    #Dict("es"=> es, "MPu"=> mpu, "majcoeffs"=> majcoeffs)
	end
end

# ╔═╡ f44f8dca-81f1-4cab-9b36-0d0cfdc4e381
md"""
Δ1start: $(@bind Δ1start Slider(range(-4t,0,length=10),default=-2t,show_value=true))
Δ1end: $(@bind Δ1end Slider(range(0,4t,length=10),default=2t,show_value=true))

μstart: $(@bind μstart Slider(range(-4t,0,length=10),default=-2t,show_value=true))
μend: $(@bind μend Slider(range(0,4t,length=10),default=2t,show_value=true))

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

h: $(@bind h Slider(range(0,50t,length=51),default=40t,show_value=true))
θ: $(@bind θ Slider(range(0,pi,length=21),default=pi/2,show_value=true))

Δ: $(@bind Δ Slider(range(0.1,4t,length=21),default=2t,show_value=true))

U: $(@bind U Slider(range(0,20t,length=21),default=10t,show_value=true))

μbias: $(@bind μbias Slider(range(0,2t,length=20),default=0,show_value=true))
MaxTime: $(@bind MaxTime Slider(range(0,10,length=101),default=0.2,show_value=true))
"""

# ╔═╡ 2b53130c-3ab3-4f43-8772-54a5b0b7b7f7
fastham!(Δ1,μs...) = fasthamall!([Δ,V,θ,h,U,α,μbias,Δ1,μs...])

# ╔═╡ 1307f3d8-d9ee-41f2-b854-7ea07302fa9e
calc_data(Δ1,μ) = solve(fastham!(Δ1, ntuple(i->μ,N)...)...)

# ╔═╡ 4a310f82-0ccc-4b5e-a684-71d4f037e83c
begin
		μs = -h .+ dμs
		iter = Base.product(Δ1s,μs);
end

# ╔═╡ 6bffd8fc-2c45-4a64-b6dd-47772051a3e8
map(sc->sum(sc)/2,[Δ1range,(μs[1],μs[end])])

# ╔═╡ 4cfbf14a-67d0-489d-ab5f-602c1a6712ad
begin
	res = bboptimize(Δ1μ-> cost_function(calc_data(Δ1μ...)...), map(sc->sum(sc)/2,[Δ1range,(μs[1],μs[end])]); SearchRange = [Δ1range,(μs[1],μs[end])], NumDimensions = 2, MaxTime)
	sweet_spot[1], sweet_spot[2] = best_candidate(res)
	sweet_gap,sweet_mpu = calc_data(sweet_spot...)
end

# ╔═╡ 3651aee7-8f85-4ccf-a8ca-663c07afe380
begin
	@time gapratios, mpus = paramscan(calc_data,iter);
	display((sweet_gap,sweet_mpu))
	heatandcontour(Δ1s ./ Δ,μs,gapratios,mpus,(sweet_spot[1]/Δ,sweet_spot[2]))
end

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
# ╟─1fb9255b-da89-4358-b1e3-91adb691b377
# ╟─127abb1f-7850-402f-a3a9-d4a24ba20246
# ╟─3fb9c1b3-8fd8-40e8-a269-3fa547e35dbd
# ╠═c16cf91f-3627-4174-85bc-9dd66119a858
# ╟─ce1150f1-5ac0-4a99-b7e6-7bab919076d2
# ╟─12eaa1c6-1c66-4ebc-9577-31569a706fa0
# ╟─b96e7112-bec6-45e5-b100-35c730d6249a
# ╟─3f430364-b749-45e2-a92c-58252e056193
# ╟─fb4e6268-d918-42a1-b8be-2cc1dfcf285e
# ╟─a1af8874-b3e6-4f36-aac2-80a978a23128
# ╠═2b53130c-3ab3-4f43-8772-54a5b0b7b7f7
# ╠═1307f3d8-d9ee-41f2-b854-7ea07302fa9e
# ╟─4a310f82-0ccc-4b5e-a684-71d4f037e83c
# ╟─574d9dbf-bab1-4563-b2a7-898e6129d508
# ╟─9ad954b1-a9c1-4c0c-a6a8-44d62d085f71
# ╟─0c8bbcfa-8dad-4471-8fb0-b3fcdf9ea294
# ╟─6bffd8fc-2c45-4a64-b6dd-47772051a3e8
# ╟─0674b1e4-4767-4b2e-99dc-abc2a5533af1
# ╟─f44f8dca-81f1-4cab-9b36-0d0cfdc4e381
# ╟─b25de208-0911-49e4-ae35-97af5b3449f8
# ╠═3651aee7-8f85-4ccf-a8ca-663c07afe380
# ╟─4cfbf14a-67d0-489d-ab5f-602c1a6712ad
# ╟─980acaa2-c82e-46e2-ad21-702fc1ee8743
