var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = QuantumDots","category":"page"},{"location":"#QuantumDots.jl-Documentation","page":"Home","title":"QuantumDots.jl Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for QuantumDots.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package provides some tools for working with quantum systems, especially interacting systems of fermions. It is not registered but can be installed by ","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg; Pkg.add(url=\"https://github.com/cvsvensson/QuantumDots.jl\")","category":"page"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Let's analyze a small fermionic system. We first define a basis","category":"page"},{"location":"","page":"Home","title":"Home","text":"using QuantumDots\nN = 2 # number of fermions\nspatial_labels = 1:N \ninternal_labels = (:↑,:↓)\nc = FermionBasis(spatial_labels, internal_labels)\n","category":"page"},{"location":"","page":"Home","title":"Home","text":"Indexing into the basis like returns sparse representations of the fermionic operators, so that one can write down Hamiltonians in a natural way","category":"page"},{"location":"","page":"Home","title":"Home","text":"H_hopping = c[1,:↑]'c[2,:↑] + c[1,:↓]'c[2,:↓] + hc \nH_coulomb = sum(c[n,:↑]'c[n,:↑]c[n,:↓]'c[n,:↓] for n in spatial_labels)\nH = H_hopping + H_coulomb\n#= 16×16 SparseArrays.SparseMatrixCSC{Int64, Int64} with 23 stored entries:\n⎡⠠⠂⠀⠀⠀⠀⠀⠀⎤\n⎢⠀⠀⠰⢂⠑⢄⠀⠀⎥\n⎢⠀⠀⠑⢄⠠⢆⠀⠀⎥\n⎣⠀⠀⠀⠀⠀⠀⠰⢆⎦ =#","category":"page"},{"location":"","page":"Home","title":"Home","text":"One can also work in the single particle basis FermionBdGBasis if the system is noninteracting. Quadratic functions of the fermionic operators produce the single particle BdG Hamiltonian.","category":"page"},{"location":"","page":"Home","title":"Home","text":"c2 = FermionBdGBasis(spatial_labels, internal_labels)\nHfree = c2[1,:↑]'c2[2,:↑] + c2[1,:↓]'c2[2,:↓] + hc\nvals, vecs = diagonalize(BdGMatrix(Hfree)) ","category":"page"},{"location":"","page":"Home","title":"Home","text":"Using diagonalize on a matrix of type BdGMatrix enforces particle-hole symmetry for the eigenvectors.","category":"page"},{"location":"#More-info","page":"Home","title":"More info","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"For a more in depth introduction see pmm_notebook.\nQubitBasis and time evolution is demonstrated in qubit_dephasing.\nSimulation of Majorana braiding with noisy gates is demonstrated in majorana_braiding.\nMost functionalities of the package are demonstrated in the tests.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [QuantumDots]","category":"page"},{"location":"#QuantumDots.AbelianFockSymmetry","page":"Home","title":"QuantumDots.AbelianFockSymmetry","text":"struct AbelianFockSymmetry{IF,FI,QN,QNfunc} <: AbstractSymmetry\n\nAbelianFockSymmetry represents a symmetry that is diagonal in fock space, i.e. particle number conservation, parity, spin consvervation.\n\nFields\n\nindtofockdict::IF: A dictionary mapping indices to Fock states.\nfocktoinddict::FI: A dictionary mapping Fock states to indices.\nqntoblocksizes::Dictionary{QN,Int}: A dictionary mapping quantum numbers to block sizes.\nqntofockstates::Dictionary{QN,Vector{Int}}: A dictionary mapping quantum numbers to Fock states.\nqntoinds::Dictionary{QN,Vector{Int}}: A dictionary mapping quantum numbers to indices.\nconserved_quantity::QNfunc: A function that computes the conserved quantity from a fock number.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumDots.BdGFermion","page":"Home","title":"QuantumDots.BdGFermion","text":"struct BdGFermion{S,B,T} <: AbstractBdGFermion\n\nThe BdGFermion struct represents a basis fermion for BdG matrices.\n\nFields\n\nid::S: The identifier of the fermion.\nbasis::B: The fermion basis.\namp::T: The amplitude of the fermion (default: true, i.e. 1).\nhole::Bool: Indicates whether the fermion is a hole (default: true).\n\n\n\n\n\n","category":"type"},{"location":"#QuantumDots.BdGMatrix","page":"Home","title":"QuantumDots.BdGMatrix","text":"struct BdGMatrix <: AbstractMatrix\n\nBdGMatrix represents a Bogoliubov-de Gennes (BdG) matrix, which is a block matrix used to describe non-interacting superconducting systems. It consists of four blocks: H, Δ, -conj(Δ), and -conj(H), where H is a Hermitian matrix and Δ is an antisymmetric matrix.\n\nFields\n\nH: The Hermitian block of the BdG matrix.\nΔ: The antisymmetric block of the BdG matrix.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumDots.DiagonalizedHamiltonian","page":"Home","title":"QuantumDots.DiagonalizedHamiltonian","text":"struct DiagonalizedHamiltonian{Vals,Vecs,H} <: AbstractDiagonalHamiltonian\n\nA struct representing a diagonalized Hamiltonian.\n\nFields\n\nvalues: The eigenvalues of the Hamiltonian.\nvectors: The eigenvectors of the Hamiltonian.\noriginal: The original Hamiltonian.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumDots.FermionBasis","page":"Home","title":"QuantumDots.FermionBasis","text":"struct FermionBasis{M,D,Sym,L} <: AbstractManyBodyBasis\n\nFermion basis for representing many-body fermions.\n\nFields\n\ndict::OrderedDict: A dictionary that maps fermion labels to a representation of the fermion.\nsymmetry::Sym: The symmetry of the basis.\njw::JordanWignerOrdering{L}: The Jordan-Wigner ordering of the basis.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumDots.KhatriRaoVectorizer","page":"Home","title":"QuantumDots.KhatriRaoVectorizer","text":"struct KhatriRaoVectorizer{T} <: AbstractVectorizer\n\nA struct representing a Khatri-Rao vectorizer. This vectorizer is used for BlockDiagonal density matrices matrices, where the superoperators respect the block structure.\n\nFields\n\nsizes::Vector{Int}: Vector of sizes.\nidvec::Vector{T}: Vector of identifiers.\ncumsum::Vector{Int}: Vector of cumulative sums.\ncumsumsquared::Vector{Int}: Vector of squared cumulative sums.\ninds::Vector{UnitRange{Int}}: Vector of index ranges.\nvectorinds::Vector{UnitRange{Int}}: Vector of vector index ranges.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumDots.KronVectorizer","page":"Home","title":"QuantumDots.KronVectorizer","text":"struct KronVectorizer{T} <: AbstractVectorizer\n\nA struct representing a KronVectorizer, the standard vectorizer where superoperators are formed from kronecker products of operators.\n\nFields\n\nsize::Int: The size of the KronVectorizer.\nidvec::Vector{T}: A vector representing the vectorized identity matrix. Saved here because it is useful when normalization is needed for computations in e.g. LindbladSystem.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumDots.LindbladCache","page":"Home","title":"QuantumDots.LindbladCache","text":"struct LindbladCache{KC, MC, SC, OC}\n\nA cache structure used in the Lindblad equation solver.\n\nFields\n\nkroncache::KC: Cache for Kronecker products.\nmulcache::MC: Cache for matrix multiplications.\nsuperopcache::SC: Cache for superoperators.\nopcache::OC: Cache for operators.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumDots.LindbladDissipator","page":"Home","title":"QuantumDots.LindbladDissipator","text":"struct LindbladDissipator{S,T,L,H,V,C} <: AbstractDissipator\n\nA struct representing a Lindblad dissipator.\n\nFields\n\nsuperop::S: The superoperator representing the dissipator.\nrate::T: The rate of the dissipator.\nlead::L: The lead associated with the dissipator.\nham::H: The Hamiltonian associated with the dissipator.\nvectorizer::V: The vectorizer used for vectorization.\ncache::C: The cache used for storing intermediate results.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumDots.LindbladSystem","page":"Home","title":"QuantumDots.LindbladSystem","text":"LindbladSystem(hamiltonian, leads, vectorizer=default_vectorizer(hamiltonian); rates=Dict(k => 1 for (k, v) in pairs(leads)), usecache=false)\n\nConstructs a Lindblad system for simulating open quantum systems.\n\nArguments\n\nhamiltonian: The Hamiltonian of the system.\nleads: An list of operators representing the leads.\nvectorizer: Determines how to vectorize the lindblad equation. Defaults to default_vectorizer(hamiltonian).\nrates: An array of rates for each lead. Defaults to an array of ones with the same length as leads.\nusecache: A boolean indicating whether to use a cache. Defaults to false.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumDots.LindbladSystem-2","page":"Home","title":"QuantumDots.LindbladSystem","text":"struct LindbladSystem{T,U,DS,V,H,C} <: AbstractOpenSystem\n\nA struct representing a Lindblad open quantum system.\n\nFields\n\ntotal::T: The total lindblad matrix operator.\nunitary::U: The unitary part of the system.\ndissipators::DS: The dissipators of the system.\nvectorizer::V: The vectorizer used for the system.\nhamiltonian::H: The Hamiltonian of the system.\ncache::C: The cache used for the system.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumDots.PauliDissipator-Union{Tuple{H}, Tuple{H, Any}} where H<:QuantumDots.DiagonalizedHamiltonian","page":"Home","title":"QuantumDots.PauliDissipator","text":"PauliDissipator(ham, lead; change_lead_basis=true)\n\nConstructs the Pauli dissipator for a given Hamiltonian and lead.\n\nArguments\n\nham: The Hamiltonian.\nlead: The leads.\nchange_lead_basis: A boolean indicating whether to change the lead basis to the energy basis. Default is true.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.PauliSystem-Tuple{Any, Any}","page":"Home","title":"QuantumDots.PauliSystem","text":"PauliSystem(ham, leads)\n\nConstructs a PauliSystem object from a Hamiltonian and a set of leads.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.QuasiParticle","page":"Home","title":"QuantumDots.QuasiParticle","text":"struct QuasiParticle{T,M,L} <: AbstractBdGFermion\n\nThe QuasiParticle struct represents a quasi-particle in the context of a BdG (Bogoliubov-de Gennes) fermion system. It is a linear combination of basis BdG fermions, and is defined by a set of weights and a basis.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumDots.QuasiParticle-Tuple{QuantumDots.BdGFermion}","page":"Home","title":"QuantumDots.QuasiParticle","text":"QuasiParticle(f::BdGFermion)\n\nConstructs a QuasiParticle object from a BdGFermion object.\n\n\n\n\n\n","category":"method"},{"location":"#Base.:*-Tuple{QuantumDots.BdGFermion, QuantumDots.BdGFermion}","page":"Home","title":"Base.:*","text":"*(f1::BdGFermion, f2::BdGFermion; symmetrize=true)\n\nMultiply two BdGFermion objects f1 and f2. By default, it symmetrizes the result, returning a BdG matrix in the convention used here.\n\n\n\n\n\n","category":"method"},{"location":"#Base.:*-Tuple{QuantumDots.QuasiParticle, QuantumDots.QuasiParticle}","page":"Home","title":"Base.:*","text":"*(f1::QuasiParticle, f2::QuasiParticle; kwargs...)\n\nReturn the BdG matrix of the product of quasiparticles f1 and f2.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.bdg_to_skew-Tuple{BdGMatrix}","page":"Home","title":"QuantumDots.bdg_to_skew","text":"bdg_to_skew(B::BdGMatrix; check=true)\n\nConvert a BdGMatrix to a skew-Hermitian matrix. If check is true, it checks that the result is skew-Hermitian.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.blockdiagonal-Tuple{AbstractMatrix, QuantumDots.AbstractManyBodyBasis}","page":"Home","title":"QuantumDots.blockdiagonal","text":"blockdiagonal(m::AbstractMatrix, basis::AbstractManyBodyBasis)\n\nConstruct a BlockDiagonal version of m using the symmetry of basis. No checking is done to ensure this is a faithful representation.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.dissipator-Tuple{Any, Any, QuantumDots.KronVectorizer}","page":"Home","title":"QuantumDots.dissipator","text":"dissipator(L, rate, kv::KronVectorizer)\n\nConstructs the dissipator associated to the jump operator L.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.embedding_unitary-Tuple{Any, Any, JordanWignerOrdering}","page":"Home","title":"QuantumDots.embedding_unitary","text":"embedding_unitary(partition, fockstates, jw)\n\nCompute the unitary matrix that maps between the tensor embedding and the fermionic embedding in the physical subspace. \n# Arguments\n- `partition`: A partition of the labels in `jw` into disjoint sets.\n- `fockstates`: The fock states in the basis\n- `jw`: The Jordan-Wigner ordering.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.eval_in_basis-Tuple{QuantumDots.FermionSym, QuantumDots.AbstractBasis}","page":"Home","title":"QuantumDots.eval_in_basis","text":"eval_in_basis(a, f::AbstractBasis)\n\nEvaluate an expression with fermions in a basis f. \n\nExamples\n\n@fermions a\nf = FermionBasis(1:2)\nQuantumDots.eval_in_basis(a[1]'*a[2] + hc, f)\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.fermion_sparse_matrix-Tuple{Any, Any, Any}","page":"Home","title":"QuantumDots.fermion_sparse_matrix","text":"fermion_sparse_matrix(fermion_number, totalsize, sym)\n\nConstructs a sparse matrix of size totalsize representing a fermionic operator at bit position fermion_number in a many-body fermionic system with symmetry sym. \n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.fermionic_embedding-Tuple{Any, Any, Any}","page":"Home","title":"QuantumDots.fermionic_embedding","text":"fermionic_embedding(m, b, bnew)\n\nCompute the fermionic embedding of a matrix m in the basis b into the basis bnew.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.fockstates-Tuple{Any, Any}","page":"Home","title":"QuantumDots.fockstates","text":"fockstates(M, n)\n\nGenerate a list of Fock states with n occupied fermions in a system with M different fermions.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.focksymmetry-Tuple{Any, Any}","page":"Home","title":"QuantumDots.focksymmetry","text":"focksymmetry(fockstates, qn)\n\nConstructs an AbelianFockSymmetry object that represents the symmetry of a many-body fermionic system. \n\nArguments\n\nfockstates: The fockstates to iterate over\nqn: A function that takes an integer representing a fock state and returns corresponding quantum number.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.jwstring-Tuple{Any, Any}","page":"Home","title":"QuantumDots.jwstring","text":"jwstring(site, focknbr)\n\nParity of the number of fermions to the right of site.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.majorana_coefficients","page":"Home","title":"QuantumDots.majorana_coefficients","text":"majorana_coefficients(f::QuasiParticle, labels=collect(keys((basis(f))))\n\nCompute the Majorana coefficients for a given QuasiParticle object f. Returns two dictionaries, for the two types of Majorana operators.\n\n\n\n\n\n","category":"function"},{"location":"#QuantumDots.many_body_density_matrix","page":"Home","title":"QuantumDots.many_body_density_matrix","text":"many_body_density_matrix(G, c=FermionBasis(1:div(size(G, 1), 2), qn=parity); alg=SkewEigenAlg())\n\nCompute the many-body density matrix for a given correlator G. The traceless version of G should be a BdGMatrix. \n\nSee also one_particle_density_matrix, many_body_hamiltonian.\n\n\n\n\n\n","category":"function"},{"location":"#QuantumDots.many_body_fermion-Tuple{QuantumDots.BdGFermion, FermionBasis}","page":"Home","title":"QuantumDots.many_body_fermion","text":"many_body_fermion(f::BdGFermion, basis::FermionBasis)\n\nReturn the representation of f in the many-body fermion basis basis.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.many_body_hamiltonian","page":"Home","title":"QuantumDots.many_body_hamiltonian","text":"many_body_hamiltonian(H::AbstractMatrix, Δ::AbstractMatrix, c::FermionBasis=FermionBasis(1:size(H, 1), qn=parity))\n\nConstruct the many-body Hamiltonian for a given BdG Hamiltonian consisting of hoppings H and pairings Δ.\n\n\n\n\n\n","category":"function"},{"location":"#QuantumDots.normalized_steady_state_rhs-Tuple{Any}","page":"Home","title":"QuantumDots.normalized_steady_state_rhs","text":"normalized_steady_state_rhs(A)\n\nFor a linear operator A, whose last row represents the normalization condition, return the rhs of the equation A * x = b where x is the normalized steady-state and b the vector with all zeros except for the last element which is 1.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.numberoperator-Tuple{FermionBasis}","page":"Home","title":"QuantumDots.numberoperator","text":"numberoperator(basis::FermionBasis)\n\nReturn the number operator of basis.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.one_particle_density_matrix-Tuple{AbstractVector{<:QuantumDots.QuasiParticle}}","page":"Home","title":"QuantumDots.one_particle_density_matrix","text":"one_particle_density_matrix(χs::AbstractVector{<:QuasiParticle})\n\nReturn the oneparticledensity_matrix for the state with all χs occupied as its ground state.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.one_particle_density_matrix-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T","page":"Home","title":"QuantumDots.one_particle_density_matrix","text":"one_particle_density_matrix(U::AbstractMatrix{T}) where {T}\n\nCompute the one-particle density matrix for the vacuum of a BdG system diagonalized by U.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.one_particle_density_matrix-Union{Tuple{QuantumDots.QuasiParticle{T}}, Tuple{T}} where T","page":"Home","title":"QuantumDots.one_particle_density_matrix","text":"one_particle_density_matrix(χ::QuasiParticle{T})\n\nReturn the one_particle_density_matrix for the state with χ occupied as its ground state.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.one_particle_density_matrix-Union{Tuple{T}, Tuple{AbstractMatrix{T}, FermionBasis}, Tuple{AbstractMatrix{T}, FermionBasis, Any}} where T","page":"Home","title":"QuantumDots.one_particle_density_matrix","text":"one_particle_density_matrix(ρ::AbstractMatrix, b::FermionBasis, labels=keys(b))\n\nCompute the one-particle density matrix for a given density matrix ρ in the many body fermion basis b.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.ordered_prod_of_embeddings-Tuple{Any, Any, Any}","page":"Home","title":"QuantumDots.ordered_prod_of_embeddings","text":"ordered_prod_of_embeddings(ms, bs, b)\n\nCompute the ordered product of the fermionic embeddings of the matrices ms in the bases bs into the basis b.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.parityoperator-Tuple{QuantumDots.AbstractBasis}","page":"Home","title":"QuantumDots.parityoperator","text":"parityoperator(basis::AbstractBasis)\n\nReturn the parity operator of basis.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.partial_trace!-Union{Tuple{M}, Tuple{T}, Tuple{Any, AbstractMatrix{T}, Any, FermionBasis{M}}, Tuple{Any, AbstractMatrix{T}, Any, FermionBasis{M}, QuantumDots.AbstractSymmetry}} where {T, M}","page":"Home","title":"QuantumDots.partial_trace!","text":"partial_trace!(mout, m::AbstractMatrix, labels, b::FermionBasis, sym::AbstractSymmetry=NoSymmetry())\n\nCompute the fermionic partial trace of a matrix m in basis b, leaving only the subsystems specified by labels. The result is stored in mout, and sym determines the ordering of the basis states.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.partial_trace-Tuple{AbstractMatrix, QuantumDots.AbstractBasis, QuantumDots.AbstractBasis}","page":"Home","title":"QuantumDots.partial_trace","text":"partial_trace(v::AbstractMatrix, bsub::AbstractBasis, bfull::AbstractBasis)\n\nCompute the partial trace of a matrix v, leaving the subsystem defined by the basis bsub.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.partial_trace-Union{Tuple{T}, Tuple{AbstractMatrix{T}, Any, QuantumDots.AbstractBasis}, Tuple{AbstractMatrix{T}, Any, QuantumDots.AbstractBasis, QuantumDots.AbstractSymmetry}} where T","page":"Home","title":"QuantumDots.partial_trace","text":"partial_trace(m::AbstractMatrix, labels, b::FermionBasis, sym::AbstractSymmetry=NoSymmetry())\n\nCompute the partial trace of a matrix m in basis b, leaving only the subsystems specified by labels. sym determines the ordering of the basis states.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.partial_transpose-Tuple{AbstractMatrix, Any, FermionBasis}","page":"Home","title":"QuantumDots.partial_transpose","text":"partial_transpose(m::AbstractMatrix, labels, b::FermionBasis{M})\n\nCompute the fermionic partial transpose of a matrix m in subsystem denoted by labels.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.quasiparticle_adjoint-Tuple{AbstractVector}","page":"Home","title":"QuantumDots.quasiparticle_adjoint","text":"quasiparticle_adjoint(v::AbstractVector)\n\nCompute the adjoint of a quasiparticle represented by the weights in v. The adjoint is computed by swapping the hole and particle parts of the vector and taking the complex conjugate of each element.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.ratetransform-Tuple{Any, AbstractVector, Any, Any}","page":"Home","title":"QuantumDots.ratetransform","text":"ratetransform(op,  energies::AbstractVector, T, μ)\n\nTransform op in the energy basis with a Fermi-Dirac distribution at temperature T and chemical potential μ.\n\nArguments\n\nop: The operator to be transformed.\nenergies::AbstractVector: The energies.\nT: The temperature.\nμ: The chemical potential.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.ratetransform-Tuple{Any, QuantumDots.DiagonalizedHamiltonian, Any, Any}","page":"Home","title":"QuantumDots.ratetransform","text":"ratetransform(op, diagham::DiagonalizedHamiltonian, T, μ)\n\nTransform op with a Fermi-Dirac distribution at temperature T and chemical potential μ.\n\nArguments\n\nop: The operator to be transformed.\ndiagham::DiagonalizedHamiltonian: The diagonalized Hamiltonian.\nT: The temperature.\nμ: The chemical potential.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.removefermion-Tuple{Any, FockNumber}","page":"Home","title":"QuantumDots.removefermion","text":"removefermion(digitposition, statefocknbr)\n\nReturn (newfocknbr, fermionstatistics) where newfocknbr is the state obtained by removing a fermion at digitposition from statefocknbr and fermionstatistics is the phase from the Jordan-Wigner string, or 0 if the operation is not allowed.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.rep-Tuple{QuantumDots.BdGFermion}","page":"Home","title":"QuantumDots.rep","text":"rep(f::BdGFermion)\n\nConstructs a sparse vector representation of a BdGFermion object.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.rotate_majorana_coefficients-Tuple{Any, Any, Any}","page":"Home","title":"QuantumDots.rotate_majorana_coefficients","text":"rotate_majorana_coefficients(w, z, phase)\n\nRotate the Majorana coefficients by `phase`\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.skew_to_bdg-Tuple{AbstractMatrix}","page":"Home","title":"QuantumDots.skew_to_bdg","text":"skew_to_bdg(A::AbstractMatrix)\n\nConvert a skew-symmetric matrix A to a BdGMatrix.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.skew_to_bdg-Tuple{AbstractVector}","page":"Home","title":"QuantumDots.skew_to_bdg","text":"skew_to_bdg(v::AbstractVector)\n\nUse the same transformation that transforms a skew-symmetric matrix to a BdGMatrix to transform a vector v.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.superoperator-NTuple{6, Any}","page":"Home","title":"QuantumDots.superoperator","text":"superoperator(lead_op, diagham, T, μ, rate, vectorizer)\n\nConstruct the superoperator associated with the operator lead_op. Transforms the operator to the energy basis and includes fermi-Dirac statistics.\n\nArguments\n\nlead_op: The operator representing the lead coupling.\ndiagham: The diagonal Hamiltonian.\nT: The temperature.\nμ: The chemical potential.\nrate: The rate of the dissipative process.\nvectorizer: The vectorizer struct.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.tensor-Union{Tuple{T}, Tuple{AbstractVector{T}, QuantumDots.AbstractBasis}} where T","page":"Home","title":"QuantumDots.tensor","text":"tensor(v::AbstractVector, b::AbstractBasis)\n\nReturn a tensor representation of the vector v in the basis b, with one index for each site.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.wedge","page":"Home","title":"QuantumDots.wedge","text":"wedge(ms::AbstractVector, bs::AbstractVector{<:FermionBasis}, b::FermionBasis=wedge(bs))\n\nCompute the wedge product of matrices or vectors in ms with respect to the fermion bases bs, respectively. Return a matrix in the fermion basis b, which defaults to the wedge product of bs.\n\n\n\n\n\n","category":"function"},{"location":"#QuantumDots.wedge-Tuple{AbstractVector{<:FermionBasis}}","page":"Home","title":"QuantumDots.wedge","text":"wedge(bs)\n\nCompute the wedge product of a list of FermionBasis objects. The symmetry of the resulting basis is computed by promote_symmetry.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumDots.@fermions-Tuple","page":"Home","title":"QuantumDots.@fermions","text":"@fermions a b ...\n\nCreate one or more fermion species with the given names. Indexing into fermions species gives a concrete fermion. Fermions in one @fermions block anticommute with each other,  and commute with fermions in other @fermions blocks.\n\nExamples:\n\n@fermions a b creates two species of fermions that anticommute:\na[1]' * a[1] + a[1] * a[1]' == 1\na[1]' * b[1] + b[1] * a[1]' == 0\n@fermions a; @fermions b creates two species of fermions that commute with each other:\na[1]' * a[1] + a[1] * a[1]' == 1\na[1] * b[1] - b[1] * a[1] == 0\n\nSee also @majoranas, QuantumDots.eval_in_basis.\n\n\n\n\n\n","category":"macro"},{"location":"#QuantumDots.@majoranas-Tuple","page":"Home","title":"QuantumDots.@majoranas","text":"@majoranas a b ...\n\nCreate one or more Majorana species with the given names. Indexing into Majorana species gives a concrete Majorana. Majoranas in one @majoranas block anticommute with each other, and commute with Majoranas in other @majoranas blocks.\n\nExamples:\n\n@majoranas a b creates two species of Majoranas that anticommute:\na[1] * a[1] + a[1] * a[1] == 1\na[1] * b[1] + b[1] * a[1] == 0\n@majoranas a; @majoranas b creates two species of Majoranas that commute with each other:\na[1] * a[1] + a[1] * a[1] == 1\na[1] * b[1] - b[1] * a[1] == 0\n\nSee also @fermions, QuantumDots.eval_in_basis.\n\n\n\n\n\n","category":"macro"}]
}
