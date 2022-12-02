#This function compares standard matrix mult vs the bitwise strategy. 
using BenchmarkTools,SparseArrays
function timetest(N)
    basis = FermionBasis(N,:a)
    ψ = rand(State,basis,Float64)
    op = QuantumDots.LinearMap(FermionCreationOperator((:a,1),basis))
    op = sum([op,2.0op])
    op = sum([op + QuantumDots.LinearMap(CreationOperator(s,basis)) for s in QuantumDots.particles(basis)])
    op2 = sum([CreationOperator(s,basis) for s in QuantumDots.particles(basis)])
    println(typeof(op))
    @time M = Matrix(op)
    @time Ms = sparse(op)
    v = vec(ψ)
    println("Fock")
    _time(op,ψ)
    println("Fock2")
    _time(op2,ψ)
    println("Sparse")
    _time(Ms,v)
    println("Dense")
    _time(M,v)
end

function _time(M,v)
    @time out1 = M*v
    return
end