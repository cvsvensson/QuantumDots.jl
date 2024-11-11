
struct SymbolicFermionBasis
    name::Symbol
end

macro fermion(x)
    :($(esc(x)) = SymbolicFermionBasis($(Expr(:quote, x))))
end
Base.getindex(f::SymbolicFermionBasis, is...) = FermionSym(false, is, f.name)
Base.getindex(f::SymbolicFermionBasis, i) = FermionSym(false, i, f.name)
struct FermionSym{L}
    creation::Bool
    label::L
    name::Symbol
end
Base.adjoint(x::FermionSym) = FermionSym(!x.creation, x.label, x.name)
Base.iszero(x::FermionSym) = false
function Base.show(io::IO, x::FermionSym)
    print(io, x.name, x.creation ? "†" : "")
    if Base.isiterable(typeof(x.label))
        Base.show_delim_array(io, x.label, "[", ",", "]", false)
    else
        print(io, "[", x.label, "]")
    end
end
function Base.isless(a::FermionSym, b::FermionSym)
    if a.creation == b.creation
        a.label < b.label
    else
        a.creation > b.creation
    end
end
Base.:(==)(a::FermionSym, b::FermionSym) = a.creation == b.creation && a.label == b.label && a.name == b.name
Base.hash(a::FermionSym, h::UInt) = hash(hash(a.creation, hash(a.label, hash(a.name, h))))

struct FermionMul{C,F<:FermionSym}
    coeff::C
    factors::Vector{F}
    function FermionMul(coeff::C, factors) where {C}
        if iszero(coeff)
            0
        elseif length(factors) == 0
            coeff
        else
            if !issorted(factors) || !sorted_noduplicates(factors)
                throw(ArgumentError("Factors must be sorted"))
            end
            new{C,eltype(factors)}(coeff, factors)
        end
    end
end
function Base.show(io::IO, x::FermionMul)
    print_coeff = !isone(x.coeff)
    if print_coeff
        print(io, x.coeff)
    end
    for (n, x) in enumerate(x.factors)
        if print_coeff || n > 1
            print(io, "*")
        end
        print(io, x)
    end
end
Base.iszero(x::FermionMul) = iszero(x.coeff)

Base.:(==)(a::FermionMul, b::FermionMul) = a.coeff == b.coeff && a.factors == b.factors
Base.:(==)(a::FermionMul, b::FermionSym) = isone(a.coeff) && length(a.factors) == 1 && only(a.factors) == b
Base.:(==)(b::FermionSym, a::FermionMul) = a == b
Base.hash(a::FermionMul, h::UInt) = hash(hash(a.coeff, hash(a.factors, h)))
FermionMul(f::FermionMul) = f
FermionMul(f::FermionSym) = FermionMul(1, [f])

struct FermionAdd{C,D}
    coeff::C
    dict::D
    function FermionAdd(coeff::C, dict::D) where {C,D}
        if length(dict) == 0
            coeff
        elseif length(dict) == 1 && iszero(coeff)
            k, v = first(dict)
            v * k
        else
            all(isone(_coeff(k)) for (k, v) in dict)
            new{C,D}(coeff, dict)
        end
    end
end
Base.:(==)(a::FermionAdd, b::FermionAdd) = a.coeff == b.coeff && a.dict == b.dict
const SM = Union{FermionSym,FermionMul}
const SMA = Union{FermionSym,FermionMul,FermionAdd}

function Base.show(io::IO, x::FermionAdd)
    print_one = !iszero(x.coeff)
    if print_one
        print(io, x.coeff, "I")
    end
    print_sign(v) = sign(v) == 1 ? print(io, " + ") : print(io, " - ")
    for (n, (k, v)) in enumerate(collect(pairs(x.dict)))
        print_sign(v)
        print(io, abs(v) * k)
    end
end

Base.:+(a::Number, b::SM) = iszero(a) ? b : FermionAdd(a, to_add(b))
Base.:+(a::SM, b::Number) = b + a
Base.:+(a::SM, b::SM) = FermionAdd(0, (_merge(+, to_add(a), to_add(b); filter=iszero)))
Base.:+(a::SM, b::FermionAdd) = FermionAdd(b.coeff, (_merge(+, b.dict, to_add(a); filter=iszero)))
Base.:+(a::FermionAdd, b::SM) = b + a

to_add(a::FermionMul, coeff=1) = Dict(FermionMul(1, a.factors) => a.coeff * coeff)
to_add(a::FermionSym, coeff=1) = Dict(FermionMul(a) => coeff)

Base.:+(a::Number, b::FermionAdd) = iszero(a) ? b : FermionAdd(a + b.coeff, b.dict)
Base.:+(a::FermionAdd, b::Number) = b + a
Base.:-(a::Number, b::SMA) = a + (-b)
Base.:-(a::SMA, b::Number) = a + (-b)
Base.:-(a::SMA, b::SMA) = a + (-b)
Base.:-(a::SMA) = -1 * a
function terms(a::FermionAdd)
    [v * k for (k, v) in pairs(a.dict)]
end
function Base.:+(a::FermionAdd, b::FermionAdd)
    a.coeff + foldr((f, b) -> f + b, terms(a); init=b)
end

function Base.:^(a::FermionSym, b)
    if b isa Number && iszero(b)
        1
    elseif b isa Number && b == 1
        a
    elseif b isa Integer && b >= 2
        0
    else
        throw(ArgumentError("Invalid exponent $b"))
    end
end
function Base.:^(a::FermionMul, x::Integer)
    prod(a for _ in 1:x)
end
_coeff(a::FermionSym) = 1
_coeff(a::FermionMul) = a.coeff
Base.:*(x::Number, a::FermionSym) = iszero(x) ? 0 : FermionMul(x, [a])
Base.:*(x::Number, a::FermionMul) = iszero(x) ? 0 : FermionMul(x * a.coeff, a.factors)
Base.:*(x::Number, a::FermionAdd) = iszero(x) ? 0 : FermionAdd(x * a.coeff, Dict(k => v * x for (k, v) in collect(a.dict)))
Base.:*(a::SMA, x::Number) = x * a

function Base.:*(a::FermionSym, b::FermionSym)
    if a == b
        0
    elseif a < b
        FermionMul(1, [a, b])
    elseif a > b
        FermionMul(-1, [b, a]) + Int(a.label == b.label)
    end
end
function sorted_noduplicates(v)
    I = eachindex(v)
    for i in I[1:end-1]
        v[i] == v[i+1] && return false
    end
    return true
end
function Base.:*(a::FermionSym, bs::FermionMul)
    coeff = bs.coeff
    ind = searchsortedfirst(bs.factors, a)
    inds = searchsorted(bs.factors, a)
    inds_conj = searchsorted(bs.factors, a')
    annihilation = if (length(inds) > 0 && length(inds_conj) > 0 && inds_conj[1] < inds[1])
        FermionMul(coeff * (-1)^(inds_conj[1] + 1),
            [bs.factors[1:inds_conj[1]-1]..., bs.factors[inds_conj[1]+1:end]...])
    else
        0
    end
    insertion = if length(inds) == 0
        FermionMul(coeff * (-1)^(ind + 1),
            [bs.factors[1:ind-1]..., a, bs.factors[ind:end]...])
    else
        0
    end
    return annihilation + insertion
end
Base.:*(as::FermionMul, b::FermionSym) = (b' * as')'
Base.:*(as::FermionMul, bs::FermionMul) = as.coeff * foldr(*, as.factors; init=bs)

Base.adjoint(x::FermionMul) = adjoint(x.coeff) * foldr(*, reverse(adjoint.(x.factors)))
Base.:*(a::FermionAdd, b::SM) = (b' * a')'
function Base.:*(a::SM, b::FermionAdd)
    a * b.coeff + sum(a * f for f in terms(b))
end
function Base.:*(a::FermionAdd, b::FermionAdd)
    a.coeff * b + sum(f * b for f in terms(a))
end

Base.adjoint(x::FermionAdd) = FermionAdd(adjoint(x.coeff), Dict(adjoint(f) => c for (f, c) in collect(x.dict)))

#From SymbolicUtils
_merge(f::F, d, others...; filter=x -> false) where {F} = _merge!(f, Dict{SM,Any}(d), others...; filter=filter)

function _merge!(f::F, d, others...; filter=x -> false) where {F}
    acc = d
    for other in others
        for (k, v) in other
            v = f(v)
            ak = get(acc, k, nothing)
            if ak !== nothing
                v = ak + v
            end
            if filter(v)
                delete!(acc, k)
            else
                acc[k] = v
            end
        end
    end
    acc
end

@testitem "SymbolicFermions" begin
    @fermion f
    f1 = f[:a]
    f2 = f[:b]
    f3 = f[1, :↑]
    @test_nowarn display(f1)
    @test_nowarn display(f3)
    @test_nowarn display(1 * f1)
    @test_nowarn display(2 * f3)
    @test_nowarn display(1 + f1)
    @test_nowarn display(1 + f3)
    @test iszero(f1 - f1)
    @test iszero(f1 * f1)
    @test f1 * f2 isa QuantumDots.FermionMul
    @test iszero(2 * f1 - 2 * f1)
    @test iszero(0 * f1)
    @test 2 * f1 isa QuantumDots.FermionMul
    @test iszero(f1 * 0)
    @test iszero(f1^2)
    @test iszero(0 * (f1 + f2))
    @test iszero((f1 + f2) * 0)
    @test iszero(f1 * f2 * f1)
    f12 = f1 * f2
    @test iszero(f12'' - f12)
    @test iszero(f12 * f12)
    @test iszero(f12' * f12')
    nf1 = f1' * f1
    @test nf1^2 == nf1
    @test f1' * f1 isa QuantumDots.FermionMul
    @test f1 * f1' isa QuantumDots.FermionAdd

    @test 1 + (f1 + f2) == 1 + f1 + f2 == f1 + f2 + 1 == f1 + 1 + f2 == 1 * f1 + f2 + 1 == f1 + 0.5 * f2 + 1 + (0 * f1 + 0.5 * f2) == (0.5 + 0.5 * f1 + 0.2 * f2) + 0.5 + (0.5 * f1 + 0.8 * f2) == (1 + f1' + (1 * f2)')'
    @test iszero((2 * f1) * (2 * f1))
    @test iszero((2 * f1)^2)
    @test (2 * f2) * (2 * f1) == -4 * f1 * f2
    @test f1 == (f1 * (f1 + 1)) == (f1 + 1) * f1
    @test iszero(f1 * (f1 + f2) * f1)
    @test (f1 * (f1 + f2)) == f1 * f2
    @test (2nf1 - 1) * (2nf1 - 1) == 1

    @test (1 * f1) * f2 == f1 * f2
    @test (1 * f1) * (1 * f2) == f1 * f2
    @test f1 * f2 == f1 * (1 * f2) == f1 * f2
    @test f1 - 1 == (1 * f1) - 1 == (0.5 + f1) - 1.5
end

_labels(a::FermionMul) = [s.label for s in a.factors]
SparseArrays.sparse(op::Union{<:FermionAdd,<:FermionMul,<:FermionAdd,<:FermionSym}, labels, instates::AbstractVector) = sparse(op, labels, instates, instates)
SparseArrays.sparse(op::Union{<:FermionMul,<:FermionSym}, labels, outstates, instates::AbstractVector) = sparse(sparsetuple(op, labels, outstates, instates)..., length(outstates), length(instates))
function sparsetuple(op::FermionMul{C}, labels, outstates, instates; fock_to_outind=Dict(map(p -> Pair(reverse(p)...), enumerate(outstates)))) where {C}
    outfocks = Int[]
    ininds_final = Int[]
    amps = C[]
    sizehint!(outfocks, length(instates))
    sizehint!(ininds_final, length(instates))
    sizehint!(amps, length(instates))
    digitpositions = reverse(siteindices(_labels(op), labels))
    daggers = reverse([s.creation for s in op.factors])
    for (n, f) in enumerate(instates)
        newfockstate, amp = togglefermions(digitpositions, daggers, f)
        if !iszero(amp)
            push!(outfocks, newfockstate)
            push!(amps, amp * op.coeff)
            push!(ininds_final, n)
        end
    end
    indsout = map(i -> fock_to_outind[i], outfocks)
    return (indsout, ininds_final, amps)
end
function SparseArrays.sparse(op::FermionAdd, labels, outstates, instates::AbstractVector)
    fock_to_outind = Dict(map(p -> Pair(reverse(p)...), enumerate(outstates)))
    tuples = [sparsetuple(op, labels, outstates, instates; fock_to_outind) for op in terms(op)]
    indsout = mapreduce(Base.Fix2(Base.getindex, 1), vcat, tuples)
    indsin_final = mapreduce(Base.Fix2(Base.getindex, 2), vcat, tuples)
    amps = mapreduce(Base.Fix2(Base.getindex, 3), vcat, tuples)
    return op.coeff * I + sparse(indsout, indsin_final, amps, length(outstates), length(instates))

end
sparsetuple(op::FermionSym, labels, outstates, instates) = sparsetuple(FermionMul(1, [op]), labels, outstates, instates)

@testitem "SparseFermion" begin
    using SparseArrays, LinearAlgebra
    @fermion f
    N = 4
    labels = 1:N
    fmb = FermionBasis(labels)
    get_mat(op) = sparse(op, labels, 0:2^N-1, 0:2^N-1)
    @test all(get_mat(f[l]) == fmb[l] for l in labels)
    @test all(get_mat(f[l]') == fmb[l]' for l in labels)
    @test all(get_mat(f[l]') == get_mat(f[l])' for l in labels)
    @test all(get_mat(f[l]'') == get_mat(f[l]) for l in labels)
    @test all(get_mat(f[l]' * f[l]) == get_mat(f[l])' * get_mat(f[l]) for l in labels)
    @test all(get_mat(f[l]' * f[l]) == fmb[l]' * fmb[l] for l in labels)

    newmat = get_mat(sum(f[l]' * f[l] for l in labels))
    mat = sum(fmb[l]' * fmb[l] for l in labels)
    @test newmat == mat

    @test all(sparse(sum(f[l]' * f[l] for l in labels), labels, QuantumDots.fockstates(N, n)) == n * I for n in 1:N)
end