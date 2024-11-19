abstract type AbstractFermionSym end
Base.:*(a::AbstractFermionSym, b::AbstractFermionSym) = ordered_prod(a, b)
unordered_prod(a::AbstractFermionSym, b::AbstractFermionSym) = FermionMul(1, [a, b])

struct FermionMul{C,F<:AbstractFermionSym}
    coeff::C
    factors::Vector{F}
    ordered::Bool
    function FermionMul(coeff::C, factors) where {C}
        if iszero(coeff)
            0
        elseif length(factors) == 0
            coeff
        else
            ordered = issorted(factors) && sorted_noduplicates(factors)
            new{C,eltype(factors)}(coeff, factors, ordered)
        end
    end
end
function Base.show(io::IO, x::FermionMul)
    print_coeff = !isone(x.coeff)
    if print_coeff
        v = x.coeff
        if isreal(v)
            neg = v < 0
            if neg isa Bool
                if neg
                    print(io, -real(v))
                else
                    print(io, real(v))
                end
            else
                print(io, "(", v, ")")
            end
        else
            print(io, "(", v, ")")
        end
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
Base.:(==)(a::FermionMul, b::AbstractFermionSym) = isone(a.coeff) && length(a.factors) == 1 && only(a.factors) == b
Base.:(==)(b::AbstractFermionSym, a::FermionMul) = a == b
Base.hash(a::FermionMul, h::UInt) = hash(hash(a.coeff, hash(a.factors, h)))
FermionMul(f::FermionMul) = f
FermionMul(f::AbstractFermionSym) = FermionMul(1, [f])
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
const SM = Union{AbstractFermionSym,FermionMul}
const SMA = Union{AbstractFermionSym,FermionMul,FermionAdd}

function Base.show(io::IO, x::FermionAdd)
    compact = get(io, :compact, false)
    args = sorted_arguments(x)
    print_one = !iszero(x.coeff)
    if print_one
        if isreal(x.coeff)
            print(io, real(x.coeff), "I")
        else
            print(io, "(", x.coeff, ")", "I")
        end
        args = args[2:end]
    end
    print_sign(s) = compact ? print(io, s) : print(io, " ", s, " ")
    for (n, arg) in enumerate(args)
        k = prod(arg.factors)
        v = arg.coeff
        if isreal(v)
            neg = v < 0
            if neg isa Bool
                if neg
                    print_sign("-")
                    print(io, -real(v) * k)
                else
                    print_sign("+")
                    print(io, real(v) * k)
                end
            else
                print_sign("+")
                print(io, "(", v, ")*", k)
            end
        else
            print_sign("+")
            print(io, "(", v, ")*", k)
        end
    end
end
print_num(io::IO, x) = isreal(x) ? print(io, real(x)) : print(io, "(", x, ")")

Base.:+(a::Number, b::SM) = iszero(a) ? b : FermionAdd(a, to_add(b))
Base.:+(a::SM, b::Number) = b + a
Base.:+(a::SM, b::SM) = FermionAdd(0, (_merge(+, to_add(a), to_add(b); filter=iszero)))
Base.:+(a::SM, b::FermionAdd) = FermionAdd(b.coeff, (_merge(+, b.dict, to_add(a); filter=iszero)))
Base.:+(a::FermionAdd, b::SM) = b + a
Base.:/(a::SMA, b::Number) = inv(b) * a
to_add(a::FermionMul, coeff=1) = Dict(FermionMul(1, a.factors) => a.coeff * coeff)
to_add(a::AbstractFermionSym, coeff=1) = Dict(FermionMul(a) => coeff)

Base.:+(a::Number, b::FermionAdd) = iszero(a) ? b : FermionAdd(a + b.coeff, b.dict)
Base.:+(a::FermionAdd, b::Number) = b + a
Base.:-(a::Number, b::SMA) = a + (-b)
Base.:-(a::SMA, b::Number) = a + (-b)
Base.:-(a::SMA, b::SMA) = a + (-b)
Base.:-(a::SMA) = -1 * a
function fermionterms(a::FermionAdd)
    [v * k for (k, v) in pairs(a.dict)]
end
function allterms(a::FermionAdd)
    [a.coeff, [v * k for (k, v) in pairs(a.dict)]...]
end
function Base.:+(a::FermionAdd, b::FermionAdd)
    a.coeff + foldr((f, b) -> f + b, fermionterms(a); init=b)
end
Base.:^(a::Union{FermionMul,FermionAdd}, b) = Base.power_by_squaring(a, b)


_coeff(a::FermionMul) = a.coeff
Base.:*(x::Number, a::AbstractFermionSym) = iszero(x) ? 0 : FermionMul(x, [a])
Base.:*(x::Number, a::FermionMul) = iszero(x) ? 0 : FermionMul(x * a.coeff, a.factors)
Base.:*(x::Number, a::FermionAdd) = iszero(x) ? 0 : FermionAdd(x * a.coeff, Dict(k => v * x for (k, v) in collect(a.dict)))
Base.:*(a::SMA, x::Number) = x * a

Base.:*(a::AbstractFermionSym, bs::FermionMul) = (1 * a) * bs
Base.:*(as::FermionMul, b::AbstractFermionSym) = (b' * as')'
Base.:*(as::FermionMul, bs::FermionMul) = order_mul(unordered_prod(as, bs))
Base.adjoint(x::FermionMul) = adjoint(x.coeff) * foldr(*, reverse(adjoint.(x.factors)))
Base.:*(a::FermionAdd, b::SM) = (b' * a')'
function Base.:*(a::SM, b::FermionAdd)
    a * b.coeff + sum(a * f for f in fermionterms(b))
end
function Base.:*(a::FermionAdd, b::FermionAdd)
    a.coeff * b + sum(f * b for f in fermionterms(a))
end

Base.adjoint(x::FermionAdd) = adjoint(x.coeff) + sum(f' for f in fermionterms(x))


unordered_prod(a::FermionMul, b::FermionAdd) = b.coeff * a + sum(unordered_prod(a, f) for f in fermionterms(b))
unordered_prod(a::FermionAdd, b::FermionMul) = a.coeff * b + sum(unordered_prod(f, b) for f in fermionterms(a))
unordered_prod(a::FermionAdd, b::FermionAdd) = sum(unordered_prod(f, g) for f in allterms(a), g in allterms(b))
unordered_prod(a::FermionMul, b::FermionMul) = FermionMul(a.coeff * b.coeff, [a.factors..., b.factors...])
unordered_prod(a, b, xs...) = foldl(*, xs; init=(*)(a, b))
unordered_prod(x::Number, a::SMA) = x * a
unordered_prod(a::SMA, x::Number) = x * a
unordered_prod(x::Number, y::Number) = x * y

function sorted_noduplicates(v)
    I = eachindex(v)
    for i in I[1:end-1]
        v[i] == v[i+1] && return false
    end
    return true
end

## Normal ordering
ordering_product(ordered_leftmul::Number, right_mul) = ordered_leftmul * order_mul(right_mul)
bubble_sort(a::FermionAdd) = a.coeff + sum(bubble_sort(f) for f in fermionterms(a))

function bubble_sort(a::FermionMul)
    if a.ordered || length(a.factors) == 1
        return a
    end
    swapped = true
    muloraddvec::Union{Number,SMA} = a

    swapped = false
    i = first(eachindex(a.factors)) - 1
    while !swapped && i < length(eachindex(a.factors)) - 1
        i += 1
        if a.factors[i] > a.factors[i+1] || (a.factors[i] == a.factors[i+1])
            swapped = true
            product = a.factors[i] * a.factors[i+1]
            left_factors = FermionMul(a.coeff, a.factors[1:i-1])
            right_factors = FermionMul(1, a.factors[i+2:end])
            muloraddvec = unordered_prod(left_factors, product, right_factors)
        end
    end
    bubble_sort(muloraddvec)
end
bubble_sort(a::Number) = a

order_mul(a::FermionMul) = bubble_sort(a)
order_mul(x::Number) = x


## Instantiating sparse matrices
_labels(a::FermionMul) = [s.label for s in a.factors]
SparseArrays.sparse(op::Union{<:FermionAdd,<:FermionMul,<:FermionAdd,<:AbstractFermionSym}, labels, instates::AbstractVector) = sparse(op, labels, instates, instates)
SparseArrays.sparse(op::Union{<:FermionMul,<:AbstractFermionSym}, labels, outstates, instates::AbstractVector) = sparse(sparsetuple(op, labels, outstates, instates)..., length(outstates), length(instates))
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
    tuples = [sparsetuple(op, labels, outstates, instates; fock_to_outind) for op in fermionterms(op)]
    indsout = mapreduce(Base.Fix2(Base.getindex, 1), vcat, tuples)
    indsin_final = mapreduce(Base.Fix2(Base.getindex, 2), vcat, tuples)
    amps = mapreduce(Base.Fix2(Base.getindex, 3), vcat, tuples)
    return op.coeff * I + sparse(indsout, indsin_final, amps, length(outstates), length(instates))

end
sparsetuple(op::AbstractFermionSym, labels, outstates, instates) = sparsetuple(FermionMul(1, [op]), labels, outstates, instates)

@testitem "Instantiating symbolic fermions" begin
    using SparseArrays, LinearAlgebra
    @fermions f
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

    @test all(QuantumDots.eval_in_basis(f[l], fmb) == fmb[l] for l in labels)
    @test all(QuantumDots.eval_in_basis(f[l]', fmb) == fmb[l]' for l in labels)
    @test all(QuantumDots.eval_in_basis(f[l]' * f[l], fmb) == fmb[l]'fmb[l] for l in labels)
    @test all(QuantumDots.eval_in_basis(f[l] + f[l]', fmb) == fmb[l] + fmb[l]' for l in labels)
end

## Convert to expression
eval_in_basis(a::FermionMul, f::AbstractBasis) = a.coeff * mapfoldl(Base.Fix2(eval_in_basis, f), *, a.factors)
eval_in_basis(a::FermionAdd, f::AbstractBasis) = a.coeff * I + mapfoldl(Base.Fix2(eval_in_basis, f), +, fermionterms(a))

##

TermInterface.head(::FermionMul) = :call
TermInterface.head(::FermionAdd) = :call
TermInterface.head(::FermionSym) = :ref
TermInterface.iscall(::FermionMul) = true
TermInterface.iscall(::FermionAdd) = true
TermInterface.iscall(::FermionSym) = false
TermInterface.isexpr(::FermionMul) = true
TermInterface.isexpr(::FermionAdd) = true
TermInterface.isexpr(::FermionSym) = true

TermInterface.operation(::FermionMul) = (*)
TermInterface.operation(::FermionAdd) = (+)
TermInterface.arguments(a::FermionMul) = [a.coeff, a.factors...]
TermInterface.arguments(a::FermionAdd) = iszero(a.coeff) ? fermionterms(a) : allterms(a)
TermInterface.sorted_arguments(a::FermionAdd) = iszero(a.coeff) ? sort(fermionterms(a), by=x -> x.factors) : [a.coeff, sort(fermionterms(a); by=x -> x.factors)...]
TermInterface.children(a::FermionMul) = arguments(a)
TermInterface.children(a::FermionAdd) = arguments(a)
TermInterface.sorted_children(a::FermionMul) = sorted_arguments(a)
TermInterface.sorted_children(a::FermionAdd) = sorted_arguments(a)


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
