abstract type NonCommutative end
import SymbolicUtils: SymbolicUtils, Term, isterm, isadd, makeadd, add_t, Add, Symbolic, BasicSymbolic, SN, Sym, mul_t, makemul, Mul, promote_symtype, symtype
# Base.:*(a::NonCommutative, b::NonCommutative) = Term(*, [a, b])
# ## and all combinations with numbers:
# Base.:*(a::Number, b::NonCommutative) = Term(*, [a, b])
# Base.:*(a::NonCommutative, b::Number) = 2*a
# Base.:+(a::Number, b::NonCommutative) = Term(+, [a, b])
# Base.:+(a::NonCommutative, b::Number) = Term(+, [a, b])
# Base.:-(a::Number, b::NonCommutative) = Term(-, [a, b])
# Base.:-(a::NonCommutative, b::Number) = Term(-, [a, b])
# Base.:-(a::NonCommutative) = Term(-, [a])

const NC = Symbolic{<:NonCommutative}
Base.:*(a::NC) = a
Base.:*(a::NC, b::Number) = b * a
struct NCMul{T<:NonCommutative} <: Symbolic{T}
    coeff
    factors::Vector{Any}
end
isncmul(a::NCMul) = true
isncmul(::Any) = false
Base.isone(a::Union{<:NC,<:NonCommutative}) = false
Base.iszero(a::Union{<:NC,<:NonCommutative}) = false
Base.iszero(a::NCMul) = iszero(a.coeff)
function Base.show(io::IO, x::NCMul)
    print_coeff = !isequal(x.coeff, one(x.coeff))
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
Base.literal_pow(::typeof(^), x::FermionSym, ::Val{p}) where {p} = ^(x, p)
function Base.:*(a::Union{<:NC,<:Number,<:SN}, b::NC)
    if iszero(a)
        return a
    elseif isone(a)
        return b
    elseif isncmul(a) && isncmul(b)
        return NCMul{mul_t(a, b)}(a.coeff * b.coeff, vcat(a.factors, b.factors))
    end
    coeff, factors = makeNCmul(1, a, b)
    NCMul{mul_t(a, b)}(coeff, factors)
end
# Term{}(::typeof(+), args; metadata) = (println("ASD"); +(args...))

function SymbolicUtils.basicsymbolic(f, args, stype::Type{<:Union{<:NonCommutative,<:NC}}, metadata)
    f(args...)
    # if f isa Symbol
    #     error("$f must not be a Symbol")
    # end
    # T = stype
    # if T === nothing
    #     T = SymbolicUtils._promote_symtype(f, args)
    # end
    # if T <: SymbolicUtils.LiteralReal
    #     @goto FALLBACK
    # elseif all(x -> symtype(x) <: Number, args)
    #     if f === (+)
    #         res = +(args...)
    #         res
    #     elseif f == (*)
    #         res = *(args...)
    #         res
    #     else
    #         @goto FALLBACK
    #     end
    # else
    #     @label FALLBACK
    #     Term{T}(f, args, metadata=metadata)
    # end
end

function makeNCmul(coeff, xs...; d=Any[])
    for x in xs
        if x isa Union{Number,SN}
            coeff *= x
        elseif isncmul(x)
            coeff *= x.coeff
            push!(d, x.factors...)
        else
            push!(d, x)
        end
    end
    (coeff, d)
end
TermInterface.children(a::NCMul) = [a.coeff, a.factors...]
TermInterface.arguments(a::NCMul) = children(a)
TermInterface.head(a::NCMul) = (*)
TermInterface.iscall(::NCMul) = true
TermInterface.isexpr(::NCMul) = true
TermInterface.operation(::NCMul) = (*)
SymbolicUtils.metadata(::NCMul) = nothing
TermInterface.maketerm(::Type{<:NCMul}, f::typeof(*), args, metadata) = *(args...)
Base.:(==)(a::NCMul, b::NCMul) = a.coeff == b.coeff && a.factors == b.factors

isFermion(x) = x isa FermionSym
NormalOrder = SymbolicUtils.@rule (~x) * (~y) => if ~x < ~y
    nothing
elseif ~x == ~y
    return 0
else
    NCMul((-1)^((~x).basis.universe == (~y).basis.universe), [(~y), (~x)]) + Int((~x).label == (~y).label && (~x).basis == (~y).basis)
end
NormalOrder2 = SymbolicUtils.@rule (~x) * (~y) => if ~x < ~y
    nothing
elseif ~x == ~y
    return 0
else
    NCMul((-1)^((~x).basis.universe == (~y).basis.universe), [(~y), (~x)]) + Int((~x).label == (~y).label && (~x).basis == (~y).basis)
end

# promote_symtype(::Union{typeof(+),typeof(*),typeof(-)}, ::Type{<:NC}, ::Type{<:NC}) = NC
# promote_symtype(::Union{typeof(+),typeof(*),typeof(-)}, ::Type{<:Number}, ::Type{<:NC}) = NC
# promote_symtype(::Union{typeof(+),typeof(*),typeof(-)}, ::Type{<:NC}, ::Type{<:Number}) = NC
# promote_symtype(::Union{typeof(+),typeof(*),typeof(-),typeof(identity)}, ::Type{<:NC}) = NC
promote_symtype(::Union{typeof(+),typeof(*),typeof(-)}, ::Type{<:NonCommutative}, ::Type{<:NonCommutative}) = NonCommutative
promote_symtype(::Union{typeof(+),typeof(*),typeof(-)}, ::Type{<:Number}, ::Type{<:NonCommutative}) = NonCommutative
promote_symtype(::Union{typeof(+),typeof(*),typeof(-)}, ::Type{<:NonCommutative}, ::Type{<:Number}) = NonCommutative
promote_symtype(::Union{typeof(+),typeof(*),typeof(-),typeof(identity)}, ::Type{<:NonCommutative}) = NonCommutative

Base.:+(a::NC, b) = b + a
function Base.:+(a::Union{NC,SN,Number}, b::NC)
    if isadd(a) && isadd(b)
        return Add(add_t(a, b),
            a.coeff + b.coeff,
            _merge(+, a.dict, b.dict, filter=_iszero))
    elseif isadd(a)
        coeff, dict = makeadd(1, 0, b)
        return Add(add_t(a, b), a.coeff + coeff, _merge(+, a.dict, dict, filter=_iszero))
    elseif isadd(b)
        return b + a
    end
    coeff, dict = makeadd(1, 0, a, b)
    Add(add_t(a, b), coeff, dict)
end
# TermInterface.maketerm(::NonCommutative, f::typeof(*), args, metadata) = *(args...)

@testitem "Term" begin
    using Symbolics
    @variables x y
    @fermions f
    f[1] * f[2]


end

# Base.:+(a::BasicSymbolic, b::NonCommutative) =
#     isterm(a) || isadd(a) ? Term(+, [a.arguments, 1]) :

## Instantiating sparse matrices
# _labels(a::FermionMul) = [s.label for s in a.factors]
# SparseArrays.sparse(op::Union{<:FermionAdd,<:FermionMul,<:FermionAdd,<:NonCommutative}, labels, instates::AbstractVector) = sparse(op, labels, instates, instates)
# SparseArrays.sparse(op::Union{<:FermionMul,<:NonCommutative}, labels, outstates, instates::AbstractVector) = sparse(sparsetuple(op, labels, outstates, instates)..., length(outstates), length(instates))
# function sparsetuple(op::FermionMul{C}, labels, outstates, instates; fock_to_outind=Dict(map(p -> Pair(reverse(p)...), enumerate(outstates)))) where {C}
#     outfocks = Int[]
#     ininds_final = Int[]
#     amps = C[]
#     sizehint!(outfocks, length(instates))
#     sizehint!(ininds_final, length(instates))
#     sizehint!(amps, length(instates))
#     digitpositions = reverse(siteindices(_labels(op), labels))
#     daggers = reverse([s.creation for s in op.factors])
#     for (n, f) in enumerate(instates)
#         newfockstate, amp = togglefermions(digitpositions, daggers, f)
#         if !iszero(amp)
#             push!(outfocks, newfockstate)
#             push!(amps, amp * op.coeff)
#             push!(ininds_final, n)
#         end
#     end
#     indsout = map(i -> fock_to_outind[i], outfocks)
#     return (indsout, ininds_final, amps)
# end
# function SparseArrays.sparse(op::FermionAdd, labels, outstates, instates::AbstractVector)
#     fock_to_outind = Dict(map(p -> Pair(reverse(p)...), enumerate(outstates)))
#     tuples = [sparsetuple(op, labels, outstates, instates; fock_to_outind) for op in fermionterms(op)]
#     indsout = mapreduce(Base.Fix2(Base.getindex, 1), vcat, tuples)
#     indsin_final = mapreduce(Base.Fix2(Base.getindex, 2), vcat, tuples)
#     amps = mapreduce(Base.Fix2(Base.getindex, 3), vcat, tuples)
#     return op.coeff * I + sparse(indsout, indsin_final, amps, length(outstates), length(instates))

# end
# sparsetuple(op::NonCommutative, labels, outstates, instates) = sparsetuple(FermionMul(1, [op]), labels, outstates, instates)

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
# eval_in_basis(a::FermionMul, f::AbstractBasis) = a.coeff * mapfoldl(Base.Fix2(eval_in_basis, f), *, a.factors)
# eval_in_basis(a::FermionAdd, f::AbstractBasis) = a.coeff * I + mapfoldl(Base.Fix2(eval_in_basis, f), +, fermionterms(a))

##
# TermInterface.head(a::Union{FermionMul,FermionAdd}) = operation(a)
# TermInterface.iscall(::Union{FermionMul,FermionAdd}) = true
# TermInterface.isexpr(::Union{FermionMul,FermionAdd}) = true
# TermInterface.head(::NonCommutative) = :ref
# TermInterface.iscall(::NonCommutative) = false
# TermInterface.isexpr(::NonCommutative) = true

# TermInterface.operation(::FermionMul) = (*)
# TermInterface.operation(::FermionAdd) = (+)
# TermInterface.arguments(a::FermionMul) = [a.coeff, a.factors...]
# TermInterface.arguments(a::FermionAdd) = iszero(a.coeff) ? fermionterms(a) : allterms(a)
# TermInterface.sorted_arguments(a::FermionAdd) = iszero(a.coeff) ? sort(fermionterms(a), by=x -> x.factors) : [a.coeff, sort(fermionterms(a); by=x -> x.factors)...]
# TermInterface.children(a::Union{FermionMul,FermionAdd}) = arguments(a)
# TermInterface.sorted_children(a::Union{FermionMul,FermionAdd}) = sorted_arguments(a)


#From SymbolicUtils
# _merge(f::F, d, others...; filter=x -> false) where {F} = _merge!(f, Dict{SM,Any}(d), others...; filter=filter)

# function _merge!(f::F, d, others...; filter=x -> false) where {F}
#     acc = d
#     for other in others
#         for (k, v) in other
#             v = f(v)
#             ak = get(acc, k, nothing)
#             if ak !== nothing
#                 v = ak + v
#             end
#             if filter(v)
#                 delete!(acc, k)
#             else
#                 acc[k] = v
#             end
#         end
#     end
#     acc
# end
