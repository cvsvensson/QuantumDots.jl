round_float(num::LinearAlgebra.BlasFloat; digits) = round(num; digits)
# for printing unroundable values
round_float(num; digits) = num

function pretty_print(v::AbstractVector, b::AbstractBasis; digits=3)
    N = length(b)
    printstyled("labels = |", bold=true)
    for (n, k) in enumerate(keys(b))
        printstyled(k, bold=true)
        n < length(keys(b)) && print(",")
    end
    printstyled(">", bold=true)
    println()
    for (n, qn) in enumerate(qns(b))
        print("QN = ", qn)
        println()
        for ind in qntoinds(qn, b)
            fs = indtofock(ind, b)
            print(" |", Int.(bits(fs, N))..., ">")
            println(" : ", round_float(v[ind]; digits))
        end
    end
end

import AxisKeys
import Crayons

struct ColoredString{C}
    s::String
    c::C
    ColoredString(s, c::C) where {C} = new{C}(s, c)
end

AxisKeys.ShowWith(val::ColoredString; hide=false, kw...) = AxisKeys.ShowWith(val.s; hide, kw..., color=val.c)
qn_colors(qns) = 1:length(qns)
function pretty_print(m::AbstractMatrix, b::AbstractManyBodyBasis)
    N = length(b)
    printstyled("labels = |", Tuple(keys(b))..., ">", bold=true)
    println()
    colors = qn_colors(qns(b))
    printstyled("QNs = [", bold=true)
    for (n, qn) in enumerate(qns(b))
        n == 1 || print(" ")
        printstyled(string(qn,), color=colors[n], bold=true)
        printstyled(",", color=colors[n], bold=true)
    end
    printstyled("]", bold=true)
    println()
    k2 = vcat([[ColoredString(string("|", Int.(bits(indtofock(ind, b), N))..., ">"), colors[n]) for ind in qntoinds(qn, b)]
               for (n, qn) in enumerate(qns(b))]...)
    ka = AxisKeys.KeyedArray(m, row=k2, col=k2)
    display(ka)
end

qns(b::FermionBasis) = qns(b.symmetry)
qns(::NoSymmetry) = (nothing,)
qns(sym::AbelianFockSymmetry) = Tuple(keys(sym.qntoinds))

qntoinds(qn, b::Union{FermionBasis,QubitBasis}) = qntoinds(qn, symmetry(b))
qntoinds(qn, sym::AbelianFockSymmetry) = sym.qntoinds[qn]
qntoinds(::Nothing, ::QubitBasis{M,<:Any,NoSymmetry}) where {M} = 1:2^M
qntoinds(::Nothing, ::FermionBasis{M,<:Any,NoSymmetry}) where {M} = 1:2^M
