round_float(num::LinearAlgebra.BlasFloat; digits) = round(num; digits)
# for printing unroundable values
round_float(num; digits) = num

function pretty_print(v::AbstractVector, H::AbstractFockHilbertSpace; digits=3)
    N = length(H.jw)
    printstyled("labels = |", bold=true)
    for (n, k) in enumerate(keys(H))
        printstyled(k, bold=true)
        n < length(keys(H)) && print(",")
    end
    printstyled(">", bold=true)
    println()
    for (n, qn) in enumerate(qns(H))
        print("QN = ", qn)
        println()
        for ind in qntoinds(qn, H)
            fs = indtofock(ind, H)
            print(" |", Int.(bits(fs, N))..., ">")
            println(" : ", round_float(v[ind]; digits))
        end
    end
end


struct ColoredString{C}
    s::String
    c::C
    ColoredString(s, c::C) where {C} = new{C}(s, c)
end

AxisKeys.ShowWith(val::ColoredString; hide=false, kw...) = AxisKeys.ShowWith(val.s; hide, kw..., color=val.c)
qn_colors(qns) = 1:length(qns)
function pretty_print(m::AbstractMatrix, H::AbstractFockHilbertSpace)
    N = length(H.jw)
    printstyled("labels = |", Tuple(keys(H))..., ">", bold=true)
    println()
    colors = qn_colors(qns(H))
    printstyled("QNs = [", bold=true)
    for (n, qn) in enumerate(qns(H))
        n == 1 || print(" ")
        printstyled(string(qn,), color=colors[n], bold=true)
        printstyled(",", color=colors[n], bold=true)
    end
    printstyled("]", bold=true)
    println()
    k2 = vcat([[ColoredString(string("|", Int.(bits(indtofock(ind, H), N))..., ">"), colors[n]) for ind in qntoinds(qn, H)]
               for (n, qn) in enumerate(qns(H))]...)
    ka = AxisKeys.KeyedArray(m, row=k2, col=k2)
    display(ka)
end

qns(b::SymmetricFockHilbertSpace) = qns(b.symmetry)
qntoinds(qn, b::SymmetricFockHilbertSpace) = qntoinds(qn, b.symmetry)
qns(::NoSymmetry) = (nothing,)
qns(::SimpleFockHilbertSpace) = (nothing,)
qns(::FockHilbertSpace) = (nothing,)
qns(sym::FockSymmetry) = Tuple(keys(sym.qntoinds))
qntoinds(qn, sym::FockSymmetry) = sym.qntoinds[qn]
# qntoinds(::Nothing, ::QubitBasis{M,<:Any,NoSymmetry}) where {M} = 1:2^M
qntoinds(::Nothing, H::FockHilbertSpace) = 1:length(focknumbers(H))
qntoinds(::Nothing, H::SimpleFockHilbertSpace) = 1:length(focknumbers(H))
