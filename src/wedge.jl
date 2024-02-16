function wedge(b1::FermionBasis, b2::FermionBasis)
    newlabels = vcat(collect(keys(b1)), collect(keys(b2)))
    if length(unique(newlabels)) != length(newlabels)
        throw(ArgumentError("The labels of the two bases are not disjoint"))
    end
    qn = promote_symmetry(b1.symmetry, b2.symmetry)
    FermionBasis(newlabels; qn)
end

promote_symmetry(s1::AbelianFockSymmetry{<:Any,<:Any,<:Any,F}, s2::AbelianFockSymmetry{<:Any,<:Any,<:Any,F}) where {F} = s1.conserved_quantity
promote_symmetry(::AbelianFockSymmetry{<:Any,<:Any,<:Any,F1}, ::AbelianFockSymmetry{<:Any,<:Any,<:Any,F2}) where {F1,F2} = NoSymmetry()
promote_symmetry(::NoSymmetry, ::S) where {S} = NoSymmetry()
promote_symmetry(::S, ::NoSymmetry) where {S} = NoSymmetry()
promote_symmetry(::NoSymmetry, ::NoSymmetry) = NoSymmetry()



function wedge(v1::AbstractVector, b1::FermionBasis, v2::AbstractVector, b2::FermionBasis)
    b3 = wedge(b1, b2)
    wedge(v1, b1, v2, b2, b3)
end
function wedge(v1::AbstractVector{T1}, b1::FermionBasis{M1}, v2::AbstractVector{T2}, b2::FermionBasis{M2}, b3::FermionBasis) where {M1,M2,T1,T2}
    M3 = length(b3)
    if M1 + M2 != M3
        throw(ArgumentError("The combined basis does not have the correct number of sites"))
    end
    T3 = promote_type(T1, T2)
    v3 = zeros(T3, 2^M3)
    for f1 in 0:2^M1 - 1, f2 in 0:2^M2 - 1
        f3 = f1 + f2 * 2^M1
        pf = parity(f2)
        v3[focktoind(f3, b3)] += v1[focktoind(f1, b1)] * v2[focktoind(f2, b2)] * pf
    end
    return v3
end