function wedge(b1::FermionBasis, b2::FermionBasis)
    newlabels = vcat(collect(keys(b1)), collect(keys(b2)))
    if length(unique(newlabels)) != length(newlabels)
        throw(ArgumentError("The labels of the two bases are not disjoint"))
    end
    qn = promote_symmetry(b1.symmetry, b2.symmetry)
    FermionBasis(newlabels; qn)
end

promote_symmetry(s1::AbelianFockSymmetry{<:Any,<:Any,<:Any,F},s2::AbelianFockSymmetry{<:Any,<:Any,<:Any,F}) where F = s1.conserved_quantity
promote_symmetry(::AbelianFockSymmetry{<:Any,<:Any,<:Any,F1},::AbelianFockSymmetry{<:Any,<:Any,<:Any,F2}) where {F1,F2} = NoSymmetry()
promote_symmetry(::NoSymmetry,::S) where S = NoSymmetry()
promote_symmetry(::S,::NoSymmetry) where S = NoSymmetry()
promote_symmetry(::NoSymmetry,::NoSymmetry) = NoSymmetry()

