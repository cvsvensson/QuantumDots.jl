
function LinearAlgebra.kron(A::BlockDiagonal{TA,VA}, B::BlockDiagonal{TB,VB}) where {TA,TB,VA,VB}
    VC = promote_type(VA,VB)
    TC = promote_type(TA,TB)
    C::BlockDiagonal{TC,VC} = BlockDiagonal(map(Ab-> similar(Ab,size(Ab) .* size(B)), A.blocks))
    # C::BlockDiagonal{TC,VC} = BlockDiagonal(map(Ab-> VC(zeros(size(Ab) .* size(B))), A.blocks))
    kron!(C,A,B)
end
function LinearAlgebra.kron(A::BlockDiagonal{TA,VA}, B::BlockDiagonal{TB,VB}) where {TA,TB,VA<:Diagonal,VB<:Diagonal}
    VC = promote_type(VA,VB)
    TC = promote_type(TA,TB)
    C::BlockDiagonal{TC,VC} = BlockDiagonal(map(Ab-> Diagonal{TC}(undef, size(Ab,1) .* size(B,1)), A.blocks))
    kron!(C,A,B)
end
Base.convert(::Type{D},bd::BlockDiagonal{<:Any,D}) where D<:Diagonal = Diagonal(bd)
function LinearAlgebra.kron!(C::BlockDiagonal, A::BlockDiagonal, B::BlockDiagonal{<:Any,V}) where V
    bmat = convert(V,B)
    for (Cb,Ab) in zip(C.blocks,A.blocks)
        kron!(Cb, Ab, bmat)
    end
    return C
end

LinearAlgebra.exp(D::BlockDiagonal) = BlockDiagonal(map(LinearAlgebra.exp, D.blocks))
LinearAlgebra.sqrt(D::BlockDiagonal) = BlockDiagonal([promote(map(LinearAlgebra.sqrt, D.blocks)...)...])

for f in (:cis, :log,
    :cos, :sin, :tan, :csc, :sec, :cot,
    :cosh, :sinh, :tanh, :csch, :sech, :coth,
    :acos, :asin, :atan, :acsc, :asec, :acot,
    :acosh, :asinh, :atanh, :acsch, :asech, :acoth,
    :one)
@eval Base.$f(D::BlockDiagonal) = BlockDiagonal(map(Base.$f, D.blocks))
end

kronblocksizes(A,B) = map(Ab->size(Ab) .* size(B),A.blocks)
