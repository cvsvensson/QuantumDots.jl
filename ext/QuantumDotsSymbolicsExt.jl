module QuantumDotsSymbolicsExt

using QuantumDots, Symbolics, LinearAlgebra
using QuantumDots.BlockDiagonals
import QuantumDots: fastgenerator, fastblockdiagonal, 
    NoSymmetry, FermionBdGBasis, fermion_to_majorana, majorana_to_fermion,
    SymbolicMajoranaBasis, SymbolicFermionBasis

function Symbolics._recursive_unwrap(val::Hermitian) # Fix for Symbolics >v.6.32
    if Symbolics.symbolic_type(val) == Symbolics.NotSymbolic() && val isa Union{AbstractArray,Tuple}
        return Symbolics._recursive_unwrap.(val)
    else
        return Symbolics.unwrap(val)
    end
end
function fastgenerator(gen, N)
    @variables x[1:N]
    m = gen(x...)
    Base.remove_linenums!(build_function(m, x..., expression=Val{false})[2])
end

function fastblockdiagonal(gen, N)
    @variables x[1:N]
    m = gen(x...)
    blocks = m.blocks
    f!s = [(Base.remove_linenums!(build_function(block, x..., expression=Val{false})[2])) for block in blocks]
    function blockdiag!(mat::BlockDiagonal, xs...)
        foreach((block, f!) -> f!(block, xs...), mat.blocks, f!s)
        return BlockDiagonal(mat.blocks)
    end
end

function Symbolics.build_function(H::BlockDiagonal, params...; kwargs...)
    fs = [build_function(block, params...; kwargs...) for block in H.blocks]
    function blockdiag!(mat::BlockDiagonal, xs...)
        foreach((block, f) -> last(f)(block, xs...), mat.blocks, fs)
        return mat
    end
    function blockdiag(xs...)
        return BlockDiagonal(map(f -> first(f)(xs...), fs))
    end
    blockdiag, blockdiag!
end

function Symbolics.build_function(H::BdGMatrix, params...; kwargs...)
    mats = [H.H, H.Δ]
    fs = [build_function(mat, params...; kwargs...) for mat in mats]
    function bdgmat!(H, xs...)
        foreach((mat, f) -> last(f)(mat, xs...), [H.H, H.Δ], fs)
        return H
    end
    function bdgmat(xs...)
        mats = map(f -> first(f)(xs...), fs)
        T1 = eltype(mats[1])
        T2 = eltype(mats[2])
        if T2 == Any
            return BdGMatrix(mats[1], zeros(T1, size(mats[1])...); check=false)
        elseif T1 == Any
            return BdGMatrix(zeros(T2, size(mats[2])...), mats[2]; check=false)
        end
        return BdGMatrix(mats...; check=false)
    end
    bdgmat, bdgmat!
end


function fermion_to_majorana(f::SymbolicFermionBasis, a::SymbolicMajoranaBasis, b::SymbolicMajoranaBasis; leijnse_convention=true)
    a.universe == b.universe || throw(ArgumentError("Majorana bases must anticommute"))
    sgn(x) = leijnse_convention ? (x.creation ? -1 : 1) : (x.creation ? 1 : -1)
    is_fermion_in_basis(x, basis) = x isa QuantumDots.FermionSym && x.basis == basis
    rw = @rule ~x::(x -> is_fermion_in_basis(x, f)) => 1 // 2 * (a[(~x).label] + sgn(~x) * 1im * b[(~x).label])
    return Rewriters.Prewalk(Rewriters.PassThrough(rw))
end

function majorana_to_fermion(a::SymbolicMajoranaBasis, b::SymbolicMajoranaBasis, f::SymbolicFermionBasis; leijnse_convention=true)
    a.universe == b.universe || throw(ArgumentError("Majorana bases must anticommute"))
    sgn = leijnse_convention ? 1 : -1
    is_majorana_in_basis(x, basis) = x isa QuantumDots.MajoranaSym && x.basis == basis
    rw1 = @rule ~x::(x -> is_majorana_in_basis(x, a)) => f[(~x).label] + f[(~x).label]'
    rw2 = @rule ~x::(x -> is_majorana_in_basis(x, b)) => sgn * 1im * (f[(~x).label]' - f[(~x).label])
    return Rewriters.Prewalk(Rewriters.Chain([rw1, rw2]))
end

end
