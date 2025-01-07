module QuantumDotsSymbolicsExt

using QuantumDots, Symbolics
using QuantumDots.BlockDiagonals
import QuantumDots: fastgenerator, fastblockdiagonal, TSL_generator,
    NoSymmetry, TSL_hamiltonian, FermionBdGBasis, fermion2majorana,
    SymbolicMajoranaBasis, SymbolicFermionBasis

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


function TSL_generator(qn=NoSymmetry(); blocks=qn !== NoSymmetry(), dense=false, bdg=false)
    @variables μL, μC, μR, h, t, Δ, tsoc, U
    c = if !bdg
        FermionBasis((:L, :C, :R), (:↑, :↓); qn)
    elseif bdg && qn == NoSymmetry()
        FermionBdGBasis(Tuple(collect(Base.product((:L, :C, :R), (:↑, :↓)))))
    end
    fdense = dense ? Matrix : identity
    fblock = blocks ? m -> blockdiagonal(m, c) : identity
    f = fblock ∘ fdense
    H = TSL_hamiltonian(c; μL, μC, μR, h, t, Δ, tsoc, U) |> f
    _tsl, _tsl! = build_function(H, μL, μC, μR, h, t, Δ, tsoc, U, expression=Val{false})
    tsl(; μL, μC, μR, h, t, Δ, tsoc, U) = _tsl(μL, μC, μR, h, t, Δ, tsoc, U)
    tsl!(m; μL, μC, μR, h, t, Δ, tsoc, U) = (_tsl!(m, μL, μC, μR, h, t, Δ, tsoc, U);
    m)
    randparams = (; zip((:μL, :μC, :μR, :h, :t, :Δ, :tsoc, :U), rand(8))...)
    m = TSL_hamiltonian(c; randparams...) |> f
    return tsl, tsl!, m, c
end

function fermion2majorana(f::SymbolicFermionBasis, γ::SymbolicMajoranaBasis, γ̃::SymbolicMajoranaBasis)
    sgn(x) = x.creation ? 1 : -1 # what convention to use? or should the user specify?
    is_fermion_in_basis(x, basis) = x isa QuantumDots.FermionSym && x.basis == basis
    rw = @rule ~x::(x -> is_fermion_in_basis(x, f)) => 1/2 * (γ[(~x).label] + sgn(~x) * 1im * γ̃[(~x).label])
    return Rewriters.Prewalk(Rewriters.PassThrough(rw))
end

function majorana2fermion(γ::SymbolicMajoranaBasis, γ̃::SymbolicMajoranaBasis, f::SymbolicFermionBasis)
    is_majorana_in_basis(x, basis) = x isa QuantumDots.MajoranaSym && x.basis == basis
    rw1 = @rule ~x::(x -> is_majorana_in_basis(x, γ)) => f[(~x).label] + f[(~x).label]'
    rw2 = @rule ~x::(x -> is_majorana_in_basis(x, γ̃)) => 1im * (f[(~x).label] - f[(~x).label]')
    return Rewriters.Prewalk(Rewriters.Chain([rw1, rw2]))
end

end
