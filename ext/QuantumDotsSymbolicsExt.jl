module QuantumDotsSymbolicsExt

using QuantumDots, Symbolics
using QuantumDots.BlockDiagonals
import QuantumDots: fastgenerator, fastblockdiagonal, TSL_generator,
 NoSymmetry, TSL_hamiltonian, FermionBdGBasis

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

end