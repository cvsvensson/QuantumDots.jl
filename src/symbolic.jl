function fastgenerator(gen,N)
    @variables x[1:N]
    m = gen(x...)
    Base.remove_linenums!(build_function(m,x...,expression=Val{false})[2])
end

function fastblockdiagonal(gen,N)
    @variables x[1:N]
    m = gen(x...)
    blocks = m.blocks
    f!s = [(Base.remove_linenums!(build_function(block,x...,expression=Val{false})[2])) for block in blocks]
    function blockdiag!(mat::BlockDiagonal,xs...)
        foreach((block, f!) -> f!(block, xs...), mat.blocks,f!s)
        return BlockDiagonal(mat.blocks)
    end
end