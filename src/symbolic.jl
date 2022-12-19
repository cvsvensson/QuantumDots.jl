function generate_fastham(gen,N)
    @variables x[1:N]
    m = gen(x...)
    Base.remove_linenums!.(build_function(m,x...,expression=Val{false}))
end