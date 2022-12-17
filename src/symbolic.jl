function generate_fastham(gen,N)
    @variables x[1:N]
    #ps = [(@variables $p)[1] for p in params]
    m = gen(x...)
    Base.remove_linenums!.(build_function(m,x...,expression=Val{false}))
end