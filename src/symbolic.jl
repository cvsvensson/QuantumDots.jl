function generate_fastham(gen,params...)
    ps = [(@variables $p)[1] for p in params]
    m = gen(ps...)
    eval.(build_function(m,ps))
end