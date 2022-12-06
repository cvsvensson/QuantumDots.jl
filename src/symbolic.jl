function generate_fastham(gen,params...)
    ps = [(@variables $p)[1] for p in params]
    m = gen(ps...)
    (build_function(m,ps,expression=Val{false}))
end