function phase_factor_f(focknbr1, focknbr2, subinds::NTuple)::Int
    bitmask = focknbr_from_site_indices(subinds)
    prod(i -> (jwstring_anti(i, bitmask & focknbr1) * jwstring_anti(i, bitmask & focknbr2))^_bit(focknbr2, i), subinds, init=1)
end
function phase_factor_f(focknbr1, focknbr2, N::Int)::Int
    prod(_phase_factor_f(focknbr1, focknbr2, i) for i in 1:N; init=1)
end

function _phase_factor_f(focknbr1, focknbr2, i::Int)::Int
    _bit(focknbr2, i) ? (jwstring_anti(i, focknbr1) * jwstring_anti(i, focknbr2)) : 1
end

function phase_factor_h(f1, f2, partition, jw)::Int
    #(120b)
    phase = 1
    for X in partition
        for Xp in partition
            Xpmask = focknbr_from_site_labels(Xp, jw)
            masked_f1 = Xpmask & f1
            masked_f2 = Xpmask & f2
            if X == Xp
                continue
            end
            for li in X
                i = siteindex(li, jw)
                if _bit(f2, i)
                    phase *= jwstring_anti(i, masked_f1) * jwstring_anti(i, masked_f2)
                end
            end
        end
    end
    return phase
end
@testitem "Phase factor f" begin
    import QuantumDots: phase_factor_h, phase_factor_f, siteindices

    ## Appendix A.2
    N = 2
    jw = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))

    @test [phase_factor_f(f1, f2, N) for f1 in fockstates, f2 in fockstates] ==
          [1 1 1 -1;
        1 1 -1 1;
        1 1 1 -1;
        1 1 -1 1]

    N = 3
    jw = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    @test [phase_factor_f(f1, f2, N) for f1 in fockstates, f2 in fockstates] ==
          [1 1 1 -1 1 -1 -1 -1;
        1 1 -1 1 -1 1 -1 -1;
        1 1 1 -1 -1 1 1 1;
        1 1 -1 1 1 -1 1 1;
        1 1 1 -1 1 -1 -1 -1;
        1 1 -1 1 -1 1 -1 -1;
        1 1 1 -1 -1 1 1 1;
        1 1 -1 1 1 -1 1 1]
end

@testitem "Phase factor h" begin
    # Appendix B.1
    import QuantumDots: phase_factor_h, phase_factor_f, siteindices
    N = 2
    jw = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    @test [phase_factor_h(f1, f2, [[1], [2]], jw) for f1 in fockstates, f2 in fockstates] ==
          [1 1 1 -1;
        1 1 -1 1;
        1 1 1 -1;
        1 1 -1 1]

    phf(f1, f2, subinds, N) = prod(s -> phase_factor_f(f1, f2, s), subinds) * phase_factor_f(f1, f2, N)
    let part = [[1], [2]]
        subinds = map(p -> Tuple(siteindices(p, jw)), part)
        N = length(jw)
        h = [phase_factor_h(f1, f2, part, jw) for f1 in fockstates, f2 in fockstates]
        f = [phf(f1, f2, subinds, N) for f1 in fockstates, f2 in fockstates]
        @test h == f
    end
    #
    N = 3
    jw = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    @test [phase_factor_h(f1, f2, [[1, 3], [2]], jw) for f1 in fockstates, f2 in fockstates] == [1 1 1 -1 1 1 -1 1;
        1 1 -1 1 1 1 1 -1;
        1 1 1 -1 -1 -1 1 -1;
        1 1 -1 1 -1 -1 -1 1;
        1 1 1 -1 1 1 -1 1;
        1 1 -1 1 1 1 1 -1;
        1 1 1 -1 -1 -1 1 -1;
        1 1 -1 1 -1 -1 -1 1]

    for part in [[[1, 3], [2]], [[1], [2, 3]], [[1], [2], [3]], [[3], [2, 1]], [[2], [1, 3]], [[1, 2, 3]]]
        subinds = map(p -> Tuple(siteindices(p, jw)), part)
        N = length(jw)
        h = [phase_factor_h(f1, f2, part, jw) for f1 in fockstates, f2 in fockstates]
        f = [phf(f1, f2, subinds, N) for f1 in fockstates, f2 in fockstates]
        @test h == f
    end
end

function phase_factor_l(f1, f2, X, Xbar, jw)::Int
    #(123b)
    phase = 1
    Xmask = focknbr_from_site_labels(X, jw)
    masked_f1 = Xmask & f1
    masked_f2 = Xmask & f2
    for li in Xbar
        i = siteindex(li, jw)
        if xor(_bit(f1, i), _bit(f2, i))
            phase *= jwstring_anti(i, masked_f1) * jwstring_anti(i, masked_f2)
        end
    end
    return phase
end
function phase_factor_l(f1, f2, partition, jw)::Int
    #(126b)
    X = partition
    phase = 1
    for (s, Xs) in enumerate(X)
        for Xr in Iterators.drop(X, s)
            mask = focknbr_from_site_labels(Iterators.flatten((Xs, Xr)), jw)
            phase *= phase_factor_l(mask & f1, mask & f2, Xs, Xr, jw)
        end
    end
    return phase
end

@testitem "Phase factor l" begin
    import QuantumDots: phase_factor_l, siteindices
    N = 2
    jw = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    lX = [phase_factor_l(f1, f2, [1], [2], jw) for f1 in fockstates, f2 in fockstates]
    lξ = [phase_factor_l(f1, f2, [[1], [2]], jw) for f1 in fockstates, f2 in fockstates]
    @test lX == lξ == [1 1 1 1;
              1 1 1 1;
              1 1 1 1;
              1 1 1 1]

    lX = [phase_factor_l(f1, f2, [2], [1], jw) for f1 in fockstates, f2 in fockstates]
    lξ = [phase_factor_l(f1, f2, [[2], [1]], jw) for f1 in fockstates, f2 in fockstates]
    @test lX == lξ == [1 1 1 -1;
              1 1 -1 1;
              1 -1 1 1;
              -1 1 1 1]
    #
    N = 3
    jw = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    lX = [phase_factor_l(f1, f2, [1], [2, 3], jw) for f1 in fockstates, f2 in fockstates]
    lξ = [phase_factor_l(f1, f2, [[1], [2, 3]], jw) for f1 in fockstates, f2 in fockstates]
    @test lX == lξ == ones(Int, 2^N, 2^N)

    lX = [phase_factor_l(f1, f2, [1, 2], [3], jw) for f1 in fockstates, f2 in fockstates]
    lξ = [phase_factor_l(f1, f2, [[1, 2], [3]], jw) for f1 in fockstates, f2 in fockstates]
    @test lX == lξ == ones(Int, 2^N, 2^N)
    lX = [phase_factor_l(f1, f2, [2], [1, 3], jw) for f1 in fockstates, f2 in fockstates]
    lξ = [phase_factor_l(f1, f2, [[2], [1, 3]], jw) for f1 in fockstates, f2 in fockstates]
    @test lX == lξ ==
          [1 1 1 1 1 1 -1 -1;
              1 1 1 1 1 1 -1 -1;
              1 1 1 1 -1 -1 1 1;
              1 1 1 1 -1 -1 1 1;
              1 1 -1 -1 1 1 1 1;
              1 1 -1 -1 1 1 1 1;
              -1 -1 1 1 1 1 1 1;
              -1 -1 1 1 1 1 1 1]

end
