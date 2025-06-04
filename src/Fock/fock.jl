
struct FockNumber
    f::Int
end
FockNumber(f::FockNumber) = f
struct JordanWignerOrdering{L}
    labels::Vector{L}
    ordering::OrderedDict{L,Int}
    function JordanWignerOrdering(labels)
        ls = collect(labels)
        dict = OrderedDict(zip(ls, Base.OneTo(length(ls))))
        new{eltype(ls)}(ls, dict)
    end
end

Base.length(jw::JordanWignerOrdering) = length(jw.labels)
Base.:(==)(jw1::JordanWignerOrdering, jw2::JordanWignerOrdering) = jw1.labels == jw2.labels && jw1.ordering == jw2.ordering
Base.keys(jw::JordanWignerOrdering) = jw.labels
Base.iterate(jw::JordanWignerOrdering) = iterate(jw.labels)
Base.iterate(jw::JordanWignerOrdering, state) = iterate(jw.labels, state)
Base.eltype(::JordanWignerOrdering{L}) = L

siteindex(label, ordering::JordanWignerOrdering) = ordering.ordering[label]
siteindices(labels, jw::JordanWignerOrdering) = map(Base.Fix2(siteindex, jw), labels)

label_at_site(n, ordering::JordanWignerOrdering) = ordering.labels[n]
focknbr_from_site_label(label, jw::JordanWignerOrdering) = focknbr_from_site_index(siteindex(label, jw))
focknbr_from_site_labels(labels, jw::JordanWignerOrdering) = mapreduce(Base.Fix2(focknbr_from_site_label, jw), +, labels, init=FockNumber(0))
focknbr_from_site_labels(labels::JordanWignerOrdering, jw::JordanWignerOrdering) = focknbr_from_site_labels(labels.labels, jw)

Base.:+(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f + f2.f)
Base.:-(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f - f2.f)
Base.:⊻(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f ⊻ f2.f)
Base.:&(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f & f2.f)
Base.:&(f1::Integer, f2::FockNumber) = FockNumber(f1 & f2.f)
Base.:|(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f | f2.f)
Base.iszero(f::FockNumber) = iszero(f.f)
Base.:*(b::Bool, f::FockNumber) = FockNumber(b * f.f)
Base.:~(f::FockNumber) = FockNumber(~f.f)

focknbr_from_bits(bits) = mapreduce(nb -> FockNumber(nb[2] * (1 << (nb[1] - 1))), +, enumerate(bits))
focknbr_from_site_index(site::Integer) = FockNumber(1 << (site - 1))
focknbr_from_site_indices(sites) = mapreduce(focknbr_from_site_index, +, sites, init=FockNumber(0))

bits(f::FockNumber, N) = digits(Bool, f.f, base=2, pad=N)
parity(f::FockNumber) = iseven(fermionnumber(f)) ? 1 : -1
fermionnumber(f::FockNumber) = count_ones(f)
Base.count_ones(f::FockNumber) = count_ones(f.f)

fermionnumber(fs::FockNumber, mask) = count_ones(fs & mask)

"""
    jwstring(site, focknbr)
    
Parity of the number of fermions to the right of site.
"""
jwstring(site, focknbr) = jwstring_left(site, focknbr)
jwstring_anti(site, focknbr) = jwstring_right(site, focknbr)
jwstring_right(site, focknbr::FockNumber) = iseven(count_ones(focknbr.f >> site)) ? 1 : -1
jwstring_left(site, focknbr::FockNumber) = iseven(count_ones(focknbr.f) - count_ones(focknbr.f >> (site - 1))) ? 1 : -1

struct FockMapper{P}
    fermionpositions::P
end
FockMapper(jws, jw::JordanWignerOrdering) = FockMapper_tuple(jws, jw)
FockMapper_collect(jws, jw::JordanWignerOrdering) = FockMapper(map(Base.Fix2(siteindices, jw) ∘ collect ∘ keys, jws)) #faster construction
FockMapper_tuple(jws, jw::JordanWignerOrdering) = FockMapper(map(Base.Fix2(siteindices, jw) ∘ Tuple ∘ keys, jws)) #faster application

struct FockShifter{M}
    shifts::M
end
(fm::FockMapper)(f::NTuple{N,FockNumber}) where {N} = mapreduce(insert_bits, +, f, fm.fermionpositions)
(fs::FockShifter)(f::NTuple{N,FockNumber}) where {N} = mapreduce((f, M) -> shift_right(f, M), +, f, fs.shifts)
shift_right(f::FockNumber, M) = FockNumber(f.f << M)

function insert_bits(_x::FockNumber, positions)
    x = _x.f
    result = 0
    bit_index = 1
    for pos in positions
        if x & (1 << (bit_index - 1)) != 0
            result |= (1 << (pos - 1))
        end
        bit_index += 1
    end
    return FockNumber(result)
end

@testitem "Fock" begin
    using Random
    Random.seed!(1234)

    N = 6
    focknumber = FockNumber(20) # = 16+4 = 00101
    fbits = bits(focknumber, N)
    @test fbits == [0, 0, 1, 0, 1, 0]

    @test QuantumDots.focknbr_from_bits(fbits) == focknumber
    @test QuantumDots.focknbr_from_bits(Tuple(fbits)) == focknumber
    @test !QuantumDots._bit(focknumber, 1)
    @test !QuantumDots._bit(focknumber, 2)
    @test QuantumDots._bit(focknumber, 3)
    @test !QuantumDots._bit(focknumber, 4)
    @test QuantumDots._bit(focknumber, 5)

    @test QuantumDots.focknbr_from_site_indices((3, 5)) == focknumber
    @test QuantumDots.focknbr_from_site_indices([3, 5]) == focknumber

    @testset "removefermion" begin
        focknbr = FockNumber(rand(1:2^N) - 1)
        fockbits = bits(focknbr, N)
        function test_remove(n)
            QuantumDots.removefermion(n, focknbr) == (fockbits[n] ? (focknbr - FockNumber(2^(n - 1)), (-1)^sum(fockbits[1:n-1])) : (FockNumber(0), 0))
        end
        @test all([test_remove(n) for n in 1:N])
    end

    @testset "ToggleFermions" begin
        focknbr = FockNumber(177) # = 1000 1101, msb to the right
        digitpositions = Vector([7, 8, 2, 3])
        daggers = BitVector([1, 0, 1, 1])
        newfocknbr, sign = QuantumDots.togglefermions(digitpositions, daggers, focknbr)
        @test newfocknbr == FockNumber(119) # = 1110 1110
        @test sign == 1
        # swap two operators
        digitpositions = Vector([7, 2, 8, 3])
        daggers = BitVector([1, 1, 0, 1])
        newfocknbr, sign = QuantumDots.togglefermions(digitpositions, daggers, focknbr)
        @test newfocknbr == FockNumber(119) # = 1110 1110
        @test sign == -1

        # annihilate twice
        digitpositions = Vector([5, 3, 5])
        daggers = BitVector([0, 1, 0])
        _, sign = QuantumDots.togglefermions(digitpositions, daggers, focknbr)
        @test sign == 0
    end

    fs = QuantumDots.fockstates(10, 5)
    @test length(fs) == binomial(10, 5)
    @test allunique(fs)
    @test all(QuantumDots.fermionnumber.(fs) .== 5)
end


##https://iopscience.iop.org/article/10.1088/1751-8121/ac0646/pdf (10c)
_bit(f::FockNumber, k) = Bool((f.f >> (k - 1)) & 1)

function FockSplitter(jw::JordanWignerOrdering, jws)
    fermionpositions = Tuple(map(Base.Fix2(siteindices, jw) ∘ Tuple ∘ collect ∘ keys, jws))
    Base.Fix2(split_focknumber, fermionpositions)
end
function split_focknumber(f::FockNumber, fermionpositions)
    map(positions -> focknbr_from_bits(map(i -> _bit(f, i), positions)), fermionpositions)
end
function split_focknumber(f::FockNumber, fockmapper::FockMapper)
    split_focknumber(f, fockmapper.fermionpositions)
end
@testitem "Split focknumber" begin
    import QuantumDots: focknbr_from_site_indices as fock
    # b1 = FermionBasis((1, 3))
    # b2 = FermionBasis((2, 4))
    # b = FermionBasis(1:4)
    jw1 = JordanWignerOrdering((1, 3))
    jw2 = JordanWignerOrdering((2, 4))
    jw = JordanWignerOrdering(1:4)
    focksplitter = QuantumDots.FockSplitter(jw, (jw1, jw2))
    @test focksplitter(fock((1, 2, 3, 4))) == (fock((1, 2)), fock((1, 2)))
    @test focksplitter(fock((1,))) == (fock((1,)), fock(()))
    @test focksplitter(fock(())) == (fock(()), fock(()))
    @test focksplitter(fock((1, 2, 3))) == (fock((1, 2)), fock((1,)))
    @test focksplitter(fock((1, 3))) == (fock((1, 2)), fock(()))
    @test focksplitter(fock((2, 4))) == (fock(()), fock((1, 2)))
    @test focksplitter(fock((3, 2))) == (fock((2,)), fock((1,)))
    @test focksplitter(fock((3, 4))) == (fock((2,)), fock((2,)))

    fockmapper = QuantumDots.FockMapper((jw1, jw2), jw)
    @test QuantumDots.split_focknumber(fock((1, 2, 4)), fockmapper) == focksplitter(fock((1, 2, 4)))

    jw1 = JordanWignerOrdering((1, 2))
    jw2 = JordanWignerOrdering((3,))
    jw = JordanWignerOrdering((1, 2, 3))
    focksplitter = QuantumDots.FockSplitter(jw, (jw1, jw2))
    @test focksplitter(fock((1, 2, 3))) == (fock((1, 2)), fock((1,)))
    @test focksplitter(fock((1, 3))) == (fock((1,)), fock((1,)))
    @test focksplitter(fock((1, 2))) == (fock((1, 2)), fock(()))
    @test focksplitter(fock((2,))) == (fock((2,)), fock(()))
    @test focksplitter(fock((2, 3))) == (fock((2,)), fock((1,)))
    @test focksplitter(fock((3,))) == (fock(()), fock((1)))
end
