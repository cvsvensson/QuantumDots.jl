abstract type AbstractHilbertSpace end
abstract type AbstractFockHilbertSpace <: AbstractHilbertSpace end
##interface?
#indtofock
#focktoind
#focknumbers -> ordered iterator of fock numbers?
#isfermionic: determine if phase factors should be used

struct FockHilbertSpace{L,F,I} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    focknumbers::F
    focktoind::I
    fermionic::Bool
    function FockHilbertSpace(labels, focknumbers::F=map(FockNumber, 0:2^length(labels)-1), fermionic=true) where F
        jw = JordanWignerOrdering(labels)
        focktoind = Dict(reverse(pair) for pair in enumerate(focknumbers))
        new{eltype(jw),F,typeof(focktoind)}(jw, focknumbers, focktoind, fermionic)
    end
end
isfermionic(H::FockHilbertSpace) = H.fermionic
focknumbers(H::FockHilbertSpace) = H.focknumbers
indtofock(ind, H::FockHilbertSpace) = focknumbers(H)[ind]
focktoind(focknbr::FockNumber, H::FockHilbertSpace) = H.focktoind[focknbr]
function Base.:(==)(H1::FockHilbertSpace, H2::FockHilbertSpace)
    if H1 === H2
        return true
    end
    if H1.jw != H2.jw
        return false
    end
    if H1.fermionic != H2.fermionic
        return false
    end
    if H1.focknumbers != H2.focknumbers
        return false
    end
    if H1.focktoind != H2.focktoind
        return false
    end
    return true
end


struct SymmetricFockHilbertSpace{L,S} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    fermionic::Bool
    symmetry::S
end
isfermionic(H::SymmetricFockHilbertSpace) = H.fermionic
indtofock(ind, H::SymmetricFockHilbertSpace) = indtofock(ind, H.symmetry)
focktoind(f::FockNumber, H::SymmetricFockHilbertSpace) = focktoind(f, H.symmetry)
focknumbers(H::SymmetricFockHilbertSpace) = focknumbers(H.symmetry)
function SymmetricFockHilbertSpace(labels, qn::AbstractSymmetry, focknumbers=map(FockNumber, 0:2^length(labels)-1); fermionic=true)
    jw = JordanWignerOrdering(labels)
    labelled_symmetry = instantiate(qn, jw)
    sym_concrete = focksymmetry(focknumbers, labelled_symmetry)
    SymmetricFockHilbertSpace(jw, fermionic, sym_concrete)
end
function Base.:(==)(H1::SymmetricFockHilbertSpace, H2::SymmetricFockHilbertSpace)
    if H1 === H2
        return true
    end
    if H1.jw != H2.jw
        return false
    end
    if H1.symmetry != H2.symmetry
        return false
    end
    if H1.fermionic != H2.fermionic
        return false
    end
    return true
end


function wedge(H1::SymmetricFockHilbertSpace, H2::SymmetricFockHilbertSpace)
    isdisjoint(keys(H1.jw), keys(H2.jw)) || throw(ArgumentError("The labels of the two bases are not disjoint"))
    newlabels = vcat(collect(keys(H1.jw)), collect(keys(H2.jw)))
    qn = promote_symmetry(H1.symmetry, H2.symmetry)
    M1 = length(H1.jw)
    newfocknumbers = vec([f1 + shift_right(f2, M1) for f1 in focknumbers(H1), f2 in focknumbers(H2)])
    SymmetricFockHilbertSpace(newlabels, qn, newfocknumbers)
end

function wedge(H1::FockHilbertSpace, H2::FockHilbertSpace)
    isdisjoint(keys(H1.jw), keys(H2.jw)) || throw(ArgumentError("The labels of the two bases are not disjoint"))
    newlabels = vcat(collect(keys(H1.jw)), collect(keys(H2.jw)))
    M1 = length(H1.jw)
    newfocknumbers = vec([f1 + shift_right(f2, M1) for f1 in focknumbers(H1), f2 in focknumbers(H2)])
    FockHilbertSpace(newlabels, newfocknumbers)
end

@testitem "Wedge product of Fock Hilbert Spaces" begin
    using QuantumDots
    H1 = FockHilbertSpace(1:2)
    H2 = FockHilbertSpace(3:4)
    Hw = wedge(H1, H2)
    H3 = FockHilbertSpace(1:4)
    @test Hw == H3

    H1 = SymmetricFockHilbertSpace(1:2, FermionConservation())
    H2 = SymmetricFockHilbertSpace(3:4, FermionConservation())
    Hw = wedge(H1, H2)
    H3 = SymmetricFockHilbertSpace(1:4, FermionConservation())
    @test focknumbers(Hw) == focknumbers(H3)

    H1 = SymmetricFockHilbertSpace(1:2, ParityConservation())
    H2 = SymmetricFockHilbertSpace(3:4, ParityConservation())
    Hw = wedge(H1, H2)
    H3 = SymmetricFockHilbertSpace(1:4, ParityConservation())
    @test focknumbers(Hw) == focknumbers(H3)
end