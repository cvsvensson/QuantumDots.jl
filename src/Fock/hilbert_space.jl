abstract type AbstractHilbertSpace end
abstract type AbstractFockHilbertSpace <: AbstractHilbertSpace end
##interface?
#indtofock
#focktoind
#focknumbers -> ordered iterator of fock numbers?
#isfermionic: determine if phase factors should be used

Base.size(H::AbstractFockHilbertSpace) = (length(focknumbers(H)), length(focknumbers(H)))
function isorderedpartition(Hs, H::AbstractHilbertSpace)
    partition = map(keys, Hs)
    isorderedpartition(partition, H.jw)
end
function isorderedsubsystem(Hsub, H::AbstractHilbertSpace)
    isorderedsubsystem(Hsub.jw, H.jw)
end


struct SimpleFockHilbertSpace{L} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    fermionic::Bool
    function SimpleFockHilbertSpace(labels; fermionic=true)
        jw = JordanWignerOrdering(labels)
        new{eltype(jw)}(jw, fermionic)
    end
end
Base.keys(H::SimpleFockHilbertSpace) = keys(H.jw)
isfermionic(H::SimpleFockHilbertSpace) = H.fermionic
focknumbers(H::SimpleFockHilbertSpace) = Iterators.map(FockNumber, 0:2^length(H.jw)-1)
indtofock(ind, ::SimpleFockHilbertSpace) = FockNumber(ind - 1)
focktoind(focknbr::FockNumber, ::SimpleFockHilbertSpace) = focknbr.f + 1
function Base.:(==)(H1::SimpleFockHilbertSpace, H2::SimpleFockHilbertSpace)
    if H1 === H2
        return true
    end
    if H1.jw != H2.jw
        return false
    end
    if H1.fermionic != H2.fermionic
        return false
    end
    return true
end

struct FockHilbertSpace{L,F,I} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    focknumbers::F
    focktoind::I
    fermionic::Bool
    function FockHilbertSpace(labels, focknumbers::F=map(FockNumber, 0:2^length(labels)-1); fermionic=true) where F
        jw = JordanWignerOrdering(labels)
        focktoind = Dict(reverse(pair) for pair in enumerate(focknumbers))
        new{eltype(jw),F,typeof(focktoind)}(jw, focknumbers, focktoind, fermionic)
    end
end
Base.keys(H::FockHilbertSpace) = keys(H.jw)
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
Base.keys(H::SymmetricFockHilbertSpace) = keys(H.jw)
isfermionic(H::SymmetricFockHilbertSpace) = H.fermionic
indtofock(ind, H::SymmetricFockHilbertSpace) = indtofock(ind, H.symmetry)
focktoind(f::FockNumber, H::SymmetricFockHilbertSpace) = focktoind(f, H.symmetry)
focknumbers(H::SymmetricFockHilbertSpace) = focknumbers(H.symmetry)
focknumbers(H::SymmetricFockHilbertSpace{<:Any,NoSymmetry}) = Iterators.map(FockNumber, 0:2^length(H.jw)-1)
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

qubit_hilbert_space(labels) = SimpleFockHilbertSpace(labels; fermionic=false)
qubit_hilbert_space(labels, focknumbers) = FockHilbertSpace(labels, focknumbers; fermionic=false)
qubit_hilbert_space(labels, qn::AbstractSymmetry, focknumbers) = SymmetricFockHilbertSpace(labels, qn, focknumbers; fermionic=false)
qubit_hilbert_space(labels, qn::AbstractSymmetry) = SymmetricFockHilbertSpace(labels, qn; fermionic=false)

hilbert_space(labels; fermionic=true) = SimpleFockHilbertSpace(labels; fermionic=fermionic)
hilbert_space(labels, focknumbers; fermionic=true) = FockHilbertSpace(labels, focknumbers; fermionic=fermionic)
hilbert_space(labels, ::NoSymmetry; fermionic=true) = SimpleFockHilbertSpace(labels; fermionic=fermionic)
hilbert_space(labels, ::NoSymmetry, focknumbers; fermionic=true) = FockHilbertSpace(labels, focknumbers; fermionic=fermionic)
hilbert_space(labels, qn::AbstractSymmetry, focknumbers; fermionic=true) = SymmetricFockHilbertSpace(labels, qn, focknumbers; fermionic)
hilbert_space(labels, qn::AbstractSymmetry; fermionic=true) = SymmetricFockHilbertSpace(labels, qn; fermionic)
