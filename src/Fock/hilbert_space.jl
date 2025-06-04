abstract type AbstractFockHilbertSpace end
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

# struct FermionicHilbertSpace{L,F} <: AbstractFockHilbertSpace
#     jw::JordanWignerOrdering{L}
#     odd_states::F
#     even_states::F
# end
# isfermionic(::FermionicHilbertSpace) = true
# focknumbers(H::FermionicHilbertSpace) = Iterators.flatten((H.odd_states, H.even_states))


struct SymmetricFockHilbertSpace{L,S} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    fermionic::Bool
    symmetry::S
end
isfermionic(H::SymmetricFockHilbertSpace) = H.fermionic
indtofock(ind, H::SymmetricFockHilbertSpace) = indtofock(ind, H.symmetry)
focktoind(f::FockNumber, H::SymmetricFockHilbertSpace) = focktoind(f, H.symmetry)
focknumbers(H::SymmetricFockHilbertSpace) = focknumbers(H.symmetry)

function SymmetricFockHilbertSpace(labels, qn, focknumbers=map(FockNumber, 0:2^length(labels)-1); fermionic=true)
    jw = JordanWignerOrdering(labels)
    labelled_symmetry = instantiate(qn, jw)
    sym_concrete = focksymmetry(focknumbers, labelled_symmetry)
    SymmetricFockHilbertSpace(jw, fermionic, sym_concrete)
end



# struct SpinfulFockHilbertSpace{L,S} <: AbstractFockHilbertSpace
#     jw::JordanWignerOrdering{L}
#     fermionic::Bool
# end