abstract type AbstractFockHilbertSpace end
##interface?
#indtofock
#focktoind



struct FockHilbertSpace{F,L,S} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    fockstates::S
end
indtofock(ind, H::AbstractFockHilbertSpace) = indtofock(ind, H.jw)
isfermionic(H::FockHilbertSpace{F}) where F = true
