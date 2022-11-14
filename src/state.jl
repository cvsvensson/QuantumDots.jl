struct FermionState{S,T,B}
    amplitudes::Vector{T}
    chainlength::Integer
    basis::B
    function FermionState(amplitudes::Vector{T},chainlength::Integer,basis::B) where {T,B}
        @assert length(amplitudes) == 2^(length(species(B))*chainlength)
        new{species(B),T,B}(amplitudes,chainlength,basis)
    end
end
chainlength(f::FermionState) = f.chainlength
Base.vec(f::FermionState) = f.amplitudes
Base.size(f::FermionState) = size(vec(f))
Base.getindex(f::FermionState,i) = getindex(vec(f),i)
Base.setindex!(f::FermionState,v,i) = setindex!(vec(f),v,i)
Base.similar(f::FermionState) = FermionState(deepcopy(vec(f)),chainlength(f),basis(f))
basis(f::FermionState) = f.basis
Base.zero(f::FermionState) = FermionState(zero(vec(f)),chainlength(f),basis(f))
Base.rand(::Type{FermionState{S,T}},l::Integer) where {S,T} = FermionState(Base.rand(T,2^(length(S)*l)),l,FermionBasis{S}())
Base.eachindex(f::FermionState) = eachindex(vec(f))
Base.pairs(f::FermionState) = pairs(vec(f))

function Base.:*(Cdag::CreationOperator{Fermion{S}}, state::FermionState{SS,T,B}) where {S,SS,B,T}
    out = zero(state)
    mult!(out,Cdag,state)
end
function mult!(state2::FermionState{SS,T,B},Cdag::CreationOperator{Fermion{S}}, state::FermionState{SS,T,B}) where {S,SS,B,T}
    for (ind,val) in pairs(state)
        newind, amp = addfermion(digitposition(Cdag.particle,Val(SS)), focknbr(ind,Val(SS),symmetry(B)))
        state2[newind] += val*amp
    end
    return state2
end