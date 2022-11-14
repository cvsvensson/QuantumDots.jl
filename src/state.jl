struct FermionState{IDs,T}
    amplitudes::Vector{T}
    function FermionState(amplitudes::Vector{T},::Val{IDs}) where {T,IDs}
        @assert length(amplitudes) == 2^(length(IDs))
        new{IDs,T}(amplitudes)
    end
end
FermionState(amplitudes::Vector,::FermionBasis{IDs}) where IDs = FermionState(amplitudes,Val(IDs))
Base.vec(f::FermionState) = f.amplitudes
Base.size(f::FermionState) = size(vec(f))
Base.getindex(f::FermionState,i) = getindex(vec(f),i)
Base.setindex!(f::FermionState,v,i) = setindex!(vec(f),v,i)
Base.getindex(f::FermionState{IDs},::Val{ID}) where {IDs,ID} = f[2^siteindex(Val(ID),Val(IDs))]
Base.setindex!(f::FermionState{IDs},::Val{ID}) where {IDs,ID}= setindex!(vec(f),v,2^siteindex(Val(ID),Val(IDs)))
Base.similar(f::FermionState) = FermionState(deepcopy(vec(f)),basis(f))
basis(f::FermionState) = f.basis
Base.zero(f::FermionState{IDs}) where {IDs} = FermionState(zero(vec(f)),Val(IDs))
Base.rand(::Type{FermionState{IDs,T}}) where {IDs,T} = FermionState(Base.rand(T,2^(length(IDs))),Val(IDs))
Base.rand(::Type{<:FermionState},::FermionBasis{IDs},::Type{T}) where {IDs,T} = FermionState(Base.rand(T,2^(length(IDs))),Val(IDs))
Base.eachindex(f::FermionState) = eachindex(vec(f))
Base.pairs(f::FermionState) = pairs(vec(f))

function Base.:*(Cdag::FermionCreationOperator, state::FermionState) 
    out = zero(state)
    mult!(out,Cdag,state)
end
function mult!(state2::FermionState{IDs,T},::FermionCreationOperator{ID}, state::FermionState{IDs,T}) where {T,IDs,ID}
    for (ind,val) in pairs(state)
        newind, amp = addfermion(siteindex(Val(ID),Val(IDs)), ind-1)
        state2[newind] += val*amp
    end
    return state2
end