struct State{S,T,B} <: AbstractArray{T,1}
    amplitudes::S
    basis::B
    function State(amplitudes::S,basis::B) where {S,B}
        @assert length(amplitudes) == length(basis)
        new{S,eltype(S),B}(amplitudes)
    end
end
Base.vec(f::State) = f.amplitudes
Base.size(f::State) = size(vec(f))
Base.getindex(f::State,i) = getindex(vec(f),i)
Base.setindex!(f::State,v,i) = setindex!(vec(f),v,i)
Base.similar(f::State) = State(deepcopy(vec(f)),basis(f))
basis(f::State) = f.basis
Base.zero(f::State) = State(zero(vec(f)),basis(f))
Base.rand(::Type{State{S,T,B}}) where {S,T,B} = State(Base.rand(T,length(B)),B())
Base.rand(::Type{<:State},basis::FermionBasis,::Type{T}) where T = State(Base.rand(T,length(basis)),basis)
Base.eachindex(f::State) = eachindex(vec(f))
Base.pairs(f::State) = pairs(vec(f))

function Base.:*(Cdag::CreationOperator, state::State) 
    out = zero(state)
    mul!(out,Cdag,state)
end
function LinearAlgebra.mul!(state2::State,op::AbstractOperator, state::State)
    for (ind,val) in pairs(state)
        state_amp = apply(op, ind,basis(state))
        for (basisstate,amp) in state_amp
            state2[index(basisstate,basis(state2))] += val*amp
        end
    end
    return state2
end
index(basisstate::Integer,::FermionBasis) = basisstate+1
apply(op::CreationOperator,ind,basis) = addparticle(particle(op),ind,basis)
addparticle(f::Fermion, ind,basis) = addfermion(siteindex(f,basis), basisstate(ind,basis))
basisstate(ind::Integer,::FermionBasis) = ind-1
