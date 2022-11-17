struct State{B,T,S} <: AbstractArray{T,1}
    amplitudes::S
    basis::B
    function State(amplitudes::S,basis::B) where {S,B}
        @assert length(amplitudes) == length(basis)
        new{B,eltype(S),S}(amplitudes,basis)
    end
end
Base.vec(f::State) = f.amplitudes
Base.size(f::State) = size(vec(f))
Base.getindex(f::State,i) = getindex(vec(f),i)
Base.setindex!(f::State,v,i) = setindex!(vec(f),v,i)
Base.similar(f::State) = State(deepcopy(vec(f)),basis(f))
basis(f::State) = f.basis
Base.zero(f::State) = State(zero(vec(f)),basis(f))
Base.rand(::Type{<:State},basis::FermionBasis,::Type{T}) where T = State(Base.rand(T,length(basis)),basis)
Base.eachindex(f::State) = eachindex(vec(f))
Base.pairs(f::State) = pairs(vec(f))
#TODO: implement  similar(ψ, promote_type(T, eltype(ψ)), length)
Base.similar(ψ::State, ::Type{T}, length::Union{Integer, AbstractUnitRange}) where T = State(similar(vec(ψ),T,length),basis(ψ))
Base.similar(ψ::State, ::Type{T}, basis::AbstractBasis) where T = State(similar(vec(ψ),T,length),basis)
Base.similar(ψ::State, ::Type{T}) where T = State(similar(vec(ψ),T),basis(ψ))


