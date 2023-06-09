
function transition_matrix(H::DiagonalizedHamiltonian, lead::NormalLead)
    transition_matrix((diag(H.eigenvalues),H.eigenvectors),lead)
end
function transition_matrix((E,vecs)::Tuple{<:AbstractVector,<:Any}, lead::NormalLead)
    Tin = vecs'*lead.jump_in*vecs
    Tout = vecs'*lead.jump_out*vecs
    Win = zero(Tin)
    Wout = zero(Tout)
    dos = density_of_states(lead)
    T = lead.temperature
    μ = lead.chemical_potential
    for I in CartesianIndices(Win)
        n1,n2 = Tuple(I)
        Win[I] = 2π*dos*abs2(Tin[I])*fermidirac(E[n1]-E[n2],T,μ)
        Wout[I] = 2π*dos*abs2(Tout[I])*fermidirac(E[n1]-E[n2],T,-μ)
    end
    D = (Win+Wout) - Diagonal(vec(sum(Win+Wout,dims=1)))
end

function add_normalizer(m::AbstractMatrix{T}) where T
    [m; fill(one(T),size(m,2))']
end

density_of_states(lead::NormalLead) = 1 #FIXME: put in some physics here