"""
    projectors(focknbrs, b::AbstractManyBodyBasis)

    This function returns the projector P and Q which projects onto the subspace spanned by the states in focknbrs (in the given order), and its complement, respectively.
"""
function projectors(focknbrs, b::AbstractManyBodyBasis)
    allfocks = fockstates(b)
    complementary_focknbrs = setdiff(allfocks, focknbrs)
    Pindices = map(f -> focktoind(f, b), focknbrs)
    Qindices = map(f -> focktoind(f, b), complementary_focknbrs)
    P = sparse(1:length(Pindices), Pindices, [1 for i in Pindices], length(Pindices), length(allfocks))
    Q = sparse(1:length(Qindices), Qindices, [1 for i in Qindices], length(Qindices), length(allfocks))
    return P, Q
end
fockstates(sym::AbelianFockSymmetry) = vcat(values(sym.qntofockstates)...)
fockstates(b::AbstractManyBodyBasis) = fockstates(symmetry(b))

"""
    block_projectors(focknbrs, b::AbstractManyBodyBasis)

    This function returns the block diagonal projector P and Q which projects onto the subspace spanned by the states in focknbrs (in the given order), and its complement, respectively.
"""
function block_projectors(focknbrs, b::AbstractManyBodyBasis)
    allfocks = fockstates(b)
    complementary_focknbrs = setdiff(allfocks, focknbrs)
    Pinds = [[focktosubind(f, b) for f in focknbrs if b.symmetry.conserved_quantity(f) == qn] for qn in qns(b)]
    Qinds = [[focktosubind(f, b) for f in complementary_focknbrs if b.symmetry.conserved_quantity(f) == qn] for qn in qns(b)]
    P = BlockDiagonal([sparse(1:length(indices), indices, [1 for i in indices], length(indices), blocksize(qn, b)) for (indices, qn) in zip(Pinds, qns(b))])
    Q = BlockDiagonal([sparse(1:length(indices), indices, [1 for i in indices], length(indices), blocksize(qn, b)) for (indices, qn) in zip(Qinds, qns(b))])
    return P, Q
end

"""
    heff(focknbrs, H, b; E = 0)

    First order effective hamiltonian at energy E in the subspace spanned by the states in focknbrs (in the given order).
"""
function heff(focknbrs, H, b; E = 0)
    P, Q = projectors(focknbrs, b)
    return P * (H + H * Q' * pinv(E*I - Q * H * Q') * Q * H) * P'
end
