"""
    wedge(b1::FermionBasis, b2::FermionBasis)

Compute the wedge product of two `FermionBasis` objects. The symmetry of the resulting basis is computed by promote_symmetry.
"""
function wedge(b1::FermionBasis, b2::FermionBasis)
    newlabels = vcat(collect(keys(b1)), collect(keys(b2)))
    if length(unique(newlabels)) != length(newlabels)
        throw(ArgumentError("The labels of the two bases are not disjoint"))
    end
    qn = promote_symmetry(b1.symmetry, b2.symmetry)
    FermionBasis(newlabels; qn)
end

promote_symmetry(s1::AbelianFockSymmetry{<:Any,<:Any,<:Any,F}, s2::AbelianFockSymmetry{<:Any,<:Any,<:Any,F}) where {F} = s1.conserved_quantity
promote_symmetry(::AbelianFockSymmetry{<:Any,<:Any,<:Any,F1}, ::AbelianFockSymmetry{<:Any,<:Any,<:Any,F2}) where {F1,F2} = NoSymmetry()
promote_symmetry(::NoSymmetry, ::S) where {S} = NoSymmetry()
promote_symmetry(::S, ::NoSymmetry) where {S} = NoSymmetry()
promote_symmetry(::NoSymmetry, ::NoSymmetry) = NoSymmetry()

function check_wedge_basis_compatibility(b1::FermionBasis{M1}, b2::FermionBasis{M2}, b3::FermionBasis{M3}) where {M1,M2,M3}
    if M1 + M2 != M3
        throw(ArgumentError("The combined basis does not have the correct number of sites"))
    end
    if vcat(collect(keys(b1)), collect(keys(b2))) != collect(keys(b3))
        throw(ArgumentError("The labels of the output basis are not the same (or ordered the same) as the labels of the input bases. $(keys(b1)) * $(keys(b2)) != $(keys(b3))"))
    end
end

#TODO: specialize for ::NoSymmetry, where kron and parity operator can be used
#TODO: Try first permuting, then kron, then permuting back
"""
    wedge(v1::AbstractVector{T1}, b1::FermionBasis{M1}, v2::AbstractVector{T2}, b2::FermionBasis{M2}, b3::FermionBasis=wedge(b1, b2)) where {M1,M2,T1,T2}

Compute the wedge product of two vectors `v1` and `v2` in the fermion basis `b1` and `b2`, respectively. Return a vector in the fermion basis `b3`.

# Arguments
- `v1::AbstractVector`: The first vector.
- `b1::FermionBasis`: The fermion basis for `v1`.
- `v2::AbstractVector`: The second vector.
- `b2::FermionBasis`: The fermion basis for `v2`.
- `b3::FermionBasis`: The fermion basis for the resulting vector. Defaults to the wedge product of `b1` and `b2`.
"""
function wedge(v1::AbstractVector{T1}, b1::FermionBasis{M1}, v2::AbstractVector{T2}, b2::FermionBasis{M2}, b3::FermionBasis=wedge(b1, b2)) where {M1,M2,T1,T2}
    M3 = length(b3)
    check_wedge_basis_compatibility(b1, b2, b3)
    T3 = promote_type(T1, T2)
    v3 = zeros(T3, 2^M3)
    for f1 in 0:2^M1-1, f2 in 0:2^M2-1
        f3 = f1 + f2 * 2^M1
        pf = parity(f2)
        v3[focktoind(f3, b3)] += v1[focktoind(f1, b1)] * v2[focktoind(f2, b2)] * pf
    end
    return v3
end
"""
    wedge(m1::AbstractMatrix{T1}, b1::FermionBasis{M1}, m2::AbstractMatrix{T2}, b2::FermionBasis{M2}, b3::FermionBasis=wedge(b1, b2)) where {M1,M2,T1,T2}

Compute the wedge product of two matrices `m1` and `m2` with respect to the fermion bases `b1` and `b2`, respectively. Return a matrix in the fermion basis `b3`, which defaults to the wedge product of `b1` and `b2`.

# Arguments
- `m1::Union{AbstractMatrix{T1}, UniformScaling{T1}}`: The first matrix.
- `b1::FermionBasis{M1}`: The fermion basis associated with `m1`.
- `m2::Union{AbstractMatrix{T2}, UniformScaling{T2}}`: The second matrix.
- `b2::FermionBasis{M2}`: The fermion basis associated with `m2`.
- `b3::FermionBasis`: The fermion basis associated with the resulting matrix. (optional)
"""
function wedge(m1::Union{AbstractMatrix{T1}, UniformScaling{T1}}, b1::FermionBasis{M1}, m2::Union{AbstractMatrix{T2}, UniformScaling{T2}}, b2::FermionBasis{M2}, b3::FermionBasis=wedge(b1, b2)) where {M1,M2,T1,T2}
    M3 = length(b3)
    check_wedge_basis_compatibility(b1, b2, b3)
    T3 = promote_type(T1, T2)
    m3 = zeros(T3, 2^M3, 2^M3)
    for f1_1 in 0:2^M1-1, f1_2 in 0:2^M2-1
        f1_3 = f1_1 + f1_2 * 2^M1
        pf1 = parity(f1_2)
        for f2_1 in 0:2^M1-1, f2_2 in 0:2^M2-1
            f2_3 = f2_1 + f2_2 * 2^M1
            pf2 = parity(f2_2)
            m3[focktoind(f1_3, b3), focktoind(f2_3, b3)] += m1[focktoind(f1_1, b1), focktoind(f2_1, b1)] * m2[focktoind(f1_2, b2), focktoind(f2_2, b2)] * pf1 * pf2
        end
    end
    return m3
end
