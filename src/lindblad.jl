function lindbladian(Heff,Ls)
    id = LinearMaps.UniformScalingMap(one(eltype(Heff)),size(Heff,1))
    -1im*(Heff⊗id - id⊗Heff) + sum(L-> L⊗L' - 1/2*kronsum(L'*L,L*L'),Ls, init = 0*id⊗id)
end