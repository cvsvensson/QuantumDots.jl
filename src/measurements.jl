majorana_operators(a::FermionBasis) = map(f->[f+f', 1im*(f'-f)], a.dict)
majorana_operators(label,a::FermionBasis) = [a[label]' + a[label], 1im*(a[label]' - a[label])]


one_body_measurements(f, ket, a; bra = ket) = map(fermion -> dot(bra, f(fermion), ket), a.dict.values)
two_body_measurements(f, ket, a; bra = ket) = map((f1,f2) -> dot(bra, f(f1,f2), ket), Base.product(a.dict,a.dict))
