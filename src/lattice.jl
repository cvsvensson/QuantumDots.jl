first_labels(basis) = unique(map(first, collect(keys(basis))))
cell(j, b::AbstractBasis) = map(l->b[l], filter(isequal(j) ∘ first, collect(keys(b))))