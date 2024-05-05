```@meta
CurrentModule = QuantumDots
```

# QuantumDots.jl Documentation

Documentation for [QuantumDots](https://github.com/cvsvensson/QuantumDots.jl).

This package provides some tools for working with quantum systems, especially interacting systems of fermions. It is not registered but can be installed by 
```julia
using Pkg; Pkg.add(url="https://github.com/cvsvensson/QuantumDots.jl")
```



## Introduction
Let's analyze a small fermionic system. We first define a basis
```julia
using QuantumDots
N = 2 # number of fermions
spatial_labels = 1:N 
internal_labels = (:↑,:↓)
c = FermionBasis(spatial_labels, internal_labels)

```

Indexing into the basis like returns sparse representations of the fermionic operators, so that one can write down Hamiltonians in a natural way
```julia
H_hopping = c[1,:↑]'c[2,:↑] + c[1,:↓]'c[2,:↓] + hc 
H_coulomb = sum(c[n,:↑]'c[n,:↑]c[n,:↓]'c[n,:↓] for n in spatial_labels)
H = H_hopping + H_coulomb
#= 16×16 SparseArrays.SparseMatrixCSC{Int64, Int64} with 23 stored entries:
⎡⠠⠂⠀⠀⠀⠀⠀⠀⎤
⎢⠀⠀⠰⢂⠑⢄⠀⠀⎥
⎢⠀⠀⠑⢄⠠⢆⠀⠀⎥
⎣⠀⠀⠀⠀⠀⠀⠰⢆⎦ =#
```

One can also work in the single particle basis `FermionBdGBasis` if the system is noninteracting. Quadratic functions of the fermionic operators produce the single particle BdG Hamiltonian.
```julia
c2 = FermionBdGBasis(spatial_labels, internal_labels)
Hfree = c2[1,:↑]'c2[2,:↑] + c2[1,:↓]'c2[2,:↓] + hc
vals, vecs = diagonalize(BdGMatrix(Hfree)) 
```
Using diagonalize on a matrix of type BdGMatrix enforces particle-hole symmetry for the eigenvectors.

## More info
* For a more in depth introduction see [pmm_notebook](https://github.com/cvsvensson/QuantumDots.jl/tree/main/examples/pmm_notebook.ipynb).

* QubitBasis and time evolution is demonstrated in [qubit_dephasing](https://github.com/cvsvensson/QuantumDots.jl/tree/main/examples/qubit_dephasing.ipynb).

* Simulation of Majorana braiding with noisy gates is demonstrated in [majorana_braiding](https://github.com/cvsvensson/QuantumDots.jl/tree/main/examples/braiding.ipynb).

* Most functionalities of the package are demonstrated in the [tests](https://github.com/cvsvensson/QuantumDots.jl/tree/main/test/runtests.jl).


```@index
```

```@autodocs
Modules = [QuantumDots]
```
