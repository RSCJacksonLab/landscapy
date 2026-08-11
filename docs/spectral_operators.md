# Random-walk spectral contract

`eigenmode_decomposition(matrix="transition")` decomposes the random-walk
Laplacian

```text
L_rw = I - D^-1 A
```

on positive-degree nodes. This matrix is generally nonsymmetric, so it is not
passed to a Hermitian eigensolver. For an undirected graph with non-negative
conductance, Landscapy instead solves the similar symmetric operator

```text
L_sym = I - D^-1/2 A D^-1/2
L_rw = D^-1/2 L_sym D^1/2.
```

If `L_sym u = lambda u`, the returned right mode is
`v = D^-1/2 u`, which satisfies `L_rw v = lambda v`. The modes are real and
orthonormal under the degree measure: `V^T D V = I`.

An isolated node has a zero row in both operators, a unit node measure for
normalization, and its own stationary zero mode. Thus the multiplicity of zero
equals the number of connected components, including isolates. On a connected
non-trivial component, the first right zero mode is constant.

Dense and sparse paths both solve `L_sym`, map modes with the same rule, and
return eigenvalues in ascending algebraic order. A request for `k` modes returns
the smallest `k` transition-Laplacian eigenvalues. Values and modes are always
real for supported undirected graphs; a complex result is treated as an
internal error rather than silently cast.
