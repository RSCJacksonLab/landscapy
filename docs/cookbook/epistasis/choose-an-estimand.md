# Choose an epistasis estimand

Choose the method from the observed design before inspecting coefficients.
Walsh, binary regression, and empirical categorical Möbius decompositions
answer different questions.

## Install and input

```bash
python -m pip install "landscapy[analysis]"
```

Every method requires equal-length sequences, finite scalar active fitness,
and an interaction order no greater than sequence length.

| Observed design | Method | Reported estimand |
| --- | --- | --- |
| Complete duplicate-free 0/1 cube | `calculate_epistasis_walsh` | Uniform-measure Fourier-Walsh coefficient |
| Complete or sampled 0/1 design | `calculate_epistasis_regression` | Effect-coded fitted coefficient; penalty-selected if regularized |
| General categorical, complete or incomplete | `calculate_epistasis_ensemble` | Observed empirical-marginal Möbius coefficient |
| Same categorical domain, no reference allele | `calculate_epistasis_reference_free` | Same observed-support Möbius estimand |

## Worked example

```python
# cookbook: test
import numpy as np
import pandas as pd

from fitness_landscape import BinarySequence
from fitness_landscape.analysis import (
    calculate_epistasis_ensemble,
    calculate_epistasis_reference_free,
    calculate_epistasis_regression,
    calculate_epistasis_walsh,
)
from fitness_landscape.core import BaseNumpySequence, FitnessLandscape, NumericFitness

binary = [BinarySequence(f"{value:02b}") for value in range(4)]
full_binary = FitnessLandscape.build(
    binary,
    graph="hamming",
    fitness_layers={"fitness": NumericFitness.from_scalars("fitness", [0, 1, 2, 4])},
)
full_binary.view("fitness")
walsh = calculate_epistasis_walsh(full_binary, order=2)

sampled = FitnessLandscape.build(
    binary[:3],
    graph="hamming",
    fitness_layers={"fitness": NumericFitness.from_scalars("fitness", [0, 1, 2])},
)
sampled.view("fitness")
regression = calculate_epistasis_regression(
    sampled, order=2, regularization="l2", alpha=0.1
)

categorical_sequences = [
    BaseNumpySequence(list(text), alphabet=["A", "B"])
    for text in ["AA", "AB", "BA"]
]
categorical = FitnessLandscape.build(
    categorical_sequences,
    graph="hamming",
    fitness_layers={"fitness": NumericFitness.from_scalars("fitness", [1, 2, 3])},
)
categorical.view("fitness")
ensemble = calculate_epistasis_ensemble(categorical, order=2)
reference_free = calculate_epistasis_reference_free(categorical, order=2)

decision_audit = pd.DataFrame(
    [
        {"design": "complete binary", "method": "walsh", "estimand": walsh["normalization"]["coefficients"]},
        {"design": "sampled rank-deficient binary", "method": "l2 regression", "estimand": regression["model"]["coefficient_solution"]},
        {"design": "incomplete categorical", "method": "ensemble", "estimand": ensemble["decomposition"]["estimand"]},
        {"design": "incomplete categorical", "method": "reference_free", "estimand": reference_free["decomposition"]["estimand"]},
    ]
)
assert walsh["domain"]["complete"] is True
assert regression["model"]["unregularized_coefficients_identifiable"] is False
assert ensemble["domain"]["complete_factorial"] is False
assert ensemble["coefficients"] == reference_free["coefficients"]
print(decision_audit.to_dict(orient="records"))
```

For binary `z_i = 1 - 2x_i`, Walsh reports
`c_S = 2^-L sum_x f(x) product_(i in S) z_i`. Regression fits the same
effect-coded columns but changes estimand when support or penalties change.
Categorical methods recursively subtract all lower-order empirical marginal
terms and do not impute missing cells.

## Common failures

- Walsh is run on a sampled or duplicate binary design.
- Rank-deficient regression is fitted without an explicit penalty.
- Penalized and data-identified coefficients are directly compared.
- Categorical observed-support coefficients are called population ANOVA effects.
- Missing cells are assumed to have been imputed.
