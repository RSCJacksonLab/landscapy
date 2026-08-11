"""Enforce non-regressing line-coverage floors for publication modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path


MINIMUM_LINE_COVERAGE = {
    "src/fitness_landscape/analysis/adaptive_walk.py": 74.0,
    "src/fitness_landscape/analysis/dirichlet_energy.py": 44.0,
    "src/fitness_landscape/analysis/epistasis.py": 84.0,
    "src/fitness_landscape/analysis/persistent_homology.py": 59.0,
    "src/fitness_landscape/analysis/statistics.py": 53.0,
}


def main(report_path: str) -> int:
    """Check a coverage.py JSON report against per-module floors."""

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    failures = []
    for filename, minimum in MINIMUM_LINE_COVERAGE.items():
        try:
            measured = report["files"][filename]["summary"][
                "percent_statements_covered"
            ]
        except KeyError:
            failures.append(f"{filename}: absent from coverage report")
            continue
        if measured < minimum:
            failures.append(f"{filename}: {measured:.1f}% < {minimum:.1f}%")

    if failures:
        print("Publication-module coverage floors failed:", file=sys.stderr)
        print("\n".join(f"- {failure}" for failure in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_coverage_floors.py COVERAGE_JSON")
    raise SystemExit(main(sys.argv[1]))
