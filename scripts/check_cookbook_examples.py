#!/usr/bin/env python3
"""Execute marked Python examples in the cookbook Markdown files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
COOKBOOK = ROOT / "docs" / "cookbook"
TEST_BLOCK = re.compile(
    r"```python\n# cookbook: test\n(?P<code>.*?)\n```",
    flags=re.DOTALL,
)


def _pages(section: str | None) -> list[Path]:
    root = COOKBOOK / section if section else COOKBOOK
    pages = sorted(root.rglob("*.md"))
    return [path for path in pages if path.name != "README.md"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        help="Cookbook directory relative to docs/cookbook (for example foundations)",
    )
    args = parser.parse_args()

    pages = _pages(args.section)
    if not pages:
        raise SystemExit("no cookbook recipe pages found")

    os.chdir(ROOT)
    executed = 0
    for page in pages:
        source = page.read_text(encoding="utf-8")
        blocks = [match.group("code") for match in TEST_BLOCK.finditer(source)]
        if not blocks:
            raise RuntimeError(f"{page.relative_to(ROOT)} has no executable cookbook block")
        for index, code in enumerate(blocks, start=1):
            label = f"{page.relative_to(ROOT)}::block-{index}"
            namespace = {"__name__": "__cookbook__", "__file__": str(page)}
            exec(compile(code, label, "exec"), namespace)
            executed += 1
            print(f"PASS {label}")

    print(f"Executed {executed} cookbook example(s) from {len(pages)} page(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
