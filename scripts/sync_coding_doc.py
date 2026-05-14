#!/usr/bin/env python3
"""Sync coding.md from the appendix block in setupAgents.md."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "setupAgents.md"
TARGET = ROOT / "coding.md"
START = "<!-- coding.md:start -->"
END = "<!-- coding.md:end -->"


def extract_body() -> str:
    text = SOURCE.read_text(encoding="utf-8")
    try:
        body = text.split(START, 1)[1].split(END, 1)[0]
    except IndexError as exc:
        raise SystemExit(f"Could not find {START} / {END} markers in {SOURCE}") from exc
    return body.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if coding.md is out of sync")
    args = parser.parse_args()

    expected = extract_body()
    if args.check:
        actual = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if actual != expected:
            raise SystemExit("coding.md is out of sync; run python scripts/sync_coding_doc.py")
        return 0

    TARGET.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
