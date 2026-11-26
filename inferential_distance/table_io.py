"""CSV reading and writing (rows are plain dicts of strings, as in the original code)."""

from __future__ import annotations

import csv


def read_rows(path: str, max_examples: int | None = None) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.lstrip("﻿") for name in reader.fieldnames]
        for row in reader:
            if max_examples is not None and len(rows) >= max_examples:
                break
            rows.append(row)
    return rows


def write_rows(path: str, rows: list[dict], first_columns: list[str] | None = None) -> None:
    """Write rows with the union of their keys as header, in order of first appearance."""
    fieldnames: list[str] = list(first_columns or [])
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
