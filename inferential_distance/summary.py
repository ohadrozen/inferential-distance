"""Step 4: per-pair D and summary tables (offline, no model calls).

This step is new code: in the original project the summary tables of the
paper were made in spreadsheets from the per-pair files. It only aggregates the
per-pair columns written by the previous steps.
"""

from __future__ import annotations

from collections import Counter

from .atoms import units_from_row
from .parsing import COL_D, COL_RAW, CANONICAL_TYPES, distance_columns, normalize_type_label


def pair_distance(row: dict):
    """D of a pair: recomputed from the stored parsed response when present,
    otherwise read from ``Total Atomic Units``. Returns an int or "NA"."""
    if row.get(COL_RAW) not in (None, ""):
        return distance_columns(units_from_row(row))[COL_D]
    d = row.get(COL_D)
    if d in (None, ""):
        return None
    return d if d == "NA" else int(float(d))


def summarize(rows: list[dict], filter_col: str | None = None, filter_value: str | None = None,
              accuracy_cols: list[str] | None = None) -> dict:
    if filter_col is not None:
        rows = [r for r in rows if str(r.get(filter_col)) == str(filter_value)]
    ds = [pair_distance(r) for r in rows]
    numeric = [d for d in ds if isinstance(d, int)]
    types = Counter()
    for r in rows:
        for k, v in r.items():
            if k.startswith("ent_type_gpt-"):
                t = normalize_type_label(v)
                if t is not None:
                    types[t] += 1
                elif v not in (None, ""):
                    types["(unmapped)"] += 1
    result = {
        "pairs": len(rows),
        "pairs_with_D": len(numeric),
        "pairs_NA": sum(d == "NA" for d in ds),
        "mean_D": (sum(numeric) / len(numeric)) if numeric else None,
        "D_distribution": dict(sorted(Counter(numeric).items())),
        "type_counts": {t: types.get(t, 0) for t in CANONICAL_TYPES + ["(unmapped)"]},
    }
    for col in accuracy_cols or []:
        per_d = {}
        for d, r in zip(ds, rows):
            if isinstance(d, int) and r.get(col) not in (None, ""):
                per_d.setdefault(d, []).append(float(r[col]))
        result[f"mean_{col}_by_D"] = {d: sum(v) / len(v) for d, v in sorted(per_d.items())}
    return result


def format_summary(s: dict) -> str:
    lines = [
        f"pairs: {s['pairs']}",
        f"pairs with numeric D: {s['pairs_with_D']} (NA: {s['pairs_NA']})",
        f"mean D: {s['mean_D']:.3f}" if s["mean_D"] is not None else "mean D: -",
        "D distribution: " + ", ".join(f"D={d}: {n}" for d, n in s["D_distribution"].items()),
        "inference types (GPT-5 labels, all atoms): " + ", ".join(f"{t}: {n}" for t, n in s["type_counts"].items()),
    ]
    for k, v in s.items():
        if k.startswith("mean_") and k.endswith("_by_D"):
            lines.append(f"{k}: " + ", ".join(f"D={d}: {a:.3f}" for d, a in v.items()))
    return "\n".join(lines)
