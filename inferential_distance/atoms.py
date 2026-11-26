"""Step 2: extract novel inference atoms with GPT-o3-mini and compute D.

For each pair the atom-extraction prompt is filled with the premise and the
highlighted hypothesis. The model is called only if the highlighted hypothesis
contains at least 4 ``*`` characters (i.e. at least one ``**word**``); otherwise
the pair has no atoms and D = 0.

If a reply is not valid JSON the pair also gets no atoms and D = 0 (the rule used
in the paper); such pairs are counted and reported in one warning at the end of
the run. If the API call itself fails, ``LLMCallError`` is raised after the rows
completed so far have been saved; no D is written for the failing pair.
"""

from __future__ import annotations

import csv
import json
import sys

from .llm_client import LLMCallError, get_openai_response
from .parsing import COL_BOLD, COL_D, COL_RAW, distance_columns, filter_units, parse_openai_response, span_columns
from .prompts import ATOM_EXTRACTION_PROMPT, fill_atom_extraction_prompt, load_template

DEFAULT_MODEL = "o3-mini"
H_HIGHLIGHTED = "h_highlighted"


def is_valid_json_reply(response: str) -> bool:
    """True if ``parsing.parse_openai_response`` can parse ``response`` (same fence rule)."""
    response = response.strip()
    if response.startswith("```json") and response.endswith("```"):
        response = response[7:-3].strip()
    try:
        json.loads(response)
    except json.JSONDecodeError:
        return False
    return True


def _extract(premise: str, h_highlighted: str, template: str, model: str, client) -> tuple[list, bool]:
    """(parsed atom list, reply was valid JSON) for one pair."""
    if h_highlighted.count("*") >= 4:
        prompt = fill_atom_extraction_prompt(template, premise, h_highlighted)
        response = get_openai_response(prompt, model, client=client)
        return parse_openai_response(response), is_valid_json_reply(response)
    return [], True


def extract_atoms(premise: str, h_highlighted: str, template: str, model: str = DEFAULT_MODEL, client=None) -> list:
    """Parsed atom list for one pair (empty list if no word is highlighted or the
    reply is not valid JSON). Raises ``LLMCallError`` if the API call fails."""
    return _extract(premise, h_highlighted, template, model, client)[0]


def unparseable_warning(rows_numbers: list[int], n_rows: int) -> str:
    shown = ", ".join(str(i) for i in rows_numbers[:20]) + (", ..." if len(rows_numbers) > 20 else "")
    return (f"WARNING: for {len(rows_numbers)} of {n_rows} pairs the model reply was not valid JSON; these pairs "
            f"were given no atoms (D = 0), the rule used in the paper. Rows (1-based): {shown}")


def save_atom_rows(path: str, rows: list[dict]) -> None:
    """Write the input columns, then D, bold-word count, the parsed response
    (JSON), and one premise/hypothesis/type column triple per kept atom."""
    base = []
    for r in rows:
        for k in r:
            if k not in base and k != COL_RAW and not k.startswith(("premise_span-", "hypothesis_span-", "entailment_type-")):
                base.append(k)
    for k in (COL_D, COL_BOLD, COL_RAW):
        if k in base:
            base.remove(k)
    max_units = max((len(filter_units(r.get(COL_RAW, []))) for r in rows), default=0)
    span_fields = [c for i in range(1, max_units + 1) for c in span_columns(i)]
    fieldnames = base + [COL_D, COL_BOLD, COL_RAW] + span_fields
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            units = r.get(COL_RAW, [])
            out = {k: r.get(k) for k in base}
            out.update(distance_columns(units))
            out[COL_RAW] = json.dumps(units, ensure_ascii=False)
            writer.writerow(out)


def annotate_inferential_distance(
    rows: list[dict],
    output_csv: str,
    premise_col: str = "premise",
    h_highlighted_col: str = H_HIGHLIGHTED,
    model: str = DEFAULT_MODEL,
    template_path=ATOM_EXTRACTION_PROMPT,
    client=None,
    save_every: int = 1,
    verbose: bool = True,
) -> list[dict]:
    """Annotate every row; the CSV is rewritten every ``save_every`` rows and at the end.

    If a call fails, the rows completed so far are saved and ``LLMCallError`` is
    re-raised (with ``rows_done`` and ``output`` set). Pairs whose reply was not
    valid JSON are reported in one warning on stderr at the end.
    """
    template = load_template(template_path)
    done = []
    unparseable = []
    try:
        for idx, row in enumerate(rows):
            p, h = row[premise_col], row[h_highlighted_col]
            row = dict(row)
            row[COL_RAW], ok = _extract(p, h, template, model, client)
            if not ok:
                unparseable.append(idx + 1)
            done.append(row)
            if verbose:
                print(f"\n--- example {len(done)} ---\npremise: {p}\nhypothesis: {h}\nGPT Response: {row[COL_RAW]}\n")
            if save_every and idx % save_every == 0:
                save_atom_rows(output_csv, done)
    except LLMCallError as e:
        if done:
            save_atom_rows(output_csv, done)
        e.rows_done, e.output = len(done), output_csv
        raise
    finally:
        if unparseable:
            print(unparseable_warning(unparseable, len(done)), file=sys.stderr)
    save_atom_rows(output_csv, done)
    return done


def units_from_row(row: dict) -> list:
    """Parsed atoms of a row written by ``save_atom_rows``."""
    raw = row.get(COL_RAW)
    if isinstance(raw, list):
        return raw
    if raw in (None, ""):
        return []
    return json.loads(raw)
