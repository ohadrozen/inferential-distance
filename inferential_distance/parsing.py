"""Parse the atom-extraction response and compute the inferential distance D.

Counting rule (unchanged from the original code):

* the response is parsed as JSON (a list of ``{"premise_span", "hypothesis_span",
  "entailment_type"}`` objects), after stripping a surrounding ```` ```json ```` fence;
  if it cannot be parsed, the pair gets an empty list (and therefore D = 0);
* only units whose ``hypothesis_span`` contains ``**`` (i.e. at least one
  highlighted novel word) are kept;
* D (``Total Atomic Units``) is the number of kept units, or ``"NA"`` if any kept
  unit has ``entailment_type == "Not Entailed"``.
"""

from __future__ import annotations

import json

HIGHLIGHT = "**"
NOT_ENTAILED = "Not Entailed"

# Output column names (kept from the original files, so that stored results and
# new results have the same layout).
COL_D = "Total Atomic Units"
COL_BOLD = "Bold Words in hypothesis"
COL_RAW = "OpenAI Response"


def span_columns(i: int) -> tuple[str, str, str]:
    """Column names of the i-th (1-based) atom."""
    return f"premise_span-{i}", f"hypothesis_span-{i}", f"entailment_type-{i}"


def type_column(i: int) -> str:
    """Column name of the GPT-5 inference type of the i-th (1-based) atom."""
    return f"ent_type_gpt-{i}"


def parse_openai_response(response: str) -> list:
    try:
        response = response.strip()
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error parsing OpenAI response JSON: {e}")
        return []


def filter_units(units: list) -> list:
    """Keep the atoms whose hypothesis span contains a highlighted word."""
    return [u for u in units if HIGHLIGHT in u.get("hypothesis_span", "")]


def total_atomic_units(filtered_units: list):
    """Inferential distance D of one pair (int), or "NA"."""
    if any(u.get("entailment_type", "") == NOT_ENTAILED for u in filtered_units):
        return "NA"
    return len(filtered_units)


def bold_words_in_hypothesis(filtered_units: list) -> int:
    return sum(u.get("hypothesis_span", "").count(HIGHLIGHT) // 2 for u in filtered_units)


def distance_columns(units: list) -> dict:
    """All per-pair output columns computed from the parsed response."""
    filtered = filter_units(units)
    out = {COL_D: total_atomic_units(filtered), COL_BOLD: bold_words_in_hypothesis(filtered)}
    for i, unit in enumerate(filtered, start=1):
        p_col, h_col, t_col = span_columns(i)
        out[p_col] = unit.get("premise_span", "")
        out[h_col] = unit.get("hypothesis_span", "")
        out[t_col] = unit.get("entailment_type", "")
    return out


# --- inference-type labels -------------------------------------------------------

PARAPHRASE = "Paraphrase or Synonyms"
COMMON_SENSE = "Common Sense Reasoning"
WORLD_KNOWLEDGE = "World Knowledge"  # called "Specialized Knowledge" in the paper
CANONICAL_TYPES = [PARAPHRASE, COMMON_SENSE, WORLD_KNOWLEDGE]


def normalize_type_label(raw):
    """Map a raw type label to one of ``CANONICAL_TYPES`` (or None).

    Same mapping as the original agreement evaluation code. The GPT-5 type
    response is stored raw; this function is used only when summarising.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s == "" or s == "nan":
        return None
    s = s.replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    if ("synonym" in s) or ("paraphrase" in s):
        return PARAPHRASE
    if ("common" in s) and ("sense" in s):
        return COMMON_SENSE
    if ("world" in s) and ("knowledge" in s):
        return WORLD_KNOWLEDGE
    return None
