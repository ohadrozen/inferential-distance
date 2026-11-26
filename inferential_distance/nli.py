"""NLI application: ask a model whether the claim (hypothesis) is supported by the
context (premise), using the fact-checking prompt.

Answer parsing (unchanged from the original): the reply must be exactly ``"1"``
(entailed / supported) or ``"0"`` (not entailed); anything else, including
replies with surrounding whitespace, is stored as ``"*********"`` and counts as
wrong. ``accuracy`` is 1 when the reply equals the gold label string.

The same prompt and parsing (with o3-mini) were used to screen summarization
pairs for entailment, see ``summarization.screen_entailment``.
"""

from __future__ import annotations

from .llm_client import LLMCallError, get_openai_response
from .prompts import NLI_FACT_CHECK_PROMPT, fill_fact_check_prompt, load_template
from .table_io import write_rows

PAPER_MODELS = ["gpt-3.5-turbo", "gpt-4o", "o3-mini"]
INVALID = "*********"
SCIFACT_LABELS = {"SUPPORT": "1", "CONTRADICT": "0"}


def parse_fact_check_response(response: str) -> str:
    return response if response in ["0", "1"] else INVALID


def predict(context: str, claim: str, template: str, model: str, client=None) -> str:
    prompt = fill_fact_check_prompt(template, context, claim)
    return parse_fact_check_response(get_openai_response(prompt, model, client=client))


def run_nli(
    rows: list[dict],
    output_csv: str,
    model: str,
    premise_col: str = "doc",
    hypothesis_col: str = "claim",
    label_col: str = "label",
    label_format: str = "binary",
    template_path=NLI_FACT_CHECK_PROMPT,
    client=None,
    verbose: bool = True,
) -> list[dict]:
    """Add ``response`` and ``accuracy`` columns.

    ``label_format``: ``"binary"`` (labels already ``0``/``1``, as in
    LLM-AggreFact) or ``"scifact"`` (``SUPPORT``/``CONTRADICT``, mapped to
    ``1``/``0`` before comparison).

    If an API call fails, the rows completed so far are saved and
    ``LLMCallError`` is re-raised.
    """
    template = load_template(template_path)
    done = []
    n_correct = 0
    try:
        for row in rows:
            row = dict(row)
            if label_format == "scifact":
                assert row[label_col] in SCIFACT_LABELS, "invalid label"
                row[label_col] = SCIFACT_LABELS[row[label_col]]
            response = predict(row[premise_col], row[hypothesis_col], template, model, client=client)
            row["response"] = response
            row["accuracy"] = 1 if row[label_col] == response else 0
            n_correct += row["accuracy"]
            done.append(row)
            if verbose:
                print(f"Example # {len(done) - 1}  label={row[label_col]}  response={response}  accuracy={row['accuracy']}")
    except LLMCallError as e:  # save the completed rows, then stop
        if done:
            write_rows(output_csv, done)
        e.rows_done, e.output = len(done), output_csv
        raise
    write_rows(output_csv, done)
    if done:
        print(f"Accuracy: {n_correct}/{len(done)} = {n_correct / len(done):.2f}")
    return done
