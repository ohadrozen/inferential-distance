"""Extreme-summarization application.

* ``generate_summaries``: ask a model for a one-sentence summary of each article.
* ``screen_entailment``: ask o3-mini, with the NLI fact-checking prompt, whether
  each summary is supported by its article; the reply (``1``/``0``/``*********``)
  is stored in ``ent pred``. Only pairs with ``ent pred == "1"`` are entailed and
  enter the inferential-distance analysis.

If an API call fails, both functions save the rows completed so far and re-raise
``LLMCallError``.
"""

from __future__ import annotations

from .llm_client import LLMCallError, get_openai_response
from .nli import parse_fact_check_response
from .prompts import (
    NLI_FACT_CHECK_PROMPT,
    SUMMARIZATION_ENCOURAGE_REPHRASING_PROMPT,
    SUMMARIZATION_PROMPT,
    fill_fact_check_prompt,
    fill_summarization_prompt,
    load_template,
)
from .table_io import write_rows

PAPER_MODELS = ["gpt-3.5-turbo", "gpt-4o-mini"]
SCREENING_MODEL = "o3-mini"
PROMPTS = {
    "standard": SUMMARIZATION_PROMPT,
    "encourage_rephrasing": SUMMARIZATION_ENCOURAGE_REPHRASING_PROMPT,
}


def generate_summaries(
    rows: list[dict],
    output_csv: str,
    model: str,
    article_col: str = "context",
    output_col: str = "response",
    prompt: str = "standard",
    max_words: int = 22,
    client=None,
    verbose: bool = True,
) -> list[dict]:
    template = load_template(PROMPTS[prompt])
    done = []
    try:
        for row in rows:
            row = dict(row)
            filled = fill_summarization_prompt(template, row[article_col], max_words)
            row[output_col] = get_openai_response(filled, model, client=client)
            done.append(row)
            if verbose:
                print(f"Example # {len(done) - 1}  summary: {row[output_col]}")
    except LLMCallError as e:  # save the completed rows, then stop
        if done:
            write_rows(output_csv, done)
        e.rows_done, e.output = len(done), output_csv
        raise
    write_rows(output_csv, done)
    return done


def screen_entailment(
    rows: list[dict],
    output_csv: str,
    premise_col: str = "premise",
    hypothesis_col: str = "hypothesis",
    model: str = SCREENING_MODEL,
    client=None,
    verbose: bool = True,
) -> list[dict]:
    template = load_template(NLI_FACT_CHECK_PROMPT)
    done = []
    try:
        for row in rows:
            row = dict(row)
            prompt = fill_fact_check_prompt(template, row[premise_col], row[hypothesis_col])
            row["ent pred"] = parse_fact_check_response(get_openai_response(prompt, model, client=client))
            done.append(row)
            if verbose:
                print(f"Example # {len(done) - 1}  ent pred: {row['ent pred']}")
    except LLMCallError as e:  # save the completed rows, then stop
        if done:
            write_rows(output_csv, done)
        e.rows_done, e.output = len(done), output_csv
        raise
    write_rows(output_csv, done)
    n = sum(r["ent pred"] == "1" for r in done)
    if done:
        print(f"Entailed: {n}/{len(done)} = {n / len(done):.2f}")
    return done
