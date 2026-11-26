"""Step 3: label the inference type of each atom with GPT-5.

One call per atom. The prompt receives the premise, the hypothesis without
``**`` marks, the premise span (as stored) and the hypothesis span without
``**`` marks. The raw model reply is stored in ``ent_type_gpt-<i>``; see
``parsing.normalize_type_label`` for the mapping used when summarising.

Atoms are read from ``premise_span-<i>`` / ``hypothesis_span-<i>`` for
i = 1..max_atoms, stopping at the first missing or (almost) empty premise span.
Default ``max_atoms`` = 5: every stored type-labelled output of the paper (NLI
and summarization) has labels for at most the first 5 atoms of a pair, also for
pairs with more atoms. The last version of the research script loops up to 9; it
was changed after the paper's runs.

If an API call fails, ``LLMCallError`` is raised; the rows completed before it
are already saved, and the partly labelled row is not written.
"""

from __future__ import annotations

from .llm_client import LLMCallError, get_openai_response
from .parsing import type_column
from .prompts import INFERENCE_TYPE_PROMPT, fill_inference_type_prompt, load_template
from .table_io import write_rows

DEFAULT_MODEL = "gpt-5"
MAX_ATOMS = 5


def label_row(
    row: dict,
    template: str,
    model: str = DEFAULT_MODEL,
    premise_col: str = "premise",
    hypothesis_col: str = "hypothesis",
    p_span_prefix: str = "premise_span-",
    h_span_prefix: str = "hypothesis_span-",
    client=None,
    max_atoms: int = MAX_ATOMS,
) -> dict:
    row = dict(row)
    p, h = row[premise_col], row[hypothesis_col].replace("**", "")
    for i in range(max_atoms):
        key = f"{p_span_prefix}{i + 1}"
        if key not in row or len(str(row[key])) < 2:
            break
        source, destination = row[f"{p_span_prefix}{i + 1}"], row[f"{h_span_prefix}{i + 1}"].replace("**", "")
        prompt = fill_inference_type_prompt(template, p, h, source, destination)
        row[type_column(i + 1)] = get_openai_response(prompt, model, client=client)
    return row


def annotate_inference_types(
    rows: list[dict],
    output_csv: str,
    model: str = DEFAULT_MODEL,
    premise_col: str = "premise",
    hypothesis_col: str = "hypothesis",
    template_path=INFERENCE_TYPE_PROMPT,
    client=None,
    verbose: bool = True,
    max_atoms: int = MAX_ATOMS,
) -> list[dict]:
    template = load_template(template_path)
    done = []
    try:
        for row in rows:
            new = label_row(row, template, model, premise_col, hypothesis_col, client=client, max_atoms=max_atoms)
            done.append(new)
            if verbose:
                labels = {k: v for k, v in new.items() if k.startswith("ent_type_gpt-")}
                print(f"--- example {len(done)}: {labels}")
            write_rows(output_csv, done)
    except LLMCallError as e:  # save the completed rows, then stop
        if done:
            write_rows(output_csv, done)
        e.rows_done, e.output = len(done), output_csv
        raise
    write_rows(output_csv, done)
    return done
