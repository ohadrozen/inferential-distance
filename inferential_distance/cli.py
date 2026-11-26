"""Command line: ``python -m inferential_distance <step> ...``.

Steps of the metric:   mark -> extract -> label-types -> summary
Applications:          nli, summarize, screen
Dataset preparation:   prepare-aggrefact, prepare-scifact, prepare-gigaword, prepare-xsum

Defaults are the settings used in the paper. Run a step with ``-h`` for its flags.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import atoms, data_prep, inference_types, nli, novel_words, summarization, summary
from .llm_client import LLMCallError, require_api_key
from .prompts import ATOM_EXTRACTION_PROMPT, INFERENCE_TYPE_PROMPT
from .table_io import read_rows, write_rows


def _io(p, needs_output=True):
    p.add_argument("--input", required=True, help="input CSV")
    if needs_output:
        p.add_argument("--output", required=True, help="output CSV")
    p.add_argument("--max-examples", type=int, default=None, help="process only the first N rows")


def cmd_mark(a):
    rows = read_rows(a.input, a.max_examples)
    nlp = novel_words.load_spacy(a.spacy_model)
    for i, r in enumerate(rows):
        r["h_highlighted"] = novel_words.highlight_novel_words(r[a.premise_col], r[a.hypothesis_col], nlp)
        if i % 20 == 0:
            print(f"marked {i + 1}/{len(rows)}")
    write_rows(a.output, rows)


def cmd_extract(a):
    require_api_key()
    rows = read_rows(a.input, a.max_examples)
    atoms.annotate_inferential_distance(rows, a.output, premise_col=a.premise_col, h_highlighted_col=a.h_col,
                                        model=a.model, template_path=a.prompt)


def cmd_label_types(a):
    require_api_key()
    rows = read_rows(a.input, a.max_examples)
    inference_types.annotate_inference_types(rows, a.output, model=a.model, premise_col=a.premise_col,
                                             hypothesis_col=a.hypothesis_col, template_path=a.prompt,
                                             max_atoms=a.max_atoms)


def cmd_summary(a):
    rows = read_rows(a.input, a.max_examples)
    s = summary.summarize(rows, a.filter_col, a.filter_value, a.accuracy_col)
    print(summary.format_summary(s))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, default=str)


def cmd_nli(a):
    require_api_key()
    rows = read_rows(a.input, a.max_examples)
    nli.run_nli(rows, a.output, a.model, premise_col=a.premise_col, hypothesis_col=a.hypothesis_col,
                label_col=a.label_col, label_format=a.label_format)


def cmd_summarize(a):
    require_api_key()
    rows = read_rows(a.input, a.max_examples)
    summarization.generate_summaries(rows, a.output, a.model, article_col=a.article_col,
                                     output_col=a.output_col, prompt=a.prompt, max_words=a.max_words)


def cmd_screen(a):
    require_api_key()
    rows = read_rows(a.input, a.max_examples)
    summarization.screen_entailment(rows, a.output, premise_col=a.premise_col, hypothesis_col=a.hypothesis_col,
                                    model=a.model)


def cmd_prepare_aggrefact(a):
    for path in data_prep.prepare_llm_aggrefact(a.out_dir, a.subsets):
        print("wrote", path)


def cmd_prepare_scifact(a):
    n = data_prep.prepare_scifact(a.corpus, a.claims, a.output)
    print(f"wrote {n} pairs to {a.output}")


def cmd_prepare_gigaword(a):
    ex = data_prep.load_gigaword(a.dir, a.max_examples)
    data_prep.save_examples_to_csv(ex, a.output)
    print(f"wrote {len(ex)} pairs to {a.output}")


def cmd_prepare_xsum(a):
    ex = data_prep.load_xsum(a.dir, a.max_examples)
    data_prep.save_examples_to_csv(ex, a.output)
    print(f"wrote {len(ex)} pairs to {a.output}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m inferential_distance", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mark", help="step 1: highlight novel hypothesis words (offline, spaCy)")
    _io(p)
    p.add_argument("--premise-col", default="premise")
    p.add_argument("--hypothesis-col", default="hypothesis")
    p.add_argument("--spacy-model", default=novel_words.DEFAULT_SPACY_MODEL)
    p.set_defaults(func=cmd_mark)

    p = sub.add_parser("extract", help="step 2: extract inference atoms and compute D (calls the API)")
    _io(p)
    p.add_argument("--premise-col", default="premise")
    p.add_argument("--h-col", default="h_highlighted", help="column with the highlighted hypothesis")
    p.add_argument("--model", default=atoms.DEFAULT_MODEL)
    p.add_argument("--prompt", default=str(ATOM_EXTRACTION_PROMPT))
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("label-types", help="step 3: label the inference type of each atom (calls the API)")
    _io(p)
    p.add_argument("--premise-col", default="premise")
    p.add_argument("--hypothesis-col", default="hypothesis")
    p.add_argument("--model", default=inference_types.DEFAULT_MODEL)
    p.add_argument("--prompt", default=str(INFERENCE_TYPE_PROMPT))
    p.add_argument("--max-atoms", type=int, default=inference_types.MAX_ATOMS,
                   help="label only the first N atoms of a pair (paper: 5)")
    p.set_defaults(func=cmd_label_types)

    p = sub.add_parser("summary", help="step 4: D statistics and type counts (offline)")
    _io(p, needs_output=False)
    p.add_argument("--filter-col", default=None, help="keep only rows where this column ...")
    p.add_argument("--filter-value", default=None, help="... equals this value (e.g. label / 1)")
    p.add_argument("--accuracy-col", action="append", default=None, help="report its mean per D (repeatable)")
    p.add_argument("--json", default=None, help="also write the summary as JSON")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("nli", help="NLI predictions with the fact-checking prompt (calls the API)")
    _io(p)
    p.add_argument("--model", required=True, help="paper: " + ", ".join(nli.PAPER_MODELS))
    p.add_argument("--premise-col", default="doc")
    p.add_argument("--hypothesis-col", default="claim")
    p.add_argument("--label-col", default="label")
    p.add_argument("--label-format", choices=["binary", "scifact"], default="binary",
                   help="binary: labels 0/1 (LLM-AggreFact); scifact: SUPPORT/CONTRADICT")
    p.set_defaults(func=cmd_nli)

    p = sub.add_parser("summarize", help="generate one-sentence summaries (calls the API)")
    _io(p)
    p.add_argument("--model", required=True, help="paper: " + ", ".join(summarization.PAPER_MODELS))
    p.add_argument("--prompt", choices=sorted(summarization.PROMPTS), default="standard")
    p.add_argument("--article-col", default="context")
    p.add_argument("--output-col", default="response")
    p.add_argument("--max-words", type=int, default=22,
                   help="word limit in the prompt: 22 as stored (XSum); the paper used 8 for Gigaword")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("screen", help="keep-entailed screening of summaries with o3-mini (calls the API)")
    _io(p)
    p.add_argument("--premise-col", default="premise")
    p.add_argument("--hypothesis-col", default="hypothesis")
    p.add_argument("--model", default=summarization.SCREENING_MODEL)
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("prepare-aggrefact", help="LLM-AggreFact dev -> balanced CSVs (needs `datasets`)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--subsets", nargs="*", default=["Lfqa", "AggreFact-XSum"])
    p.set_defaults(func=cmd_prepare_aggrefact)

    p = sub.add_parser("prepare-scifact", help="SciFact corpus + claims -> CSV")
    p.add_argument("--corpus", required=True, help="corpus.jsonl")
    p.add_argument("--claims", required=True, help="claims_train.jsonl or claims_dev.jsonl")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_prepare_scifact)

    p = sub.add_parser("prepare-gigaword", help="Gigaword train.article.txt/train.title.txt -> CSV")
    p.add_argument("--dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-examples", type=int, default=1000)
    p.set_defaults(func=cmd_prepare_gigaword)

    p = sub.add_parser("prepare-xsum", help="XSum bbc-summary-data folder -> CSV")
    p.add_argument("--dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-examples", type=int, default=1301)
    p.set_defaults(func=cmd_prepare_xsum)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except LLMCallError as e:
        print(f"\n{args.cmd}: stopped because a model call failed.\n{e}", file=sys.stderr)
        if e.rows_done:
            print(f"{e.rows_done} row(s) were completed and saved to {e.output}.", file=sys.stderr)
        elif e.output:
            print(f"No row was completed; nothing was written to {e.output}.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
