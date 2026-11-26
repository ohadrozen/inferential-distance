"""Turn the public datasets into the CSV files the other steps read.

Nothing is downloaded by this module except through the Hugging Face
``datasets`` library for LLM-AggreFact (optional dependency). Download SciFact,
Gigaword and XSum yourself (see README) and pass their local paths.

Each function reproduces the selection used for the paper (first N lines/files,
balancing with a fixed seed, truncation thresholds), quirks included.
"""

from __future__ import annotations

import csv
import json
import os

AGGREFACT_HF_NAME = "lytang/LLM-AggreFact"
AGGREFACT_MAX_DOC_CHARS = 31000
XSUM_MAX_CONTEXT_CHARS = 30000


# --- NLI -------------------------------------------------------------------------

def prepare_llm_aggrefact(out_dir: str, subsets: list[str] | None = None) -> list[str]:
    """Load the LLM-AggreFact dev split, truncate long documents, balance each
    sub-dataset on the label (seeded), shuffle, and write one CSV per sub-dataset
    (``LLM-AggreFact_dev_truncated_balanced_<subset>.csv``; columns include
    ``doc``, ``claim``, ``label``). The paper uses the ``Lfqa`` and
    ``AggreFact-XSum`` subsets."""
    import numpy as np
    import pandas as pd
    from datasets import load_dataset

    df = load_dataset(AGGREFACT_HF_NAME)["dev"].to_pandas()

    def truncate(text):
        if isinstance(text, str) and len(text) > AGGREFACT_MAX_DOC_CHARS:
            return text[:AGGREFACT_MAX_DOC_CHARS] + "…"
        return text

    df["doc"] = df["doc"].apply(truncate)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    rng = np.random.default_rng(seed=42)
    # All sub-datasets are processed in the original order even if only some are
    # written, because they share one random generator.
    for ds, group in df.groupby("dataset"):
        counts = group["label"].value_counts()
        if len(counts) < 2:
            balanced_group = group
        else:
            n0, n1 = counts.get(0, 0), counts.get(1, 0)
            target = min(n0, n1)
            balanced_group = pd.concat([
                group[group["label"] == 0].sample(n=target, random_state=rng),
                group[group["label"] == 1].sample(n=target, random_state=rng),
            ])
        balanced_group = balanced_group.sample(frac=1, random_state=42).reset_index(drop=True)
        if subsets is not None and ds not in subsets:
            continue
        path = os.path.join(out_dir, f"LLM-AggreFact_dev_truncated_balanced_{ds}.csv")
        balanced_group.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n", encoding="utf-8")
        written.append(path)
    return written


def prepare_scifact(corpus_jsonl: str, claims_jsonl: str, out_csv: str) -> int:
    """One row per (claim, evidence sentence): columns ``id, doc, claim, label``
    with label ``SUPPORT``/``CONTRADICT``. The paper used the train and the dev
    claims (run once per claims file and concatenate)."""
    corpus = {}
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            corpus[str(entry["doc_id"])] = entry.get("abstract", [])
    n = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as out, open(claims_jsonl, "r", encoding="utf-8") as claims:
        writer = csv.writer(out)
        writer.writerow(["id", "doc", "claim", "label"])
        for line in claims:
            claim_entry = json.loads(line)
            for doc_id, evidences in claim_entry.get("evidence", {}).items():
                abstract = corpus.get(str(doc_id), [])
                for ev in evidences:
                    for sentence_index in ev.get("sentences", []):
                        if 0 <= sentence_index < len(abstract):
                            writer.writerow([claim_entry["id"], abstract[sentence_index], claim_entry["claim"], ev.get("label")])
                            n += 1
    return n


# --- summarization ---------------------------------------------------------------

def save_examples_to_csv(examples: list[dict], output_csv: str) -> None:
    """Columns ``context``, ``summary`` first, then the rest sorted; every field quoted."""
    if not examples:
        print("No examples to save.")
        return
    all_keys = set().union(*(e.keys() for e in examples))
    preferred = [k for k in ["context", "summary"] if k in all_keys]
    fieldnames = preferred + sorted(k for k in all_keys if k not in preferred)

    def _sanitize(value):
        if value is None:
            return ""
        return str(value).replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, doublequote=True, escapechar="\\", lineterminator="\n")
        writer.writeheader()
        for e in examples:
            writer.writerow({k: _sanitize(e.get(k, "")) for k in fieldnames})


def load_gigaword(gigaword_train_dir: str, max_examples: int = 1000) -> list[dict]:
    """Article-headline pairs from ``train.article.txt`` / ``train.title.txt``
    (the tokenised Gigaword release of Rush et al. 2015). Lines 1..max_examples-1
    are taken, i.e. the first line of each file is skipped (original behaviour).
    Values keep their trailing newline."""
    with open(os.path.join(gigaword_train_dir, "train.article.txt"), "r", encoding="utf8") as f:
        lines_article = f.readlines()
    with open(os.path.join(gigaword_train_dir, "train.title.txt"), "r", encoding="utf8") as f:
        lines_summary = f.readlines()
    return [dict(context=a, summary=s) for a, s in zip(lines_article[1:max_examples], lines_summary[1:max_examples])]


def load_xsum(xsum_dir: str, max_examples: int = 1301) -> list[dict]:
    """Pairs from the per-article files of the original XSum release
    (``bbc-summary-data``): line 7 is the summary (first sentence), line 4 the
    title, lines 10+ the article. Files are taken in ``os.listdir`` order (not
    sorted, original behaviour); articles over 30,000 characters are skipped."""
    files = os.listdir(xsum_dir)
    examples = []
    for filename in files[:max_examples]:
        with open(os.path.join(xsum_dir, filename), "r", encoding="utf8") as f:
            lines = f.readlines()
        example = {"context": " ".join(lines[10:]), "summary": lines[7], "title": lines[4]}
        if len(example["context"]) > XSUM_MAX_CONTEXT_CHARS:
            print(f"skipping example '{example['title']}' since it's too long")
            continue
        examples.append(example)
    return examples
