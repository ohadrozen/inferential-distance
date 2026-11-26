# Inferential Distance

Code for the paper *How Much Is Inferred? Measuring Lexically-Grounded Inferential Distance in Datasets and Models* (EACL 2027).

**Inferential distance (D)** counts the *novel inference atoms* in a hypothesis relative to a premise that entails it. A novel inference atom is a minimal span of hypothesis words that contains at least one word absent from the premise (ignoring stopwords and morphological variants) and expresses a single piece of new information entailed by some span of the premise. D is defined only for entailed pairs and is not normalised by hypothesis length. Each atom is also given an inference type: *Paraphrase or Synonyms*, *Common Sense Reasoning*, or *Specialized Knowledge* (called "World Knowledge" in the prompts and output files).

## Pipeline

| Step | Command | What it does | Model |
|---|---|---|---|
| 1 | `mark` | Highlights the novel hypothesis words as `**word**` (spaCy lemma + Lancaster stem; offline) | none |
| 2 | `extract` | Extracts the inference atoms and computes D | `o3-mini` |
| 3 | `label-types` | Labels the inference type of every atom (one call per atom) | `gpt-5` |
| 4 | `summary` | D statistics, type counts, accuracy per D (offline) | none |

Applications used in the paper:

| Command | What it does | Models in the paper |
|---|---|---|
| `nli` | Entailed / not-entailed prediction with a fact-checking prompt (answer `1`/`0`) | `gpt-3.5-turbo`, `gpt-4o`, `o3-mini` |
| `summarize` | One-sentence summaries of news articles | `gpt-3.5-turbo`, `gpt-4o-mini` |
| `screen` | Keeps only summaries entailed by their article (same prompt as `nli`) | `o3-mini` |
| `prepare-*` | Builds the input CSVs from LLM-AggreFact, SciFact, Gigaword, XSum | none |

Flag defaults are the settings used for the paper, except `summarize --max-words`: the default of 22 is the XSum setting, and the Gigaword summaries of the paper use 8. All model calls send a single user message with no other parameters (default temperature etc.).

**`o3-mini` has been retired.** OpenAI retired `o3-mini` after the paper's runs, so `extract` (and `screen`) with the default model now stop with an error saying the model is not available. Pass another model with `--model` (e.g. `--model o3`) to run them. Results will be close to, but not identical to, the paper's. The default stays `o3-mini` because it is the model the paper used.

**API errors stop the run.** A temporary API error (rate limit, timeout, connection error, server error) is retried up to 3 attempts in all, with a pause of a few seconds. Any other error (for example an unknown or retired model, or a wrong key), or a temporary error that persists, stops the command with a non-zero exit status and a message on stderr; the rows completed before the error are already saved in the output file. An API error is never stored as a model reply. This is the one deliberate change from the research code, which stored the error text as the reply (giving D = 0 without warning).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf
export OPENAI_API_KEY=...   # or: cp .env.example .env, fill it in, then: set -a; source .env; set +a
```

The API key is read only from the `OPENAI_API_KEY` environment variable (the code does not read `.env` files itself). Steps that call the API stop at once if it is not set.

The code was tested with Python 3.14, spaCy 3.8.18 and `en_core_web_trf` 3.8.0. The spaCy version used for the paper's runs is not recorded. On the stored pairs of the paper, this setup reproduces the stored highlighted hypothesis exactly for 100% of the AggreFact-XSum, Gigaword and XSum pairs checked and for about 98% of the LFQA and SciFact pairs; the remaining pairs differ in one or a few marked words, which we attribute to differences in lemmatisation between spaCy versions (not verified).

## Quick start (offline)

```bash
python -m inferential_distance mark --input examples/pairs.csv --output my_marked.csv
diff my_marked.csv examples/pairs_marked.csv          # no output: identical
python -m inferential_distance summary --input examples/atoms_example.csv
```

`examples/pairs.csv` holds four hand-written pairs. `examples/pairs_marked.csv` is the expected output of `mark`. `examples/atoms_example.csv` shows the format written by `extract` and `label-types`; it was produced with the hand-written replies in `examples/fake_replies.json`, **not** by the real models. `examples/summary_expected.txt` is the output of `summary` on it.

## Running the steps

```bash
# 1. novel words (adds column h_highlighted)
python -m inferential_distance mark --input pairs.csv --output marked.csv \
    --premise-col premise --hypothesis-col hypothesis

# 2. atoms and D (o3-mini)
python -m inferential_distance extract --input marked.csv --output atoms.csv \
    --premise-col premise --h-col h_highlighted

# 3. inference types (gpt-5)
python -m inferential_distance label-types --input atoms.csv --output typed.csv \
    --premise-col premise --hypothesis-col hypothesis

# 4. summary; e.g. only entailed pairs, with the accuracy of a model per D
python -m inferential_distance summary --input typed.csv --filter-col label --filter-value 1 \
    --accuracy-col accuracy --json summary.json
```

Use `--max-examples N` on any step to try it on the first N rows. Every step reads and writes CSV (UTF-8) and keeps all input columns.

### Input and output columns

* `mark`: needs a premise and a hypothesis column; adds `h_highlighted`.
* `extract`: needs the premise and `h_highlighted`; adds
  * `Total Atomic Units` — **D**, or `NA` (see below);
  * `Bold Words in hypothesis` — number of highlighted words inside the counted atoms;
  * `OpenAI Response` — the parsed model reply (JSON list of `{premise_span, hypothesis_span, entailment_type}`);
  * `premise_span-i`, `hypothesis_span-i`, `entailment_type-i` for each counted atom i = 1, 2, ...
* `label-types`: needs the premise, the hypothesis and the span columns; adds `ent_type_gpt-i` (the raw GPT-5 reply for atom i).
* `nli`: needs `doc`, `claim`, `label` (change with flags); adds `response` (`1`, `0`, or `*********` for any other reply) and `accuracy`. Use `--label-format scifact` for SciFact labels `SUPPORT`/`CONTRADICT`.
* `summarize`: needs `context`; adds `response`.
* `screen`: adds `ent pred` (`1` = entailed).

### How D is computed

The model call is made only if the highlighted hypothesis contains at least one `**word**` (otherwise D = 0). The reply is parsed as JSON (a surrounding ```` ```json ```` fence is removed). Only atoms whose `hypothesis_span` contains `**` are kept, and **D is the number of kept atoms**. If any kept atom has `entailment_type` equal to `Not Entailed`, D is `NA`. The `entailment_type` returned by o3-mini is not used in the paper; the reported types are the GPT-5 labels of step 3. When summarising, GPT-5 replies are mapped to the three types by keyword (`paraphrase`/`synonym`, `common`+`sense`, `world`+`knowledge`).

## Reproducing the paper's experiments

Download the datasets yourself and convert them:

```bash
# NLI: LLM-AggreFact dev (Hugging Face "lytang/LLM-AggreFact"; needs `pip install datasets pandas`)
python -m inferential_distance prepare-aggrefact --out-dir data/   # writes the Lfqa and AggreFact-XSum CSVs
# NLI: SciFact (https://github.com/allenai/scifact); the paper uses the train and the dev claims
python -m inferential_distance prepare-scifact --corpus corpus.jsonl --claims claims_train.jsonl --output data/scifact_train.csv
python -m inferential_distance prepare-scifact --corpus corpus.jsonl --claims claims_dev.jsonl --output data/scifact_dev.csv
# Summarization: Gigaword (train.article.txt / train.title.txt) and XSum (bbc-summary-data folder)
python -m inferential_distance prepare-gigaword --dir gigaword/train --output data/gigaword.csv --max-examples 1000
python -m inferential_distance prepare-xsum --dir bbc-summary-data --output data/xsum.csv --max-examples 1301
```

NLI (per dataset; repeat `nli` for each model):

```bash
python -m inferential_distance nli --input data/LLM-AggreFact_dev_truncated_balanced_Lfqa.csv --output lfqa_gpt-4o.csv --model gpt-4o
python -m inferential_distance mark --input lfqa_gpt-4o.csv --output lfqa_marked.csv --premise-col doc --hypothesis-col claim
python -m inferential_distance extract --input lfqa_marked.csv --output lfqa_atoms.csv --premise-col doc
python -m inferential_distance label-types --input lfqa_atoms.csv --output lfqa_typed.csv --premise-col doc --hypothesis-col claim
python -m inferential_distance summary --input lfqa_typed.csv --filter-col label --filter-value 1 --accuracy-col accuracy
```

Summarization (gold summaries shown; for model summaries, run `summarize` first and use `--hypothesis-col response`):

```bash
python -m inferential_distance summarize --input data/xsum.csv --output xsum_gpt-4o-mini.csv --model gpt-4o-mini
python -m inferential_distance mark --input data/xsum.csv --output xsum_gold_marked.csv --premise-col context --hypothesis-col summary
python -m inferential_distance screen --input xsum_gold_marked.csv --output xsum_gold_screened.csv --premise-col context --hypothesis-col summary
python -m inferential_distance extract --input xsum_gold_screened.csv --output xsum_gold_atoms.csv --premise-col context
python -m inferential_distance label-types --input xsum_gold_atoms.csv --output xsum_gold_typed.csv --premise-col context --hypothesis-col summary
python -m inferential_distance summary --input xsum_gold_typed.csv --filter-col "ent pred" --filter-value 1
```

## Prompts

`prompts/` holds the exact prompt files used for the paper's runs (do not reformat them; `.gitattributes` keeps their bytes, and a test checks their SHA-256):

| File | Used by | Placeholders |
|---|---|---|
| `atom_extraction.txt` | `extract` (o3-mini) | `{{premise}}`, `{{hypothesis}}` (the highlighted hypothesis) |
| `inference_type.txt` | `label-types` (gpt-5) | `{{premise}}`, `{{hypothesis}}`, `{{_source}}`, `{{_destination}}` |
| `nli_fact_check.txt` | `nli`, `screen` | `[prompt_context]`, `[prompt_claim]` |
| `summarization.txt` | `summarize --prompt standard` | `prompt_article` |
| `summarization_encourage_rephrasing.txt` | `summarize --prompt encourage_rephrasing` | `prompt_article` |

## Known quirks (kept on purpose)

This is a cleaned-up release of the research code; behaviour that affects results was kept as it was:

* Highlighting replaces every substring occurrence of each novel word in the hypothesis, one word after another, so a novel word inside a longer word is also marked (e.g. `art` inside `start`), and a novel word that occurs twice is marked twice.
* Words with non-ASCII characters, non-alphabetic tokens (numbers, hyphenated tokens) and stopwords are never novel. Two words count as the same when the Lancaster stem of one word's lemma is a substring of the other's, which can hide genuinely new words.
* If the model reply is not valid JSON, the pair gets no atoms and D = 0. `extract` prints one warning on stderr at the end of the run with the number of such pairs and their row numbers. (A failed API call does not give D = 0: it stops the run, see above.)
* Only the first 5 atoms of a pair are type-labelled (`--max-atoms`, default 5, as in all stored results of the paper; the last version of the research script used 9, changed after the paper's runs).
* The summarization prompt replaces only the word `prompt_article`, so the article appears in square brackets.
* `nli` and `screen` accept only the exact replies `1` and `0`; any other reply (including whitespace around the digit) is stored as `*********` and counted as wrong.

## Reproducibility and cost

The models are proprietary and are updated or retired by their provider, so new runs can give different atoms, types and predictions than those reported in the paper, even with the same prompts. The paper's runs were made in 2025. Every pair costs one o3-mini call (pairs with no novel word cost nothing) plus one GPT-5 call per atom; check current prices before running a full dataset.

## Tests

```bash
pytest -q
```

The tests run offline: model calls go to a fake client and no API key is needed. The test that runs spaCy is skipped if `en_core_web_trf` is not installed.

## Citation

The paper is to appear at EACL 2027; page numbers and the link will be added when the proceedings are out.

```bibtex
@inproceedings{rozen2027inferential,
  title     = {How Much Is Inferred? Measuring Lexically-Grounded Inferential Distance in Datasets and Models},
  author    = {Rozen, Ohad and Cattan, Arie and Dagan, Ido},
  booktitle = {Proceedings of EACL 2027},
  year      = {2027},
  note      = {To appear}
}
```

## License

MIT (see `LICENSE`).
