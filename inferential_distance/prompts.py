"""Prompt templates and how premise / hypothesis / spans are inserted into them.

The template files in ``prompts/`` are the files used for the paper's runs,
byte for byte (two of them have Windows line endings; they are read in text
mode, so the model received ``\\n`` line breaks, as in the original runs).
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

ATOM_EXTRACTION_PROMPT = PROMPTS_DIR / "atom_extraction.txt"
INFERENCE_TYPE_PROMPT = PROMPTS_DIR / "inference_type.txt"
NLI_FACT_CHECK_PROMPT = PROMPTS_DIR / "nli_fact_check.txt"
SUMMARIZATION_PROMPT = PROMPTS_DIR / "summarization.txt"
SUMMARIZATION_ENCOURAGE_REPHRASING_PROMPT = PROMPTS_DIR / "summarization_encourage_rephrasing.txt"


def load_template(path) -> str:
    # Text mode (universal newlines), as the original code read the files.
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def fill_atom_extraction_prompt(template: str, premise: str, h_highlighted: str) -> str:
    return template.replace("{{premise}}", premise).replace("{{hypothesis}}", h_highlighted)


def fill_inference_type_prompt(template: str, premise: str, hypothesis: str, source: str, destination: str) -> str:
    return (
        template.replace("{{premise}}", premise)
        .replace("{{hypothesis}}", hypothesis)
        .replace("{{_source}}", source)
        .replace("{{_destination}}", destination)
    )


def fill_fact_check_prompt(template: str, context: str, claim: str) -> str:
    # The claim is inserted with any ``**`` highlighting removed.
    return template.replace("[prompt_context]", context).replace("[prompt_claim]", claim.replace("**", ""))


SUMMARY_WORD_LIMIT = 22  # the limit written in the stored prompt files (XSum)


def fill_summarization_prompt(template: str, article: str, max_words: int = SUMMARY_WORD_LIMIT) -> str:
    # Original behaviour: only the word ``prompt_article`` is replaced, so the
    # square brackets around it stay in the prompt ("News Article: [<article>]").
    # ``max_words`` other than 22 replaces the word limit in the prompt text.
    if max_words != SUMMARY_WORD_LIMIT:
        template = template.replace(f"not exceeding {SUMMARY_WORD_LIMIT} words", f"not exceeding {max_words} words")
    return template.replace("prompt_article", article)
