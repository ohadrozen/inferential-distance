"""Offline tests. No test contacts a model service: calls go to FakeClient."""

import hashlib
import json

import pytest

from conftest import FakeClient, ROOT, spacy_model_available
from inferential_distance import atoms, inference_types, nli, novel_words, parsing, prompts, summarization, summary
from inferential_distance import cli, llm_client
from inferential_distance.llm_client import LLMCallError, get_openai_response
from inferential_distance.table_io import read_rows

EXAMPLES = ROOT / "examples"

# SHA-256 of the prompt files used for the paper's runs. A change here means the
# prompts no longer match the published results.
PROMPT_SHA256 = {
    "atom_extraction.txt": "f69797c87e2d3fa23a79ac3ea9c1805dfb581e79cdf124daf3646e3939a1cd3f",
    "inference_type.txt": "d3e0cd36736c3b37ab2cbb0db15ea1d6a5b2410bbfef7de212cd8647de336a43",
    "nli_fact_check.txt": "f26dfbcce36f93eccc6aa8febaf9abf62e5918613dfa082844f5c442033d8a2c",
    "summarization.txt": "d701632b4af145d4e1ac9e9a4799989b0583753ae9c428ac4847833ab3f539cd",
    "summarization_encourage_rephrasing.txt": "a4f1a020890e9f27641878d3c58ef85a06e71bb275bddef70d1819f773d19c12",
}


# --- prompts -------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(PROMPT_SHA256))
def test_prompt_files_unchanged(name):
    data = (ROOT / "prompts" / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PROMPT_SHA256[name]


def test_atom_prompt_assembly():
    t = prompts.load_template(prompts.ATOM_EXTRACTION_PROMPT)
    assert "\r" not in t  # read in text mode, as originally
    p = prompts.fill_atom_extraction_prompt(t, "PREMISE TEXT", "a **new** word")
    assert p.rstrip().endswith("premise: PREMISE TEXT\n\nHypothesis: a **new** word")
    assert "{{" not in p


def test_type_prompt_assembly():
    t = prompts.load_template(prompts.INFERENCE_TYPE_PROMPT)
    p = prompts.fill_inference_type_prompt(t, "P", "H", "src", "dst")
    assert p.rstrip().endswith("Premise: P\n\nHypothesis: H\n\nInference to be categorized: src >> dst")


def test_summarization_prompt_keeps_brackets():
    t = prompts.load_template(prompts.SUMMARIZATION_PROMPT)
    p = prompts.fill_summarization_prompt(t, "ARTICLE")
    assert "News Article: [ARTICLE]" in p
    assert "not exceeding 22 words" in p
    assert "not exceeding 8 words" in prompts.fill_summarization_prompt(t, "ARTICLE", max_words=8)


def test_fact_check_prompt_strips_highlight():
    t = prompts.load_template(prompts.NLI_FACT_CHECK_PROMPT)
    p = prompts.fill_fact_check_prompt(t, "CTX", "a **new** claim")
    assert 'Claim: "a new claim"' in p and '"CTX"' in p


# --- parsing and D -------------------------------------------------------------------

def test_parse_fenced_and_invalid():
    assert parsing.parse_openai_response('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert parsing.parse_openai_response("  [] ") == []
    assert parsing.parse_openai_response("Error: rate limit") == []


def test_counting_rule():
    units = [
        {"premise_span": "x", "hypothesis_span": "no marks", "entailment_type": "t"},
        {"premise_span": "y", "hypothesis_span": "**one** and **two**", "entailment_type": "t"},
        {"premise_span": "z", "hypothesis_span": "**three**", "entailment_type": "t"},
    ]
    cols = parsing.distance_columns(units)
    assert cols["Total Atomic Units"] == 2
    assert cols["Bold Words in hypothesis"] == 3
    assert cols["hypothesis_span-1"] == "**one** and **two**"
    assert "hypothesis_span-3" not in cols


def test_not_entailed_gives_na():
    units = [{"hypothesis_span": "**a**", "entailment_type": "Not Entailed"}]
    assert parsing.distance_columns(units)["Total Atomic Units"] == "NA"
    # an unmarked unit does not count, even if "Not Entailed"
    units = [{"hypothesis_span": "a", "entailment_type": "Not Entailed"}]
    assert parsing.distance_columns(units)["Total Atomic Units"] == 0


def test_normalize_type_label():
    assert parsing.normalize_type_label("paraphrase or synonyms") == parsing.PARAPHRASE
    assert parsing.normalize_type_label("Common-Sense reasoning") == parsing.COMMON_SENSE
    assert parsing.normalize_type_label("World Knowledge") == parsing.WORLD_KNOWLEDGE
    assert parsing.normalize_type_label("Two-Way Entailment") is None
    assert parsing.normalize_type_label("") is None


# --- novel words (no spaCy needed) ---------------------------------------------------

def test_overlap_and_highlight():
    s1 = [("run", "running", None), ("hous", "house", None)]
    s2 = [("ran", "ran", None), ("housing", "housing", None), ("runn", "runner", None)]
    # "hous" is a substring of "housing", "run" of "runn": only "ran" is novel
    assert novel_words.remove_overlapping_strings(s1, s2) == ["ran"]
    # plain substring replacement, word by word (original behaviour)
    assert novel_words.highlight("art and start", ["art"]) == "**art** and st**art**"


@pytest.mark.skipif(not spacy_model_available(), reason="spaCy model en_core_web_trf not installed")
def test_mark_reproduces_example():
    nlp = novel_words.load_spacy()
    for row in read_rows(str(EXAMPLES / "pairs_marked.csv")):
        assert novel_words.highlight_novel_words(row["premise"], row["hypothesis"], nlp) == row["h_highlighted"]


# --- client --------------------------------------------------------------------------

def test_client_sends_only_model_and_message():
    c = FakeClient(lambda m, p: "ok")
    assert get_openai_response("hello", "o3-mini", client=c, verbose=False) == "ok"
    assert c.calls == [{"model": "o3-mini", "messages": [{"role": "user", "content": "hello"}]}]


class FakeAPIError(Exception):
    """Stands in for an openai error: has the HTTP status code, like ``openai.APIStatusError``."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


DEPRECATED = "Error code: 404 - The model `o3-mini` has been deprecated"


def test_client_errors_raise_instead_of_becoming_text():
    def boom(m, p):
        raise RuntimeError("down")
    c = FakeClient(boom)
    with pytest.raises(LLMCallError, match="down") as info:
        get_openai_response("x", "m", client=c, verbose=False, sleep=lambda s: None)
    assert "'m'" in str(info.value) and len(c.calls) == 1  # not transient: no retry
    # no key in the environment (conftest removes it): no network call, an error is raised
    with pytest.raises(LLMCallError):
        get_openai_response("x", "m", client=None, verbose=False)


def test_transient_error_is_retried():
    replies = iter([FakeAPIError("Rate limit reached", 429), "ok"])

    def flaky(m, p):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r
    c, pauses = FakeClient(flaky), []
    assert get_openai_response("x", "m", client=c, verbose=False, sleep=pauses.append) == "ok"
    assert len(c.calls) == 2 and pauses == [2.0]


def test_transient_error_gives_up_after_max_attempts():
    def down(m, p):
        raise FakeAPIError("Internal server error", 500)
    c, pauses = FakeClient(down), []
    with pytest.raises(LLMCallError, match="after 3 attempts"):
        get_openai_response("x", "m", client=c, verbose=False, sleep=pauses.append)
    assert len(c.calls) == 3 and pauses == [2.0, 4.0]


def test_deprecated_model_message_mentions_model_flag():
    def gone(m, p):
        raise FakeAPIError(DEPRECATED, 404)
    c = FakeClient(gone)
    with pytest.raises(LLMCallError) as info:
        get_openai_response("x", "o3-mini", client=c, verbose=False, sleep=lambda s: None)
    msg = str(info.value)
    assert "--model" in msg and "o3-mini" in msg and "retired" in msg and "has been deprecated" in msg
    assert len(c.calls) == 1


# --- full metric pipeline on the example ---------------------------------------------

def test_example_pipeline(tmp_path):
    from fake_pipeline import run

    client, typed_csv, text = run(tmp_path)
    # 3 extraction calls (the pair without highlighted words is skipped) + 8 type calls
    models = [c["model"] for c in client.calls]
    assert models.count("o3-mini") == 3 and models.count("gpt-5") == 8
    # type prompts get the hypothesis and the span without ** marks
    for c in client.calls:
        if c["model"] == "gpt-5":
            assert "**" not in c["messages"][0]["content"].rsplit("Premise: ", 1)[1]
    assert typed_csv.read_text(encoding="utf-8") == (EXAMPLES / "atoms_example.csv").read_text(encoding="utf-8")
    assert text + "\n" == (EXAMPLES / "summary_expected.txt").read_text(encoding="utf-8")


def test_summary_recomputes_from_raw_response():
    rows = read_rows(str(EXAMPLES / "atoms_example.csv"))
    assert [summary.pair_distance(r) for r in rows] == [4, 3, 0, 1]
    rows[0]["OpenAI Response"] = ""  # falls back to the stored column
    assert summary.pair_distance(rows[0]) == 4


def test_type_labelling_stops_at_five():
    row = {"premise": "p", "hypothesis": "h"}
    for i in range(1, 12):
        row[f"premise_span-{i}"] = f"src{i}"
        row[f"hypothesis_span-{i}"] = f"**dst{i}**"
    c = FakeClient(lambda m, p: "Common Sense Reasoning")
    out = inference_types.label_row(row, "{{_source}} >> {{_destination}}", client=c)
    assert len(c.calls) == 5 and "ent_type_gpt-6" not in out
    assert c.calls[0]["messages"][0]["content"] == "src1 >> dst1"


# --- applications --------------------------------------------------------------------

def test_fact_check_parsing():
    assert nli.parse_fact_check_response("1") == "1"
    assert nli.parse_fact_check_response("0") == "0"
    assert nli.parse_fact_check_response(" 1") == nli.INVALID
    assert nli.parse_fact_check_response("1.") == nli.INVALID


def test_run_nli_scifact_labels(tmp_path):
    rows = [{"doc": "d1", "claim": "c1", "label": "SUPPORT"}, {"doc": "d2", "claim": "c2", "label": "CONTRADICT"}]
    c = FakeClient(lambda m, p: "1")
    out = nli.run_nli(rows, str(tmp_path / "o.csv"), "gpt-4o", label_format="scifact", client=c, verbose=False)
    assert [(r["label"], r["response"], r["accuracy"]) for r in out] == [("1", "1", 1), ("0", "1", 0)]


def test_summaries_and_screening(tmp_path):
    c = FakeClient(lambda m, p: "A short summary.")
    out = summarization.generate_summaries([{"context": "ART"}], str(tmp_path / "s.csv"), "gpt-4o-mini",
                                           prompt="encourage_rephrasing", client=c, verbose=False)
    assert out[0]["response"] == "A short summary."
    assert "rephrase all other details" in c.calls[0]["messages"][0]["content"]
    c2 = FakeClient(lambda m, p: "1")
    out = summarization.screen_entailment([{"premise": "P", "h_highlighted": "a **b**"}], str(tmp_path / "e.csv"),
                                          hypothesis_col="h_highlighted", client=c2, verbose=False)
    assert out[0]["ent pred"] == "1" and c2.calls[0]["model"] == "o3-mini"
    assert '"a b"' in c2.calls[0]["messages"][0]["content"]


def test_extraction_skips_pairs_without_highlight():
    c = FakeClient(lambda m, p: "[]")
    assert atoms.extract_atoms("p", "no marks here", "{{premise}} {{hypothesis}}", client=c) == []
    assert atoms.extract_atoms("p", "one *mark*", "{{premise}} {{hypothesis}}", client=c) == []
    assert c.calls == []


# --- failed calls and unparseable replies at the command line ------------------------

def _fail_on_call(n, reply):
    """Reply function that answers with ``reply`` and raises a 404 on call number n."""
    count = {"n": 0}

    def f(m, p):
        count["n"] += 1
        if count["n"] == n:
            raise FakeAPIError(DEPRECATED, 404)
        return reply(m, p)
    return f


def _use_fake_client(monkeypatch, client):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-test-value")  # only to pass require_api_key; nothing is sent
    monkeypatch.setattr(llm_client, "make_client", lambda: client)


def test_cli_extract_stops_and_keeps_completed_rows(tmp_path, monkeypatch, capsys):
    from fake_pipeline import make_reply

    client = FakeClient(_fail_on_call(2, make_reply()))  # rows 1 and 2 call the model; row 2 fails
    _use_fake_client(monkeypatch, client)
    out = tmp_path / "atoms.csv"
    with pytest.raises(SystemExit) as info:
        cli.main(["extract", "--input", str(EXAMPLES / "pairs_marked.csv"), "--output", str(out)])
    assert info.value.code == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err and "--model" in err
    assert f"1 row(s) were completed and saved to {out}" in err
    saved = read_rows(str(out))
    assert len(saved) == 1 and saved[0]["Total Atomic Units"] == "4"  # nothing written for the failing row


@pytest.mark.parametrize("cmd,extra,input_csv,reply", [
    ("label-types", [], "atoms_example.csv", "Common Sense Reasoning"),
    ("nli", ["--model", "gpt-4o", "--premise-col", "premise", "--hypothesis-col", "hypothesis"], "pairs.csv", "1"),
    ("summarize", ["--model", "gpt-4o-mini", "--article-col", "premise"], "pairs.csv", "A summary."),
    ("screen", [], "pairs.csv", "1"),
])
def test_cli_steps_stop_and_keep_completed_rows(tmp_path, monkeypatch, capsys, cmd, extra, input_csv, reply):
    # label-types makes 4 calls for the first pair of the example, so fail on call 5 (second pair)
    fail_at = 5 if cmd == "label-types" else 2
    _use_fake_client(monkeypatch, FakeClient(_fail_on_call(fail_at, lambda m, p: reply)))
    out = tmp_path / "out.csv"
    with pytest.raises(SystemExit) as info:
        cli.main([cmd, "--input", str(EXAMPLES / input_csv), "--output", str(out)] + extra)
    assert info.value.code == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err and f"1 row(s) were completed and saved to {out}" in err
    saved = read_rows(str(out))
    assert len(saved) == 1
    assert all("Error" not in v for v in saved[0].values())


def test_cli_failure_on_first_row_writes_nothing(tmp_path, monkeypatch, capsys):
    _use_fake_client(monkeypatch, FakeClient(_fail_on_call(1, lambda m, p: "1")))
    out = tmp_path / "out.csv"
    with pytest.raises(SystemExit) as info:
        cli.main(["screen", "--input", str(EXAMPLES / "pairs.csv"), "--output", str(out)])
    assert info.value.code == 1 and not out.exists()
    assert "No row was completed" in capsys.readouterr().err


def test_unparseable_replies_are_counted(tmp_path, capsys):
    from fake_pipeline import make_reply

    good = make_reply()
    c = FakeClient(lambda m, p: "sorry, no JSON" if "Hypothesis: " in p and len(c.calls) != 1 else good(m, p))
    rows = read_rows(str(EXAMPLES / "pairs_marked.csv"))
    out = atoms.annotate_inferential_distance(rows, str(tmp_path / "a.csv"), client=c, verbose=False)
    # 3 model calls: the first reply is valid, the other two are not (D = 0, the paper's rule)
    assert [parsing.distance_columns(r["OpenAI Response"])["Total Atomic Units"] for r in out] == [4, 0, 0, 0]
    err = capsys.readouterr().err
    assert err.count("WARNING") == 1
    assert "for 2 of 4 pairs the model reply was not valid JSON" in err and "Rows (1-based): 2, 4" in err
