"""Run extract -> label-types -> summary on the example pairs with hand-written
replies (examples/fake_replies.json). Used by the tests; run directly to
regenerate examples/atoms_example.csv and examples/summary_expected.txt."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import FakeClient  # noqa: E402
from inferential_distance import atoms, inference_types, summary  # noqa: E402
from inferential_distance.table_io import read_rows  # noqa: E402

EXAMPLES = ROOT / "examples"


def make_reply():
    replies = json.loads((EXAMPLES / "fake_replies.json").read_text(encoding="utf-8"))

    def reply(model, prompt):
        if model == "o3-mini":
            h = prompt.rstrip().rsplit("Hypothesis: ", 1)[1]
            return replies["extraction"][h]
        if model == "gpt-5":
            dest = prompt.rstrip().rsplit(" >> ", 1)[1]
            return replies["types"][dest]
        raise AssertionError(f"unexpected model {model}")

    return reply


def run(out_dir: Path):
    client = FakeClient(make_reply())
    rows = read_rows(str(EXAMPLES / "pairs_marked.csv"))
    atoms_csv = out_dir / "atoms_only.csv"
    atoms.annotate_inferential_distance(rows, str(atoms_csv), client=client, verbose=False)
    typed_csv = out_dir / "atoms_example.csv"
    inference_types.annotate_inference_types(read_rows(str(atoms_csv)), str(typed_csv), client=client, verbose=False)
    s = summary.format_summary(summary.summarize(read_rows(str(typed_csv))))
    return client, typed_csv, s


if __name__ == "__main__":
    _, typed, s = run(EXAMPLES)
    (EXAMPLES / "atoms_only.csv").unlink()
    (EXAMPLES / "summary_expected.txt").write_text(s + "\n", encoding="utf-8")
    print(s)
