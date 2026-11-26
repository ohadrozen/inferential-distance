"""Shared fixtures. All tests are offline: model calls go to FakeClient."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FakeClient:
    """Stands in for ``openai.OpenAI()``: records every request and answers with
    ``reply(model, prompt)``."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        content = self.reply(kwargs["model"], prompt)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """Make sure no test can reach the real API."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def fake_client():
    return FakeClient


def spacy_model_available(name="en_core_web_trf"):
    try:
        import spacy

        spacy.load(name)
        return True
    except Exception:
        return False
