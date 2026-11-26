"""Thin wrapper around the OpenAI Chat Completions API.

The API key is read only from the ``OPENAI_API_KEY`` environment variable.
Every call sends a single user message and no other parameters (no temperature,
no max_tokens, no seed), as in the runs reported in the paper.

Error handling is the one deliberate deviation from the original research
script, which made no retries and returned the error text (``"Error: ..."``)
as if it were the model's reply, so that a failed call silently became D = 0,
a type label, a prediction or a summary. Here a failed call is never returned
as reply text:

* temporary failures (rate limit, timeout, connection error, server error 5xx)
  are retried, ``MAX_ATTEMPTS`` attempts in all, with a short pause in between;
* any other failure (unknown or retired model, authentication, bad request), or
  a temporary one that still fails on the last attempt, raises ``LLMCallError``,
  whose message names the model and gives the API's error text.

A successful reply is returned unchanged, so results are not affected.
"""

from __future__ import annotations

import os
import re
import sys
import time

API_KEY_ENV = "OPENAI_API_KEY"
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0  # pause before attempt k+1 is BACKOFF_SECONDS * 2**(k-1): 2 s, then 4 s

PAPER_MODEL_RETIRED_NOTE = (
    "o3-mini, the model used in the paper for `extract` and `screen`, has been retired by OpenAI; "
    "results with another model will be close to, but not identical to, the paper's."
)


class LLMCallError(RuntimeError):
    """A model call failed. Raised instead of returning the error as reply text.

    The steps that call the API fill in ``rows_done`` and ``output`` before
    re-raising, so that the command line can say what was saved.
    """

    def __init__(self, message: str, model: str | None = None):
        super().__init__(message)
        self.model = model
        self.rows_done: int | None = None
        self.output: str | None = None


def make_client():
    import openai

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set the {API_KEY_ENV} environment variable to call the OpenAI API.")
    return openai.OpenAI(api_key=api_key)


def require_api_key() -> None:
    """Fail fast before a long run if no key is configured."""
    if not os.environ.get(API_KEY_ENV):
        raise SystemExit(f"{API_KEY_ENV} is not set. Export it (see .env.example) and run again.")


def _openai_classes(*names) -> tuple:
    """The named ``openai`` exception classes, or () if ``openai`` is not installed."""
    try:
        import openai
    except ImportError:
        return ()
    return tuple(getattr(openai, n) for n in names if hasattr(openai, n))


def _status_code(e):
    code = getattr(e, "status_code", None)
    return code if isinstance(code, int) else None


def is_transient(e: Exception) -> bool:
    """Rate limit, timeout, connection error or 5xx server error."""
    if isinstance(e, _openai_classes("RateLimitError", "APITimeoutError", "APIConnectionError",
                                     "InternalServerError")):
        return True
    if isinstance(e, (TimeoutError, ConnectionError)):
        return True
    code = _status_code(e)
    return code is not None and (code == 429 or code >= 500)


def is_model_unavailable(e: Exception) -> bool:
    """The error says that the model does not exist or is deprecated."""
    if isinstance(e, _openai_classes("NotFoundError")) or _status_code(e) == 404:
        return True
    text = str(e).lower()
    return "model_not_found" in text or ("model" in text and ("deprecated" in text or "does not exist" in text))


def _redact(text: str) -> str:
    """Remove anything that looks like an API key from an error text."""
    return re.sub(r"sk-[A-Za-z0-9_\-*]{4,}", "sk-[redacted]", text)


def _error_message(e: Exception, model: str, attempts: int) -> str:
    api_text = _redact(f"{type(e).__name__}: {e}")
    if is_model_unavailable(e):
        return (f"The model '{model}' is not available from the OpenAI API (it does not exist, has been "
                f"deprecated or retired, or your key has no access to it). Pass another model with --model. "
                f"Note: {PAPER_MODEL_RETIRED_NOTE}\nAPI error: {api_text}")
    if is_transient(e):
        return f"OpenAI API call to model '{model}' still failing after {attempts} attempts. API error: {api_text}"
    return f"OpenAI API call to model '{model}' failed. API error: {api_text}"


def get_openai_response(prompt: str, model: str, client=None, verbose: bool = True,
                        max_attempts: int = MAX_ATTEMPTS, sleep=None) -> str:
    """Send ``prompt`` as one user message to ``model`` and return the reply text.

    ``client`` is any object with ``client.chat.completions.create(...)``; when it
    is None a real OpenAI client is created from the environment. ``sleep`` is the
    function used to pause between attempts (default ``time.sleep``). Raises
    ``LLMCallError`` if the call fails (see the module docstring).
    """
    sleep = sleep or time.sleep
    if client is None:
        try:
            client = make_client()
        except Exception as e:  # noqa: BLE001 - missing key or missing openai package
            raise LLMCallError(f"Cannot create the OpenAI client for model '{model}': {_redact(str(e))}",
                               model) from e
    for attempt in range(1, max_attempts + 1):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            if verbose:
                print(f"GPT call took {time.time() - start_time:.2f} seconds")
            return response.choices[0].message.content
        except Exception as e:  # noqa: BLE001 - classified below; never returned as text
            if is_transient(e) and attempt < max_attempts:
                pause = BACKOFF_SECONDS * 2 ** (attempt - 1)
                print(f"Temporary API error from '{model}' ({type(e).__name__}); retrying in {pause:g} s "
                      f"(attempt {attempt + 1}/{max_attempts})", file=sys.stderr)
                sleep(pause)
                continue
            raise LLMCallError(_error_message(e, model, attempt), model) from e
    raise AssertionError("unreachable")  # pragma: no cover
