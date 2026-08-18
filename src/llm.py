"""The provider seam: one client, one call, one usage accounting.

Lifted out of `profiler.py` unchanged in behaviour, because two callers now need
it -- the persona layer writes one narrative per household, and `elicit.py` asks
one question per household per round. `docs/experiment_design.md` §8 flags the
provider itself as an open decision (the client uses Claude in production), so
everything that knows the OpenAI Responses API lives behind these four
functions and a swap is a change here rather than a rewrite there.

`request` is always the API's own field names -- `{"model", "instructions",
"input"}` -- so the dict that goes into the run artefact is byte-identical to
the dict that goes to the provider. That property is what makes a prompt
recoverable from an output file instead of from whichever version of the source
produced it, and it is why nothing in this module builds or edits a request.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

import openai

KEYS_PATH = Path(__file__).resolve().parent.parent / "keys.json"

# Pinned for the whole campaign, per docs/experiment_design.md §10.3: an arm
# comparison across a model version change is not an arm comparison. The
# profiles in output/profiles/ were written by this model.
DEFAULT_MODEL = "gpt-5.4-nano"

_RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)


def load_client(provider: str = "openai", keys_path: Path = KEYS_PATH) -> openai.OpenAI:
    """The client for one credential block in `keys.json`.

    `provider` names the block, not the wire protocol: `keys.example.json` carries
    `openai`, `claude` and `grok`, and the last two are reached through the same
    OpenAI client with `base_url` pointed elsewhere. Every caller outside the
    adoption-rate pilot takes the default, which is the behaviour this function
    had before the pilot needed three of them.
    """
    if not keys_path.is_file():
        raise FileNotFoundError(
            f"no {keys_path.name} found at {keys_path}. Copy keys.example.json to keys.json and fill in "
            "your OpenAI credentials."
        )
    keys = json.loads(keys_path.read_text(encoding="utf-8"))
    cfg = keys.get(provider) or {}
    if not cfg.get("api_key"):
        raise ValueError(f"{keys_path} has no {provider}.api_key set")
    return openai.OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url") or None,
        organization=cfg.get("organization") or None,
        project=cfg.get("project") or None,
    )


def usage_dict(resp: object) -> dict[str, int]:
    """What one call cost, in tokens, off the Responses API's `usage`.

    `input_tokens` is the prompt, `output_tokens` the completion. Two sub-counts
    are recorded when the API reports them, because both change what the totals
    mean: `cached_input_tokens` is the part of the prompt served from the
    provider-side cache -- which is the whole reason a persona is a stable
    prefix -- and `reasoning_tokens` is the share of `output_tokens` spent
    thinking rather than on the answer. Empty dict if the API reported nothing.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    out: dict[str, int] = {}
    for field_name in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, field_name, None)
        if value is not None:
            out[field_name] = int(value)
    cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", None)
    if cached is not None:
        out["cached_input_tokens"] = int(cached)
    reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None)
    if reasoning is not None:
        out["reasoning_tokens"] = int(reasoning)
    return out


def total_usage(records: Iterable[dict]) -> dict[str, int]:
    """Token totals over records carrying a `usage` block, summed key by key.

    Works on a whole `profiles_<village>.json` as well as on one run's decision
    log. Records written before usage was recorded contribute nothing.
    """
    out: dict[str, int] = {}
    for rec in records:
        for key, value in (rec.get("usage") or {}).items():
            out[key] = out.get(key, 0) + int(value)
    return out


def add_usage(into: dict[str, int], extra: dict[str, int]) -> dict[str, int]:
    """Accumulate `extra` into `into` in place, key by key. Returns `into`."""
    for key, value in (extra or {}).items():
        into[key] = into.get(key, 0) + int(value)
    return into


def format_usage(usage: dict[str, int]) -> str:
    """One human line: prompt tokens, response tokens, and what is inside each."""
    if not usage:
        return "no token usage recorded"
    inp, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    line = f"{inp:,} prompt + {out:,} response = {usage.get('total_tokens', inp + out):,} tokens"
    notes = []
    if usage.get("cached_input_tokens"):
        notes.append(f"{usage['cached_input_tokens']:,} of the prompt cached")
    if usage.get("reasoning_tokens"):
        notes.append(f"{usage['reasoning_tokens']:,} of the response reasoning")
    return f"{line} ({'; '.join(notes)})" if notes else line


def one_call(
    client: openai.OpenAI,
    request: dict[str, object],
    *,
    temperature: float | None = None,
    max_output_tokens: int,
    max_attempts: int = 3,
) -> tuple[str, dict[str, int]]:
    """One Responses API call, retrying transient errors only.

    `request` is sent as-is so that what an artefact records and what the model
    receives cannot drift apart. Returns the usage of the attempt that
    succeeded, not of the retries it took. Its values are not all strings: the
    adoption-rate pilot's DT designs put the response schema under `text`, which
    is a nested dict, and it goes to the provider the same way the rest does.

    `temperature=None` omits the parameter rather than sending a default for it,
    which is what the adoption-rate pilot runs on: leaving every sampling knob at
    whatever each provider ships means the three models are compared under their
    own defaults instead of under a number we picked for one of them.
    """
    delay = 1.0
    last_exc: Exception | None = None
    options: dict[str, object] = {"max_output_tokens": max_output_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.responses.create(**request, **options)
            text = (resp.output_text or "").strip()
            if not text:
                raise RuntimeError("empty response from the model")
            return text, usage_dict(resp)
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"model call failed after {max_attempts} attempts") from last_exc
