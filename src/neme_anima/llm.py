"""Image-description via an OpenAI-compatible chat-completions endpoint.

Targets LMStudio first (``http://localhost:1234``) — its ``/v1/models`` and
``/v1/chat/completions`` endpoints follow the OpenAI shape closely, so any
other compatible server (Ollama with ``--openai``, vLLM, etc.) works without
changes. Vision is delivered as a base64 ``image_url`` data URL inline with
the user message, which is what LMStudio + GGUF VLMs accept.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import httpx

DEFAULT_PROMPT = (
    "Describe this image in 1-2 sentences for a LoRA training caption. "
    "Focus on the subject's pose, clothing, expression, background, lighting, "
    "and any distinctive details. Be concise, factual, and avoid speculating "
    "about names, intent, or off-camera context."
)
DEFAULT_ENDPOINT = "http://localhost:1234"

# Connect quickly so the UI doesn't hang on a typo'd endpoint, but allow
# generous time for the server to actually answer. `/v1/models` looks like
# it should be instantaneous, but LMStudio in particular sometimes blocks
# on it for many seconds while it's loading or enumerating GGUFs — a
# 10-second read window led to spurious "timed out" errors against LAN
# hosts that were just slow to wake up. 60s is plenty without making a
# typo'd hostname feel hung (the connect leg is what catches that).
_MODELS_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=10.0)
_DESCRIBE_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=10.0)

# Reasoning models (Qwen3, DeepSeek-R1, …) burn output tokens on a thinking
# pass before the answer, and that pass is billed against max_tokens. A budget
# sized for "1-2 sentences" gets eaten whole, so the caption comes back cut
# mid-word or empty. 800 leaves room for the think block; non-reasoning models
# are unaffected — they still hit their stop token after ~60 tokens.
_DESCRIBE_MAX_TOKENS = 800


class LLMUnavailable(RuntimeError):
    """Raised when the configured endpoint can't be reached or returned an error."""


def _normalize_endpoint(endpoint: str) -> str:
    return endpoint.rstrip("/")


def _auth_headers(api_key: str | None) -> dict[str, str]:
    """Return Bearer-auth headers when an API key is set, else an empty dict.

    LMStudio (the default target) doesn't gate either endpoint — sending an
    Authorization header is harmless but emitting one only when needed keeps
    server-side logs clean and avoids confusing intermediaries.
    """
    if api_key and api_key.strip():
        return {"Authorization": f"Bearer {api_key.strip()}"}
    return {}


def _thinking_params(disable_thinking: bool) -> dict:
    """Extra request fields that ask a reasoning model to skip its think pass.

    There is no universal knob for this. ``reasoning_effort`` is the
    OpenAI-standard spelling and the most widely honoured, so it is the one we
    send; servers that don't know it ignore unknown fields, but a strict one
    (or OpenAI itself, which rejects "none" for non-reasoning models) will
    400. That risk is why this is opt-in per project — see
    :class:`neme_anima.storage.project.LLMConfig`.
    """
    return {"reasoning_effort": "none"} if disable_thinking else {}


def discover_models(endpoint: str, api_key: str | None = None) -> list[str]:
    """Return the model IDs the endpoint exposes via ``GET /v1/models``."""
    url = f"{_normalize_endpoint(endpoint)}/v1/models"
    try:
        resp = httpx.get(url, timeout=_MODELS_TIMEOUT, headers=_auth_headers(api_key))
    except httpx.ConnectTimeout as exc:
        raise LLMUnavailable(
            f"could not reach {url}: connection timed out — check the host/port",
        ) from exc
    except httpx.ReadTimeout as exc:
        raise LLMUnavailable(
            f"{url} accepted the connection but took too long to respond — "
            "the server is reachable but stuck (often: LMStudio loading models)",
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"could not reach {url}: {exc}") from exc
    if resp.status_code != 200:
        raise LLMUnavailable(
            f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise LLMUnavailable(f"non-JSON response from {url}: {exc}") from exc
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise LLMUnavailable(f"unexpected schema from {url}: missing 'data' list")
    out: list[str] = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("id"), str):
            out.append(it["id"])
    return sorted(out)


def _image_to_data_url(image_path: Path, *, max_dim: int | None = None) -> str:
    """Encode an image as a base64 data URL.

    With ``max_dim`` set the image is downscaled to fit a ``max_dim`` box and
    re-encoded as JPEG. Vision-token cost scales with resolution, so a 2 MB
    full-res crop wastes context for detail the tagger doesn't need — the
    review path (multi-round, tool-calling, tight context budgets) passes a
    ceiling; ``describe_image`` keeps the original behaviour by leaving it None.
    """
    if max_dim:
        try:
            import io

            from PIL import Image

            with Image.open(image_path) as im:
                im = im.convert("RGB")
                im.thumbnail((max_dim, max_dim))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=88)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:  # noqa: BLE001 — fall back to raw bytes on any PIL issue
            pass
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def describe_image(
    *,
    endpoint: str,
    model: str,
    image_path: Path,
    prompt: str = DEFAULT_PROMPT,
    danbooru_tags: str | None = None,
    api_key: str | None = None,
    disable_thinking: bool = False,
) -> str:
    """Send the image to ``/v1/chat/completions`` and return the description text.

    ``danbooru_tags`` is passed as additional grounding context so the VLM
    doesn't contradict the tagger's labels — useful when the LoRA pipeline
    cares about both lines staying coherent.
    """
    url = f"{_normalize_endpoint(endpoint)}/v1/chat/completions"
    user_text = prompt
    if danbooru_tags:
        user_text = (
            f"{prompt}\n\nReference tags from a tagger (use as hints, not as a "
            f"verbatim copy): {danbooru_tags}"
        )
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url",
                 "image_url": {"url": _image_to_data_url(image_path)}},
            ],
        }],
        "temperature": 0.2,
        "max_tokens": _DESCRIBE_MAX_TOKENS,
        **_thinking_params(disable_thinking),
    }
    try:
        resp = httpx.post(
            url, json=body, timeout=_DESCRIBE_TIMEOUT,
            headers=_auth_headers(api_key),
        )
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"could not reach {url}: {exc}") from exc
    if resp.status_code != 200:
        raise LLMUnavailable(
            f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise LLMUnavailable(f"non-JSON response from {url}: {exc}") from exc
    try:
        choice = data["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMUnavailable(f"unexpected response shape from {url}: {data!r}") from exc
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise LLMUnavailable(f"non-string content from {url}: {text!r}")
    text = _clean_description(text)
    if not text:
        raise LLMUnavailable(
            f"{url} returned no description — the model spent its whole "
            f"{_DESCRIBE_MAX_TOKENS}-token budget on reasoning. Turn off "
            "thinking for this model, or use a non-reasoning one."
        )
    if choice.get("finish_reason") == "length":
        # The answer was cut mid-sentence. A half sentence is worse than a
        # shorter one in a training caption, so drop the dangling fragment —
        # but only when a complete sentence survives.
        trimmed = _trim_to_last_sentence(text)
        if trimmed:
            text = trimmed
    return text


# Servers split the thinking pass one of two ways: a separate
# ``reasoning_content`` field (which we simply ignore) or inlined into
# ``content`` wrapped in tags — LMStudio, the default target, does the latter
# with Qwen3-style GGUFs. Strip the inline form so it never lands in a caption.
_THINK_CLOSE_RE = re.compile(r"</(?:think|thinking|reasoning)\s*>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<(?:think|thinking|reasoning)\s*>", re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Drop an inline think block, closed or not.

    Everything up to the last closing tag is reasoning (this also catches
    servers that emit the closing tag without an opening one). A leftover
    opening tag means the budget ran out mid-thought, so nothing after it is
    an answer.
    """
    closes = list(_THINK_CLOSE_RE.finditer(text))
    if closes:
        text = text[closes[-1].end():]
    opened = _THINK_OPEN_RE.search(text)
    if opened:
        text = text[: opened.start()]
    return text


def _trim_to_last_sentence(text: str) -> str:
    """Return ``text`` up to its last sentence terminator, or "" if none."""
    end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    return text[: end + 1].strip() if end >= 0 else ""


def _clean_description(text: str) -> str:
    """Collapse to a single line — LoRA caption sidecars are line-delimited."""
    lines = [ln.strip() for ln in _strip_reasoning(text).strip().splitlines() if ln.strip()]
    return " ".join(lines)


# --------------------------------------------------------------------------- #
# Tag review (vision + tool-calling)
# --------------------------------------------------------------------------- #
# A second-pass curation aid: the VLM looks at the (cropped) image, judges each
# existing tag against what's actually visible, and proposes new tags — using a
# danbooru-search tool to ground its suggestions in real, canonical tags. The
# whole exchange happens in one chat with two tools; see
# docs/validate_tag_review.py for the feasibility spike this mirrors.

# Multi-round + a verbose reasoner means the per-call read window must be
# generous; each round is a fresh request so the ceiling is per-round.
_REVIEW_TIMEOUT = httpx.Timeout(connect=5.0, read=240.0, write=30.0, pool=10.0)

# Existing tags come from WD14 (already canonical danbooru tags in space form),
# so the model only needs the search tool to validate the *new* tags it wants
# to add — re-verifying all existing tags just burns context and tool rounds.
REVIEW_SYSTEM_PROMPT = (
    "You are a meticulous dataset curator for anime-style LoRA training. You are "
    "given an image and a list of danbooru-style tags a tagger assigned to it. "
    "Work through it as follows:\n"
    "1. Look at the image carefully.\n"
    "2. For EACH existing tag, decide keep or remove based ONLY on what is "
    "visibly true in THIS image. Remove tags that are wrong, not visible, or "
    "redundant. The existing tags are ALREADY valid danbooru tags — do NOT look "
    "them up.\n"
    "3. Propose NEW tags only for clearly-visible attributes the list is missing.\n"
    "4. ONLY for a NEW tag you want to add, call search_danbooru_tags once to "
    "confirm it is a real danbooru tag and use the most popular canonical form it "
    "returns (e.g. prefer 'blonde hair' over 'yellow hair'). Tags use spaces, not "
    "underscores.\n"
    "Keep your reasoning brief. When finished you MUST call the submit_review tool "
    "with the final keep/remove/add lists. Do not write the review as prose."
)

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_danbooru_tags",
        "description": (
            "Search the danbooru tag vocabulary for real, canonical tags matching "
            "a query. Use this to confirm a NEW tag you want to add is real and to "
            "get its canonical spelling. Results are sorted by popularity. Spaces "
            "or underscores both work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Tag or partial tag, e.g. 'angel wings' or 'halo'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 8).",
                },
            },
            "required": ["query"],
        },
    },
}

_TAG_REASON_ITEMS = {
    "type": "object",
    "properties": {
        "tag": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["tag", "reason"],
}

_SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": (
            "Submit your FINAL tag review. Call this exactly once when done. Every "
            "existing tag must appear in either keep or remove; only genuinely new "
            "tags (not already in the existing list) go in add."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keep": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Existing tags that are accurate and should stay.",
                },
                "remove": {
                    "type": "array",
                    "items": _TAG_REASON_ITEMS,
                    "description": "Existing tags to delete, each with a short reason.",
                },
                "add": {
                    "type": "array",
                    "items": _TAG_REASON_ITEMS,
                    "description": "New tags for clearly-visible attributes, with reasons.",
                },
            },
            "required": ["keep", "remove", "add"],
        },
    },
}


def _post_chat(url: str, body: dict, api_key: str | None) -> dict:
    """POST one chat-completions turn; raise a friendly error on failure.

    LM Studio returns HTTP 400 with a 'context size' message when the loaded
    context window is too small for the image + tools + tool results. That's a
    config fix (load the model with a larger context), so we surface it plainly
    rather than as an opaque 400.
    """
    try:
        resp = httpx.post(url, json=body, timeout=_REVIEW_TIMEOUT, headers=_auth_headers(api_key))
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"could not reach {url}: {exc}") from exc
    if resp.status_code != 200:
        detail = resp.text[:300]
        if "context" in detail.lower() and ("size" in detail.lower() or "length" in detail.lower()):
            raise LLMUnavailable(
                "the model's loaded context window is too small for image + tools — "
                "load the model in LM Studio with a larger context length (8k+). "
                f"Server said: {detail}"
            )
        raise LLMUnavailable(f"{url} returned HTTP {resp.status_code}: {detail}")
    try:
        return resp.json()
    except ValueError as exc:
        raise LLMUnavailable(f"non-JSON response from {url}: {exc}") from exc


def _extract_json_object(text: str) -> dict | None:
    """Pull the JSON object out of a model's free-text reply.

    Tolerates a leading ``<think>`` block and prose around the object by taking
    the span from the first ``{`` to the last ``}``. Returns None if nothing
    parseable is found.
    """
    if not text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def review_tags(
    *,
    endpoint: str,
    model: str,
    image_path: Path,
    existing_tags: list[str],
    search_fn,
    api_key: str | None = None,
    max_search_rounds: int = 1,
    max_image_dim: int = 640,
    disable_thinking: bool = False,
) -> dict:
    """Run the vision + tool-calling tag review and return the model's verdict.

    ``search_fn(query: str, limit: int) -> list[dict]`` executes the danbooru
    search tool (injected so this module stays transport-only). Returns the raw
    ``{"keep": [...], "remove": [{tag, reason}], "add": [{tag, reason}]}`` the
    model submitted; the caller is responsible for reconciling it against the
    vocabulary and the on-disk tags. Raises :class:`LLMUnavailable` on transport
    errors or if the model never produces a structured review.
    """
    url = f"{_normalize_endpoint(endpoint)}/v1/chat/completions"
    messages: list[dict] = [
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Existing tags:\n" + ", ".join(existing_tags)},
                {"type": "image_url",
                 "image_url": {"url": _image_to_data_url(image_path, max_dim=max_image_dim)}},
            ],
        },
    ]
    def _message(data: dict) -> dict:
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(f"unexpected response shape from {url}: {data!r}") from exc

    def _record_assistant(msg: dict, tool_calls: list) -> None:
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            **({"tool_calls": tool_calls} if tool_calls else {}),
        })

    def _submit_from(argv: dict) -> dict:
        return {
            "keep": argv.get("keep") or [],
            "remove": argv.get("remove") or [],
            "add": argv.get("add") or [],
        }

    def _args(tc: dict) -> dict:
        try:
            return json.loads(tc.get("function", {}).get("arguments") or "{}")
        except json.JSONDecodeError:
            return {}

    # --- Phase 1: bounded exploration. The model inspects the image and may
    # call search_danbooru_tags to validate NEW tags, then submit_review. Kept
    # short on purpose: this model is verbose and, given room, will keep
    # "verifying" tags it was told to leave alone — so we cap the search budget
    # and rely on the guaranteed finalize below rather than letting it ramble.
    for _ in range(max_search_rounds):
        data = _post_chat(url, {
            "model": model, "messages": messages,
            "tools": [_SEARCH_TOOL, _SUBMIT_TOOL],
            "temperature": 0.2, "max_tokens": 2000,
            **_thinking_params(disable_thinking),
        }, api_key)
        msg = _message(data)
        tool_calls = msg.get("tool_calls") or []
        _record_assistant(msg, tool_calls)
        if not tool_calls:
            break  # nothing to submit and no search — go straight to finalize
        submitted = None
        for tc in tool_calls:
            name = tc.get("function", {}).get("name")
            if name == "submit_review":
                submitted = _submit_from(_args(tc))
                break
            argv = _args(tc)
            result = (
                search_fn(str(argv.get("query", "")), int(argv.get("limit", 8) or 8))
                if name == "search_danbooru_tags"
                else {"error": f"unknown tool {name}"}
            )
            messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                             "content": json.dumps(result, ensure_ascii=False)})
        if submitted is not None:
            return submitted

    # --- Phase 2: guaranteed finalize WITHOUT tools. Forcing a specific tool is
    # unreliable here — this model ignores a restricted tool list and emits
    # phantom search calls, so a "required" submit_review can loop forever. By
    # sending NO tools, the model physically cannot call one and must return the
    # verdict as JSON text, which we parse. This bounds a whole review to
    # max_search_rounds + 1 model calls — no unbounded loop is possible.
    messages.append({
        "role": "user",
        "content": (
            "Stop. Do not search any further. Reply with ONLY a JSON object and no "
            "other text, in exactly this shape:\n"
            '{"keep": ["tag", ...], "remove": [{"tag": "...", "reason": "..."}], '
            '"add": [{"tag": "...", "reason": "..."}]}'
        ),
    })
    data = _post_chat(url, {
        "model": model, "messages": messages,
        "temperature": 0.2, "max_tokens": 2000,
        **_thinking_params(disable_thinking),
    }, api_key)
    parsed = _extract_json_object(_message(data).get("content") or "")
    if parsed is None:
        raise LLMUnavailable("the model did not return a parseable tag review")
    return _submit_from(parsed)
