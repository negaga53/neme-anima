"""Tests for /api/llm/discover-models — connectivity probe to LMStudio etc."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from neme_anima.server.app import create_app


@dataclass
class _FakeResponse:
    status_code: int
    _payload: dict | None = None
    text: str = ""

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


@pytest.fixture
def app(tmp_path: Path):
    return create_app(state_dir=tmp_path / "state")


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_discover_models_returns_sorted_list(client, monkeypatch):
    def _fake_get(url, timeout=None, headers=None):
        assert url == "http://localhost:1234/v1/models"
        # No api_key in the request -> no Authorization header attached.
        assert not (headers or {}).get("Authorization")
        return _FakeResponse(
            status_code=200,
            _payload={"data": [
                {"id": "qwen2-vl-7b"},
                {"id": "llava-1.6-mistral"},
            ]},
        )

    monkeypatch.setattr("neme_anima.llm.httpx.get", _fake_get)
    resp = await client.post(
        "/api/llm/discover-models",
        json={"endpoint": "http://localhost:1234"},
    )
    assert resp.status_code == 200
    assert resp.json()["models"] == ["llava-1.6-mistral", "qwen2-vl-7b"]


async def test_discover_models_forwards_api_key(client, monkeypatch):
    seen: dict = {}

    def _fake_get(url, timeout=None, headers=None):
        seen["headers"] = headers or {}
        return _FakeResponse(status_code=200, _payload={"data": []})

    monkeypatch.setattr("neme_anima.llm.httpx.get", _fake_get)
    resp = await client.post(
        "/api/llm/discover-models",
        json={"endpoint": "https://api.openai.com", "api_key": "sk-test"},
    )
    assert resp.status_code == 200
    assert seen["headers"].get("Authorization") == "Bearer sk-test"


async def test_discover_models_422_on_unreachable(client, monkeypatch):
    def _fake_get(url, timeout=None, headers=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("neme_anima.llm.httpx.get", _fake_get)
    resp = await client.post(
        "/api/llm/discover-models",
        json={"endpoint": "http://nope:9999"},
    )
    assert resp.status_code == 422
    assert "could not reach" in resp.json()["detail"]


async def test_discover_models_422_on_blank_endpoint(client):
    resp = await client.post(
        "/api/llm/discover-models", json={"endpoint": "   "},
    )
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# describe_image response handling
# --------------------------------------------------------------------------- #
# Reasoning models split their thinking pass out into `reasoning_content` or
# inline it in `content`, and either way it competes with the caption for the
# token budget. These cover both shapes plus the truncation fallout.


def _describe(monkeypatch, *, content, finish_reason="stop"):
    """Run describe_image against a canned chat-completions payload."""
    from neme_anima import llm

    payload = {
        "choices": [{"finish_reason": finish_reason,
                     "message": {"content": content, "role": "assistant"}}]
    }
    monkeypatch.setattr(
        llm.httpx, "post",
        lambda *a, **k: _FakeResponse(200, payload),
    )
    return llm.describe_image(
        endpoint="http://x", model="m", image_path=Path("/nonexistent.png"),
    )


@pytest.fixture(autouse=False)
def _no_image_read(monkeypatch):
    from neme_anima import llm
    monkeypatch.setattr(llm, "_image_to_data_url", lambda *a, **k: "data:image/png;base64,x")


def test_describe_strips_inline_think_block(monkeypatch, _no_image_read):
    out = _describe(
        monkeypatch,
        content="<think>The tags say brown hair but I see none.</think>A girl in a red dress.",
    )
    assert out == "A girl in a red dress."


def test_describe_strips_unclosed_think_block(monkeypatch, _no_image_read):
    """A closing tag with no opening one still marks everything before it as
    reasoning — some servers only emit the close."""
    out = _describe(monkeypatch, content="Let me look carefully.</think>A cat.")
    assert out == "A cat."


def test_describe_rejects_reasoning_only_answer(monkeypatch, _no_image_read):
    """Budget consumed mid-thought: there is no caption, so fail loudly rather
    than writing an empty description to the sidecar."""
    from neme_anima.llm import LLMUnavailable

    with pytest.raises(LLMUnavailable, match="no description"):
        _describe(monkeypatch, content="<think>Still thinking about the",
                  finish_reason="length")


def test_describe_rejects_empty_content(monkeypatch, _no_image_read):
    from neme_anima.llm import LLMUnavailable

    with pytest.raises(LLMUnavailable, match="no description"):
        _describe(monkeypatch, content="", finish_reason="length")


def test_describe_trims_truncated_tail_to_last_sentence(monkeypatch, _no_image_read):
    out = _describe(
        monkeypatch,
        content="A girl in a red dress. She stands against a dark purple backg",
        finish_reason="length",
    )
    assert out == "A girl in a red dress."


def test_describe_keeps_fragment_when_no_sentence_completed(monkeypatch, _no_image_read):
    """Nothing complete to fall back to — a partial caption still beats none."""
    out = _describe(monkeypatch, content="A girl in a red dre", finish_reason="length")
    assert out == "A girl in a red dre"


def test_describe_collapses_multiline_to_one_line(monkeypatch, _no_image_read):
    out = _describe(monkeypatch, content="A girl in a red dress.\n\nDark background.")
    assert out == "A girl in a red dress. Dark background."


def test_describe_omits_thinking_field_by_default(monkeypatch, _no_image_read):
    """The suppression field is non-standard enough that a strict server 400s
    on it, so it must never ride along unasked."""
    from neme_anima import llm

    sent: dict = {}

    def fake_post(url, **kwargs):
        sent.update(kwargs["json"])
        return _FakeResponse(200, {"choices": [
            {"finish_reason": "stop", "message": {"content": "A cat."}}]})

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    llm.describe_image(endpoint="http://x", model="m", image_path=Path("/n.png"))
    assert "reasoning_effort" not in sent

    sent.clear()
    llm.describe_image(endpoint="http://x", model="m", image_path=Path("/n.png"),
                       disable_thinking=True)
    assert sent["reasoning_effort"] == "none"
