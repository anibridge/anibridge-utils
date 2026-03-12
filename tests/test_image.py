"""Tests for shared image helpers."""

import base64

import pytest

from anibridge.utils.image import fetch_image_as_data_url


def test_fetch_image_as_data_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Data URL helper should encode payload and preserve content type."""
    payload = b"abc123"

    class DummyResponse:
        headers = {"Content-Type": "image/png"}
        content = payload

        @staticmethod
        def raise_for_status() -> None:
            return None

    called = {}

    def fake_get(url: str, *, headers: dict[str, str], timeout: float):
        called["url"] = url
        called["headers"] = headers
        called["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("anibridge.utils.image.requests.get", fake_get)

    result = fetch_image_as_data_url(
        "https://example.com/poster",
        headers={"Authorization": "Bearer token"},
        timeout=1.5,
    )

    expected_b64 = base64.b64encode(payload).decode("utf-8")
    assert result == f"data:image/png;base64,{expected_b64}"
    assert called == {
        "url": "https://example.com/poster",
        "headers": {"Authorization": "Bearer token"},
        "timeout": 1.5,
    }
