"""Shared image helpers for AniBridge providers."""

import base64
from collections.abc import Mapping

import requests

__all__ = ["fetch_image_as_data_url"]


def fetch_image_as_data_url(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 3,
    default_mime: str = "image/jpeg",
) -> str:
    """Fetch an image over HTTP and return a base64 data URL.

    Args:
        url (str): The URL of the image to fetch.
        headers (Mapping[str, str] | None): Optional HTTP headers to include in the request.
        timeout (float): Timeout for the HTTP request in seconds.
        default_mime (str): Default MIME type to use if the response does not specify one.

    Returns:
        str: A data URL containing the base64-encoded image.

    Raises:
        requests.RequestException: If HTTP request fails.
    """
    response = requests.get(url, headers=dict(headers or {}), timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", default_mime)
    encoded = base64.b64encode(response.content).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"
