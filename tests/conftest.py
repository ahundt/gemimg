"""Pytest configuration and fixtures for gemimg tests.

Provides network-level mocking for Gemini API using respx.
No real API calls are made during tests.
"""

import base64
import io
from pathlib import Path

import pytest
import respx
from httpx import Response
from PIL import Image


@pytest.fixture
def api_key():
    """Provide a test API key."""
    return "test-api-key-12345"


# =============================================================================
# Checkerboard Test Images (Deterministic, Trivially Verifiable)
# =============================================================================


@pytest.fixture
def checkerboard_5x5() -> Image.Image:
    """Generate deterministic 5x5 checkerboard test image.

    Checkerboard pattern:
    B W B W B
    W B W B W
    B W B W B
    W B W B W
    B W B W B

    Where B=black (0,0,0), W=white (255,255,255)
    """
    img = Image.new("RGB", (5, 5), "white")
    pixels = img.load()
    for y in range(5):
        for x in range(5):
            if (x + y) % 2 == 0:
                pixels[x, y] = (0, 0, 0)  # Black
            else:
                pixels[x, y] = (255, 255, 255)  # White
    return img


@pytest.fixture
def checkerboard_b64(checkerboard_5x5) -> str:
    """Base64-encoded PNG of 5x5 checkerboard for mock API response."""
    buffer = io.BytesIO()
    checkerboard_5x5.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture
def checkerboard_1024x1024() -> Image.Image:
    """1024x1024 checkerboard for icon tests (matches Gemini output size).

    Uses 8x8 grid of 128px cells for visual clarity.
    """
    img = Image.new("RGB", (1024, 1024), "white")
    pixels = img.load()
    cell_size = 128  # 8x8 grid of 128px cells
    for y in range(1024):
        for x in range(1024):
            cell_x, cell_y = x // cell_size, y // cell_size
            if (cell_x + cell_y) % 2 == 0:
                pixels[x, y] = (0, 0, 0)
    return img


@pytest.fixture
def checkerboard_1024_b64(checkerboard_1024x1024) -> str:
    """Base64-encoded PNG of 1024x1024 checkerboard for mock API response."""
    buffer = io.BytesIO()
    checkerboard_1024x1024.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# =============================================================================
# Gemini API Mock Responses
# =============================================================================


@pytest.fixture
def mock_gemini_image_response(checkerboard_b64):
    """Mock Gemini image generation API response (single image).

    Matches actual API format from gemimg/gemimg.py response parsing.
    """
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": checkerboard_b64,
                            }
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 100,
            "totalTokenCount": 110,
        },
        "responseId": "test-response-id-12345",
    }


@pytest.fixture
def mock_gemini_1024_response(checkerboard_1024_b64):
    """Mock Gemini image generation API response with 1024x1024 image."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": checkerboard_1024_b64,
                            }
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 100,
            "totalTokenCount": 110,
        },
        "responseId": "test-response-1024-12345",
    }


# =============================================================================
# respx Mock Fixtures for Gemini API
# =============================================================================


@pytest.fixture
def mock_gemini_api(mock_gemini_image_response):
    """respx mock for Gemini generateContent endpoint (5x5 checkerboard).

    Intercepts: POST https://generativelanguage.googleapis.com/v1beta/models/*:generateContent
    Used by: gemimg/gemimg.py API calls
    """
    with respx.mock:
        route = respx.post(
            url__regex=r"https://generativelanguage\.googleapis\.com/v1beta/models/.+:generateContent"
        ).mock(return_value=Response(200, json=mock_gemini_image_response))
        yield route


@pytest.fixture
def mock_gemini_api_1024(mock_gemini_1024_response):
    """respx mock for Gemini generateContent endpoint (1024x1024 checkerboard).

    Use this for icon generation tests that expect 1024x1024 output.
    """
    with respx.mock:
        route = respx.post(
            url__regex=r"https://generativelanguage\.googleapis\.com/v1beta/models/.+:generateContent"
        ).mock(return_value=Response(200, json=mock_gemini_1024_response))
        yield route


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Provide a temporary output directory for icon generation tests."""
    output_dir = tmp_path / "icons_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
