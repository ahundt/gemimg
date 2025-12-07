"""Pytest configuration and fixtures for gemimg tests."""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import io
import base64


@pytest.fixture
def api_key():
    """Provide a test API key."""
    return "test-api-key-12345"


@pytest.fixture
def mock_httpx_client():
    """Provide a mocked httpx.Client."""
    with patch("httpx.Client") as mock_client:
        yield mock_client


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    img = Image.new("RGB", (100, 100), color="red")
    return img


@pytest.fixture
def sample_image_b64(sample_image):
    """Create a base64-encoded test image."""
    buffer = io.BytesIO()
    sample_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture
def mock_successful_response(sample_image_b64):
    """Create a mock successful API response."""
    return {
        "responseId": "test-response-id",
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": sample_image_b64,
                            }
                        }
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
        },
    }


@pytest.fixture
def mock_error_response():
    """Create a mock error API response."""
    return {
        "error": {
            "code": 400,
            "message": "Invalid request",
        }
    }


@pytest.fixture
def gemimg_instance(api_key, mock_httpx_client):
    """Create a GemImg instance with mocked client."""
    from gemimg import GemImg

    return GemImg(api_key=api_key)
