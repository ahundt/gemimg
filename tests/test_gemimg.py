"""Tests for the GemImg class - Gemini 3 Pro Image support."""

import base64
import io
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from gemimg import GemImg


def create_valid_b64_image():
    """Create a valid base64-encoded PNG image for testing."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class TestIsGemini3Property:
    """Tests for the is_gemini3 property detection."""

    def test_is_gemini3_with_gemini3_pro_image_preview(self, api_key):
        """Should return True for gemini-3-pro-image-preview model."""
        gem = GemImg(api_key=api_key, model="gemini-3-pro-image-preview")
        assert gem.is_gemini3 is True

    def test_is_gemini3_with_gemini3_prefix(self, api_key):
        """Should return True for any model starting with gemini-3."""
        gem = GemImg(api_key=api_key, model="gemini-3-pro")
        assert gem.is_gemini3 is True

    def test_is_gemini3_with_flash_model(self, api_key):
        """Should return False for gemini-2.5-flash-image model."""
        gem = GemImg(api_key=api_key, model="gemini-2.5-flash-image")
        assert gem.is_gemini3 is False

    def test_is_gemini3_with_default_model(self, api_key):
        """Should return False for default model (gemini-2.5-flash-image)."""
        gem = GemImg(api_key=api_key)
        assert gem.is_gemini3 is False

    def test_is_gemini3_with_pro_2_model(self, api_key):
        """Should return False for gemini-2.0-pro model."""
        gem = GemImg(api_key=api_key, model="gemini-2.0-pro-image")
        assert gem.is_gemini3 is False


class TestGeminiVersion:
    """Tests for the gemini_version property (robust version detection)."""

    def test_gemini_version_extracts_2_from_flash(self, api_key):
        """Should extract version 2 from gemini-2.5-flash-image."""
        gem = GemImg(api_key=api_key, model="gemini-2.5-flash-image")
        assert gem.gemini_version == 2

    def test_gemini_version_extracts_3_from_pro(self, api_key):
        """Should extract version 3 from gemini-3-pro-image-preview."""
        gem = GemImg(api_key=api_key, model="gemini-3-pro-image-preview")
        assert gem.gemini_version == 3

    def test_gemini_version_extracts_2_from_2_0_flash(self, api_key):
        """Should extract version 2 from gemini-2.0-flash-exp."""
        gem = GemImg(api_key=api_key, model="gemini-2.0-flash-exp")
        assert gem.gemini_version == 2

    def test_gemini_version_returns_none_for_unknown(self, api_key):
        """Should return None for non-Gemini models."""
        gem = GemImg(api_key=api_key, model="some-other-model")
        assert gem.gemini_version is None

    def test_gemini_version_handles_future_version_4(self, api_key):
        """Should extract version 4 for hypothetical future model."""
        gem = GemImg(api_key=api_key, model="gemini-4-ultra-image")
        assert gem.gemini_version == 4

    def test_is_gemini3_uses_gemini_version(self, api_key):
        """is_gemini3 should use gemini_version >= 3 check."""
        gem3 = GemImg(api_key=api_key, model="gemini-3-pro-image-preview")
        gem2 = GemImg(api_key=api_key, model="gemini-2.5-flash-image")
        gem4 = GemImg(api_key=api_key, model="gemini-4-ultra-image")

        assert gem3.is_gemini3 is True
        assert gem2.is_gemini3 is False
        assert gem4.is_gemini3 is True  # Future versions should also be True


class TestThinkingModeFiltering:
    """Tests for filtering out Gemini 3 thinking mode interim images."""

    def test_filters_out_thinking_images(self, api_key):
        """Should filter out images with thought=True flag."""
        gem = GemImg(api_key=api_key, model="gemini-3-pro-image-preview")
        valid_b64 = create_valid_b64_image()

        with patch.object(gem, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            # Simulate Gemini 3 thinking mode response with 2 thought images + 1 final
            mock_response.json.return_value = {
                "responseId": "test",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                # First thinking image (should be filtered)
                                {
                                    "inlineData": {"mimeType": "image/png", "data": valid_b64},
                                    "thought": True,
                                },
                                # Second thinking image (should be filtered)
                                {
                                    "inlineData": {"mimeType": "image/png", "data": valid_b64},
                                    "thought": True,
                                },
                                # Final image (should be kept)
                                {
                                    "inlineData": {"mimeType": "image/png", "data": valid_b64},
                                },
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
            }
            mock_client.post.return_value = mock_response

            result = gem.generate(prompt="test", save=False)

            # Should only return 1 image (the final one without thought=True)
            assert result is not None
            assert len(result.images) == 1

    def test_keeps_images_without_thought_flag(self, api_key):
        """Should keep images that don't have the thought flag."""
        gem = GemImg(api_key=api_key, model="gemini-2.5-flash-image")
        valid_b64 = create_valid_b64_image()

        with patch.object(gem, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            # Standard response without any thought flags
            mock_response.json.return_value = {
                "responseId": "test",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"inlineData": {"mimeType": "image/png", "data": valid_b64}},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
            }
            mock_client.post.return_value = mock_response

            result = gem.generate(prompt="test", save=False)

            assert result is not None
            assert len(result.images) == 1

    def test_keeps_images_with_thought_false(self, api_key):
        """Should keep images with explicit thought=False."""
        gem = GemImg(api_key=api_key, model="gemini-3-pro-image-preview")
        valid_b64 = create_valid_b64_image()

        with patch.object(gem, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "responseId": "test",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {"mimeType": "image/png", "data": valid_b64},
                                    "thought": False,
                                },
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
            }
            mock_client.post.return_value = mock_response

            result = gem.generate(prompt="test", save=False)

            assert result is not None
            assert len(result.images) == 1


class TestIsProProperty:
    """Tests for the existing is_pro property (ensure no regression)."""

    def test_is_pro_with_pro_model(self, api_key):
        """Should return True for pro models."""
        gem = GemImg(api_key=api_key, model="gemini-2.0-pro-image")
        assert gem.is_pro is True

    def test_is_pro_with_gemini3_pro(self, api_key):
        """Should return True for gemini-3-pro-image-preview."""
        gem = GemImg(api_key=api_key, model="gemini-3-pro-image-preview")
        assert gem.is_pro is True

    def test_is_pro_with_flash_model(self, api_key):
        """Should return False for flash models."""
        gem = GemImg(api_key=api_key, model="gemini-2.5-flash-image")
        assert gem.is_pro is False


class TestNetworkLevelMocking:
    """Network-level tests using respx to mock HTTP requests.

    These tests verify the full request/response pipeline without
    making real API calls, using deterministic checkerboard images.
    """

    def test_generate_with_respx_mock(self, mock_gemini_api, api_key):
        """GemImg.generate() should work with mocked API responses."""
        gem = GemImg(api_key=api_key)
        result = gem.generate(prompt="checkerboard pattern", save=False)

        assert result is not None
        assert result.images is not None
        assert len(result.images) == 1
        assert result.images[0].size == (5, 5)
        assert mock_gemini_api.called

    def test_generate_verifies_checkerboard_pattern(self, mock_gemini_api, api_key):
        """Mock response should return verifiable checkerboard pattern."""
        gem = GemImg(api_key=api_key)
        result = gem.generate(prompt="test", save=False)

        # Verify checkerboard pattern: black at (0,0), white at (1,0)
        pixels = result.images[0].load()
        assert pixels[0, 0] == (0, 0, 0)  # Black (top-left)
        assert pixels[1, 0] == (255, 255, 255)  # White
        assert pixels[0, 1] == (255, 255, 255)  # White
        assert pixels[1, 1] == (0, 0, 0)  # Black

    def test_generate_1024_with_respx_mock(self, mock_gemini_api_1024, api_key):
        """GemImg.generate() with 1024x1024 mock response."""
        gem = GemImg(api_key=api_key)
        result = gem.generate(prompt="large icon", save=False)

        assert result is not None
        assert result.images[0].size == (1024, 1024)

    def test_mock_returns_correct_response_metadata(self, mock_gemini_api, api_key):
        """Mock response should include proper metadata structure."""
        gem = GemImg(api_key=api_key)
        result = gem.generate(prompt="test", save=False)

        # GemImgResult should have images from mocked response
        assert result.images is not None
        assert len(result.images) >= 1
