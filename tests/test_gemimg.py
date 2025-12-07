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


class TestInputImageLimit:
    """Tests for the input image count validation."""

    def test_gemini3_allows_14_input_images(self, api_key):
        """Gemini 3 Pro should allow up to 14 input images."""
        gem = GemImg(api_key=api_key, model="gemini-3-pro-image-preview")
        # Create 14 test images
        imgs = [Image.new("RGB", (100, 100), color="red") for _ in range(14)]

        # This should not raise an error when we implement validation
        # For now, we're just setting up the test
        with patch.object(gem, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "responseId": "test",
                "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
            }
            mock_client.post.return_value = mock_response

            # Should not raise ValueError for 14 images on Gemini 3
            try:
                gem.generate(prompt="test", imgs=imgs)
            except ValueError as e:
                if "Maximum" in str(e) and "input images" in str(e):
                    pytest.fail(f"Should allow 14 images for Gemini 3: {e}")
                raise

    def test_gemini3_rejects_15_input_images(self, api_key):
        """Gemini 3 Pro should reject more than 14 input images."""
        gem = GemImg(api_key=api_key, model="gemini-3-pro-image-preview")
        imgs = [Image.new("RGB", (100, 100), color="red") for _ in range(15)]

        with pytest.raises(ValueError, match=r"Maximum 14 input images"):
            gem.generate(prompt="test", imgs=imgs)

    def test_flash_model_allows_6_input_images(self, api_key):
        """Flash model should allow up to 6 input images."""
        gem = GemImg(api_key=api_key, model="gemini-2.5-flash-image")
        imgs = [Image.new("RGB", (100, 100), color="red") for _ in range(6)]

        with patch.object(gem, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "responseId": "test",
                "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
            }
            mock_client.post.return_value = mock_response

            try:
                gem.generate(prompt="test", imgs=imgs)
            except ValueError as e:
                if "Maximum" in str(e) and "input images" in str(e):
                    pytest.fail(f"Should allow 6 images for Flash: {e}")
                raise

    def test_flash_model_rejects_7_input_images(self, api_key):
        """Flash model should reject more than 6 input images."""
        gem = GemImg(api_key=api_key, model="gemini-2.5-flash-image")
        imgs = [Image.new("RGB", (100, 100), color="red") for _ in range(7)]

        with pytest.raises(ValueError, match=r"Maximum 6 input images"):
            gem.generate(prompt="test", imgs=imgs)


class TestGoogleSearchGrounding:
    """Tests for Google Search grounding parameter."""

    def test_google_search_included_in_request(self, api_key):
        """Google Search tool should be included in API request when enabled."""
        gem = GemImg(api_key=api_key, model="gemini-3-pro-image-preview")
        valid_b64 = create_valid_b64_image()

        with patch.object(gem, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "responseId": "test",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": valid_b64,
                                    }
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
            }
            mock_client.post.return_value = mock_response

            gem.generate(prompt="current weather in Paris", google_search=True)

            # Check that the API was called with googleSearch tool
            call_args = mock_client.post.call_args
            json_payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "tools" in json_payload
            assert {"googleSearch": {}} in json_payload["tools"]

    def test_google_search_requires_gemini3(self, api_key):
        """Google Search should raise error for non-Gemini 3 models."""
        gem = GemImg(api_key=api_key, model="gemini-2.5-flash-image")

        with pytest.raises(
            ValueError, match=r"Google Search grounding requires a Gemini 3 model"
        ):
            gem.generate(prompt="test", google_search=True)

    def test_google_search_disabled_by_default(self, api_key):
        """Google Search should not be included when not explicitly enabled."""
        gem = GemImg(api_key=api_key, model="gemini-3-pro-image-preview")
        valid_b64 = create_valid_b64_image()

        with patch.object(gem, "client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "responseId": "test",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": valid_b64,
                                    }
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
            }
            mock_client.post.return_value = mock_response

            gem.generate(prompt="test")

            # Check that tools is not in the request
            call_args = mock_client.post.call_args
            json_payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "tools" not in json_payload


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
