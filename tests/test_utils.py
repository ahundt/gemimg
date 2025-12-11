"""Tests for utility functions."""

import pytest
from PIL import Image

from gemimg.utils import _validate_aspect, resize_image


class TestValidateAspect:
    """Tests for aspect ratio validation."""

    def test_valid_standard_aspect_ratios(self):
        """Standard aspect ratios should be valid for all models."""
        valid_ratios = ["1:1", "3:4", "4:3", "9:16", "16:9"]
        for ratio in valid_ratios:
            assert _validate_aspect(ratio, is_pro=False) == ratio
            assert _validate_aspect(ratio, is_pro=True) == ratio

    def test_all_aspect_ratios_valid_for_both_models(self):
        """All aspect ratios should be valid for both flash and pro models."""
        all_ratios = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
        for ratio in all_ratios:
            # Should work for both flash and pro
            assert _validate_aspect(ratio, is_pro=False) == ratio
            assert _validate_aspect(ratio, is_pro=True) == ratio

    def test_invalid_aspect_ratio(self):
        """Invalid aspect ratios should raise ValueError."""
        with pytest.raises(ValueError):
            _validate_aspect("7:3", is_pro=False)
        with pytest.raises(ValueError):
            _validate_aspect("7:3", is_pro=True)


class TestResizeImage:
    """Tests for image resizing."""

    def test_resize_preserves_aspect_ratio(self):
        """Resize should preserve aspect ratio."""
        img = Image.new("RGB", (200, 100), color="red")
        resized = resize_image(img, 100)
        # Should maintain 2:1 aspect ratio
        assert resized.width == 100
        assert resized.height == 50

    def test_resize_square_image(self):
        """Resize square image should produce square result."""
        img = Image.new("RGB", (500, 500), color="blue")
        resized = resize_image(img, 100)
        assert resized.width == 100
        assert resized.height == 100
