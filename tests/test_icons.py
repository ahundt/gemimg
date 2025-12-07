"""Tests for the icons module."""

import pytest
from pathlib import Path
from PIL import Image
import tempfile
import json

from gemimg.icons import (
    Platform,
    Preset,
    IconSpec,
    IconType,
    IconGenerator,
    IconGeneratorConfig,
    IconVariants,
)
from gemimg.icons.constants import (
    PRESET_PLATFORMS,
    IOS_ICON_SPECS,
    MACOS_ICON_SPECS,
    ANDROID_ICON_SPECS,
    WINDOWS_ICO_SIZES,
    PWA_ICON_SPECS,
)
from gemimg.icons.platforms import (
    IOSProcessor,
    MacOSProcessor,
    AndroidProcessor,
    WindowsProcessor,
    PWAProcessor,
    get_processor,
)


@pytest.fixture
def sample_icon():
    """Create a sample 1024x1024 icon image."""
    return Image.new("RGBA", (1024, 1024), color=(255, 0, 0, 255))


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPlatformEnum:
    """Tests for Platform enum."""

    def test_all_platforms_defined(self):
        """All expected platforms should be defined."""
        assert Platform.IOS.value == "ios"
        assert Platform.MACOS.value == "macos"
        assert Platform.ANDROID.value == "android"
        assert Platform.WINDOWS.value == "windows"
        assert Platform.PWA.value == "pwa"


class TestPresetEnum:
    """Tests for Preset enum and mappings."""

    def test_mobile_preset(self):
        """Mobile preset should include iOS and Android."""
        platforms = PRESET_PLATFORMS[Preset.MOBILE]
        assert Platform.IOS in platforms
        assert Platform.ANDROID in platforms
        assert len(platforms) == 2

    def test_desktop_preset(self):
        """Desktop preset should include macOS and Windows."""
        platforms = PRESET_PLATFORMS[Preset.DESKTOP]
        assert Platform.MACOS in platforms
        assert Platform.WINDOWS in platforms
        assert len(platforms) == 2

    def test_apple_preset(self):
        """Apple preset should include iOS and macOS."""
        platforms = PRESET_PLATFORMS[Preset.APPLE]
        assert Platform.IOS in platforms
        assert Platform.MACOS in platforms
        assert len(platforms) == 2

    def test_all_preset(self):
        """All preset should include all 5 platforms."""
        platforms = PRESET_PLATFORMS[Preset.ALL]
        assert len(platforms) == 5
        assert Platform.IOS in platforms
        assert Platform.MACOS in platforms
        assert Platform.ANDROID in platforms
        assert Platform.WINDOWS in platforms
        assert Platform.PWA in platforms


class TestIconSpec:
    """Tests for IconSpec dataclass."""

    def test_size_property(self):
        """Size property should return (width, height) tuple."""
        spec = IconSpec(512, 512)
        assert spec.size == (512, 512)

    def test_scaled_size_property(self):
        """Scaled size should account for scale factor."""
        spec = IconSpec(256, 256, scale=2)
        assert spec.scaled_size == (512, 512)


class TestIOSProcessor:
    """Tests for iOS icon processor."""

    def test_process_creates_all_sizes(self, sample_icon, temp_output_dir):
        """iOS processor should create all required icon sizes."""
        processor = IOSProcessor(temp_output_dir)
        result = processor.process(sample_icon)

        assert len(result) == len(IOS_ICON_SPECS)
        for filename, img in result.items():
            assert filename.startswith("AppIcon-")
            assert filename.endswith(".png")

    def test_app_icon_removes_transparency(self, temp_output_dir):
        """App icon mode should convert RGBA to RGB."""
        # Create image with transparency
        rgba_icon = Image.new("RGBA", (1024, 1024), color=(255, 0, 0, 128))

        processor = IOSProcessor(temp_output_dir, icon_type=IconType.APP_ICON)
        result = processor.process(rgba_icon)

        # Check first result is RGB
        first_img = list(result.values())[0]
        assert first_img.mode == "RGB"

    def test_menu_icon_preserves_transparency(self, temp_output_dir):
        """Menu icon mode should preserve transparency."""
        rgba_icon = Image.new("RGBA", (1024, 1024), color=(255, 0, 0, 128))

        processor = IOSProcessor(temp_output_dir, icon_type=IconType.MENU_ICON)
        result = processor.process(rgba_icon)

        first_img = list(result.values())[0]
        assert first_img.mode == "RGBA"


class TestMacOSProcessor:
    """Tests for macOS icon processor."""

    def test_process_creates_all_sizes(self, sample_icon, temp_output_dir):
        """macOS processor should create all required sizes including @2x."""
        processor = MacOSProcessor(temp_output_dir, apply_shadow=False)
        result = processor.process(sample_icon)

        assert len(result) == len(MACOS_ICON_SPECS)

        # Check for @2x variants
        filenames = list(result.keys())
        assert any("@2x" in f for f in filenames)


class TestAndroidProcessor:
    """Tests for Android icon processor."""

    def test_process_creates_all_densities(self, sample_icon, temp_output_dir):
        """Android processor should create all density variants."""
        processor = AndroidProcessor(temp_output_dir, validate_safe_zone=False)
        result = processor.process(sample_icon)

        assert len(result) == len(ANDROID_ICON_SPECS)

        # Check for density suffixes
        filenames = list(result.keys())
        assert any("xxxhdpi" in f for f in filenames)
        assert any("mdpi" in f for f in filenames)

    def test_tinted_variant_creates_monochrome(self, sample_icon, temp_output_dir):
        """Tinted variant should create monochrome icon."""
        processor = AndroidProcessor(
            temp_output_dir, variant="tinted", validate_safe_zone=False
        )
        result = processor.process(sample_icon)

        assert "ic_launcher_monochrome.png" in result


class TestWindowsProcessor:
    """Tests for Windows ICO processor."""

    def test_save_creates_ico_file(self, sample_icon, temp_output_dir):
        """Windows processor should create multi-size ICO file."""
        processor = WindowsProcessor(temp_output_dir)
        processed = processor.process(sample_icon)
        saved = processor.save(processed)

        assert len(saved) == 1
        assert saved[0].suffix == ".ico"
        assert saved[0].exists()


class TestPWAProcessor:
    """Tests for PWA icon processor."""

    def test_process_creates_all_sizes(self, sample_icon, temp_output_dir):
        """PWA processor should create all required sizes."""
        processor = PWAProcessor(temp_output_dir)
        result = processor.process(sample_icon)

        assert len(result) == len(PWA_ICON_SPECS)

    def test_maskable_icons_have_padding(self, sample_icon, temp_output_dir):
        """Maskable icons should have padding applied."""
        processor = PWAProcessor(temp_output_dir)
        result = processor.process(sample_icon)

        # Check that maskable icons exist
        maskable_files = [f for f in result.keys() if "maskable" in f]
        assert len(maskable_files) > 0

    def test_save_creates_manifest(self, sample_icon, temp_output_dir):
        """PWA processor should create manifest-icons.json."""
        processor = PWAProcessor(temp_output_dir)
        processed = processor.process(sample_icon)
        saved = processor.save(processed)

        manifest_path = temp_output_dir / "pwa" / "light" / "manifest-icons.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)
            assert isinstance(manifest, list)
            assert len(manifest) > 0
            assert all("src" in item for item in manifest)
            assert all("sizes" in item for item in manifest)


class TestGetProcessor:
    """Tests for processor factory function."""

    def test_returns_correct_processor_types(self, temp_output_dir):
        """Factory should return correct processor type for each platform."""
        assert isinstance(get_processor(Platform.IOS, temp_output_dir), IOSProcessor)
        assert isinstance(get_processor(Platform.MACOS, temp_output_dir), MacOSProcessor)
        assert isinstance(get_processor(Platform.ANDROID, temp_output_dir), AndroidProcessor)
        assert isinstance(get_processor(Platform.WINDOWS, temp_output_dir), WindowsProcessor)
        assert isinstance(get_processor(Platform.PWA, temp_output_dir), PWAProcessor)


class TestIconVariants:
    """Tests for IconVariants dataclass."""

    def test_default_values(self):
        """Default values should be None."""
        variants = IconVariants()
        assert variants.light is None
        assert variants.dark is None
        assert variants.tinted is None

    def test_with_images(self, sample_icon):
        """Should accept image instances."""
        variants = IconVariants(light=sample_icon, dark=sample_icon)
        assert variants.light is not None
        assert variants.dark is not None
        assert variants.tinted is None

    def test_has_any_variant_true(self, sample_icon):
        """Should return True when at least one variant exists."""
        variants = IconVariants(light=sample_icon)
        assert variants.has_any_variant() is True

    def test_has_any_variant_false(self):
        """Should return False when no variants exist."""
        variants = IconVariants()
        assert variants.has_any_variant() is False

    def test_count_variants(self, sample_icon):
        """Should count non-None variants correctly."""
        assert IconVariants().count() == 0
        assert IconVariants(light=sample_icon).count() == 1
        assert IconVariants(light=sample_icon, dark=sample_icon).count() == 2
        assert IconVariants(light=sample_icon, dark=sample_icon, tinted=sample_icon).count() == 3


class TestIconGeneratorConfig:
    """Tests for IconGeneratorConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = IconGeneratorConfig()
        assert config.icon_type == IconType.APP_ICON
        assert len(config.platforms) == 5  # ALL preset
        assert config.themed is False
        assert config.macos_shadow is True
        assert config.validate_safe_zone is True

    def test_custom_values(self):
        """Config should accept custom values."""
        config = IconGeneratorConfig(
            icon_type=IconType.MENU_ICON,
            platforms={Platform.IOS, Platform.ANDROID},
            themed=True,
        )
        assert config.icon_type == IconType.MENU_ICON
        assert len(config.platforms) == 2
        assert config.themed is True

    def test_raises_on_empty_platforms(self):
        """Should raise ValueError when platforms is empty."""
        with pytest.raises(ValueError, match="At least one platform must be specified"):
            IconGeneratorConfig(platforms=set())

    def test_image_size_validation(self):
        """Should validate image_size is 1K, 2K, or 4K."""
        # Valid sizes should work
        assert IconGeneratorConfig(image_size="1K").image_size == "1K"
        assert IconGeneratorConfig(image_size="2K").image_size == "2K"
        assert IconGeneratorConfig(image_size="4K").image_size == "4K"

        # Invalid sizes should raise
        with pytest.raises(ValueError, match="image_size must be"):
            IconGeneratorConfig(image_size="invalid")

    def test_temperature_validation(self):
        """Should validate temperature is between 0.0 and 2.0."""
        # Valid temperatures should work
        assert IconGeneratorConfig(temperature=0.0).temperature == 0.0
        assert IconGeneratorConfig(temperature=1.0).temperature == 1.0
        assert IconGeneratorConfig(temperature=2.0).temperature == 2.0

        # Invalid temperatures should raise
        with pytest.raises(ValueError, match="temperature must be between"):
            IconGeneratorConfig(temperature=-0.1)
        with pytest.raises(ValueError, match="temperature must be between"):
            IconGeneratorConfig(temperature=2.1)

    def test_system_prompt_override(self):
        """Should accept custom system prompt."""
        config = IconGeneratorConfig(system_prompt="Custom prompt for icons")
        assert config.system_prompt == "Custom prompt for icons"


class TestIconType:
    """Tests for IconType enum properties."""

    def test_app_icon_requires_opaque_background(self):
        """APP_ICON should require opaque background."""
        assert IconType.APP_ICON.requires_opaque_background is True
        assert IconType.APP_ICON.allows_transparency is False

    def test_menu_icon_allows_transparency(self):
        """MENU_ICON should allow transparency."""
        assert IconType.MENU_ICON.requires_opaque_background is False
        assert IconType.MENU_ICON.allows_transparency is True

    def test_favicon_allows_transparency(self):
        """FAVICON should allow transparency."""
        assert IconType.FAVICON.requires_opaque_background is False
        assert IconType.FAVICON.allows_transparency is True


class TestVariantEnum:
    """Tests for Variant enum."""

    def test_all_variants_defined(self):
        """All expected variants should be defined."""
        from gemimg.icons import Variant
        assert Variant.LIGHT.value == "light"
        assert Variant.DARK.value == "dark"
        assert Variant.TINTED.value == "tinted"


class TestIconGenerator:
    """Tests for IconGenerator orchestration."""

    def test_explicit_light_skips_generation(self, sample_icon, temp_output_dir):
        """Providing --light should use that image, not generate."""
        # Save sample icon to temp file
        light_path = temp_output_dir / "light.png"
        sample_icon.save(str(light_path))

        config = IconGeneratorConfig(
            platforms={Platform.IOS},
            output_dir=temp_output_dir / "output",
        )
        generator = IconGenerator(gemimg=None, config=config)

        result = generator.generate(
            light=light_path,
        )

        assert result.api_calls == 0  # No generation needed
        assert result.variants.light is not None
        assert "ios/light" in result.output_paths

    def test_input_images_are_style_references(self, sample_icon, temp_output_dir):
        """Input images should be style references, requiring a prompt."""
        # Save sample icon to temp file
        ref_path = temp_output_dir / "ref.png"
        sample_icon.save(str(ref_path))

        config = IconGeneratorConfig(
            platforms={Platform.IOS},
            output_dir=temp_output_dir / "output",
        )
        generator = IconGenerator(gemimg=None, config=config)

        # Should require prompt since input_images are references, not variants
        with pytest.raises(ValueError, match="Prompt is required"):
            generator.generate(input_images=[ref_path])

    def test_generate_requires_prompt_for_generation(self, temp_output_dir):
        """Should raise error if generation needed but no prompt."""
        config = IconGeneratorConfig(
            platforms={Platform.IOS},
            output_dir=temp_output_dir / "output",
        )
        generator = IconGenerator(gemimg=None, config=config)

        with pytest.raises(ValueError, match="Prompt is required"):
            generator.generate()

    def test_from_preset(self):
        """Should create generator from preset."""
        generator = IconGenerator.from_preset(Preset.MOBILE)

        assert Platform.IOS in generator.config.platforms
        assert Platform.ANDROID in generator.config.platforms
        assert len(generator.config.platforms) == 2

    def test_generate_raises_on_nonexistent_input_file(self, temp_output_dir):
        """Should raise FileNotFoundError for non-existent input files."""
        config = IconGeneratorConfig(
            platforms={Platform.IOS},
            output_dir=temp_output_dir / "output",
        )
        generator = IconGenerator(gemimg=None, config=config)

        with pytest.raises(FileNotFoundError, match="Image file not found"):
            generator.generate(input_images=[Path("/nonexistent/image.png")])

    def test_generate_raises_on_nonexistent_variant_file(self, temp_output_dir):
        """Should raise FileNotFoundError for non-existent variant files."""
        config = IconGeneratorConfig(
            platforms={Platform.IOS},
            output_dir=temp_output_dir / "output",
        )
        generator = IconGenerator(gemimg=None, config=config)

        with pytest.raises(FileNotFoundError, match="Image file not found"):
            generator.generate(light=Path("/nonexistent/light.png"))

    def test_generate_requires_gemimg_for_generation(self, temp_output_dir):
        """Should raise ValueError when GemImg needed but not provided."""
        config = IconGeneratorConfig(
            platforms={Platform.IOS},
            output_dir=temp_output_dir / "output",
        )
        generator = IconGenerator(gemimg=None, config=config)

        with pytest.raises(ValueError, match="GemImg instance is required"):
            generator.generate(prompt="test icon")
