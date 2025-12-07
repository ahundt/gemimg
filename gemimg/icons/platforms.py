"""Platform-specific icon processors for post-processing."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from .constants import (
    ANDROID_ICON_SPECS,
    ANDROID_SAFE_ZONE_PERCENT,
    IOS_ICON_SPECS,
    MACOS_ICON_SPECS,
    PWA_ICON_SPECS,
    PWA_MASKABLE_PADDING_PERCENT,
    WINDOWS_ICO_SIZES,
    IconSpec,
    IconType,
    Platform,
)

logger = logging.getLogger(__name__)


class PlatformProcessor(ABC):
    """Abstract base class for platform-specific icon processing."""

    platform: Platform

    def __init__(
        self,
        output_dir: Path,
        icon_type: IconType = IconType.APP_ICON,
        variant: str = "light",
    ):
        """
        Initialize the processor.

        Args:
            output_dir: Base output directory for icons
            icon_type: Type of icon being processed
            variant: Icon variant (light, dark, tinted)
        """
        self.output_dir = output_dir
        self.icon_type = icon_type
        self.variant = variant

    @abstractmethod
    def get_specs(self) -> List[IconSpec]:
        """Return the icon specifications for this platform."""
        pass

    @abstractmethod
    def process(self, source_image: Image.Image) -> Dict[str, Image.Image]:
        """
        Process the source image for this platform.

        Args:
            source_image: The source icon image

        Returns:
            Dictionary mapping filename to processed image
        """
        pass

    def save(self, processed_images: Dict[str, Image.Image]) -> List[Path]:
        """
        Save processed images to the output directory.

        Args:
            processed_images: Dictionary mapping filename to image

        Returns:
            List of saved file paths
        """
        platform_dir = self.output_dir / self.platform.value / self.variant
        platform_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for filename, img in processed_images.items():
            path = platform_dir / filename
            img.save(str(path))
            saved_paths.append(path)
            logger.debug(f"Saved: {path}")

        return saved_paths


class IOSProcessor(PlatformProcessor):
    """iOS/iPadOS icon processor."""

    platform = Platform.IOS

    def get_specs(self) -> List[IconSpec]:
        return IOS_ICON_SPECS

    def process(self, source_image: Image.Image) -> Dict[str, Image.Image]:
        """
        Process source image for iOS.

        For app icons: Convert RGBA to RGB (no transparency allowed).
        For menu icons: Preserve transparency.
        """
        results = {}

        for spec in self.get_specs():
            # Resize to target size
            resized = source_image.resize(spec.size, Image.Resampling.LANCZOS)

            # For app icons, ensure no transparency
            if self.icon_type == IconType.APP_ICON and resized.mode == "RGBA":
                # Create RGB image with white background
                rgb = Image.new("RGB", resized.size, (255, 255, 255))
                rgb.paste(resized, mask=resized.split()[3])
                resized = rgb

            filename = f"AppIcon-{spec.filename_suffix}.png"
            results[filename] = resized

        return results


class MacOSProcessor(PlatformProcessor):
    """macOS icon processor with optional shadow template."""

    platform = Platform.MACOS

    def __init__(
        self,
        output_dir: Path,
        icon_type: IconType = IconType.APP_ICON,
        variant: str = "light",
        apply_shadow: bool = True,
        shadow_template: Optional[Path] = None,
    ):
        super().__init__(output_dir, icon_type, variant)
        self.apply_shadow = apply_shadow
        self.shadow_template = shadow_template

    def get_specs(self) -> List[IconSpec]:
        return MACOS_ICON_SPECS

    def process(self, source_image: Image.Image) -> Dict[str, Image.Image]:
        """
        Process source image for macOS.

        Optionally applies drop shadow template.
        """
        results = {}

        for spec in self.get_specs():
            # Calculate actual output size (accounting for @2x)
            output_size = spec.scaled_size
            resized = source_image.resize(output_size, Image.Resampling.LANCZOS)

            # Apply shadow template if available and enabled
            if self.apply_shadow and self.shadow_template and self.shadow_template.exists():
                resized = self._apply_shadow(resized)

            filename = f"AppIcon-{spec.filename_suffix}.png"
            results[filename] = resized

        return results

    def _apply_shadow(self, icon: Image.Image) -> Image.Image:
        """Apply macOS-style drop shadow to icon."""
        try:
            shadow = Image.open(self.shadow_template).convert("RGBA")
            # Resize shadow to match icon size
            shadow = shadow.resize(icon.size, Image.Resampling.LANCZOS)

            # Ensure icon has alpha channel
            if icon.mode != "RGBA":
                icon = icon.convert("RGBA")

            # Composite icon over shadow
            shadow.paste(icon, (0, 0), icon)
            return shadow
        except Exception as e:
            logger.warning(f"Failed to apply shadow template: {e}")
            return icon


class AndroidProcessor(PlatformProcessor):
    """Android icon processor with adaptive icon support."""

    platform = Platform.ANDROID

    def __init__(
        self,
        output_dir: Path,
        icon_type: IconType = IconType.APP_ICON,
        variant: str = "light",
        validate_safe_zone: bool = True,
    ):
        super().__init__(output_dir, icon_type, variant)
        self.validate_safe_zone = validate_safe_zone

    def get_specs(self) -> List[IconSpec]:
        return ANDROID_ICON_SPECS

    def process(self, source_image: Image.Image) -> Dict[str, Image.Image]:
        """
        Process source image for Android.

        Validates safe zone for adaptive icons.
        """
        results = {}

        # Validate safe zone if enabled
        if self.validate_safe_zone:
            self._check_safe_zone(source_image)

        for spec in self.get_specs():
            resized = source_image.resize(spec.size, Image.Resampling.LANCZOS)
            filename = f"ic_launcher-{spec.filename_suffix}.png"
            results[filename] = resized

        # Generate monochrome variant for themed icons (from tinted variant)
        if self.variant == "tinted":
            mono = self._create_monochrome(source_image)
            results["ic_launcher_monochrome.png"] = mono

        return results

    def _check_safe_zone(self, img: Image.Image) -> None:
        """
        Check if content is within the 66% safe zone.

        Logs a warning if content appears outside the safe zone.
        """
        if img.mode != "RGBA":
            return

        width, height = img.size
        safe_margin = int(width * (1 - ANDROID_SAFE_ZONE_PERCENT) / 2)

        # Check if there are non-transparent pixels near the edges
        pixels = img.load()
        edge_content = False

        for x in range(width):
            for y in range(height):
                if x < safe_margin or x >= width - safe_margin:
                    if y < safe_margin or y >= height - safe_margin:
                        if pixels[x, y][3] > 128:  # Non-transparent
                            edge_content = True
                            break
            if edge_content:
                break

        if edge_content:
            logger.warning(
                "Icon content may extend beyond Android adaptive icon safe zone (66%). "
                "Content near edges may be clipped on some devices."
            )

    def _create_monochrome(self, img: Image.Image) -> Image.Image:
        """Create monochrome version from tinted icon."""
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Extract alpha channel as the monochrome mask
        _, _, _, alpha = img.split()

        # Create a white silhouette on transparent background
        mono = Image.new("RGBA", img.size, (0, 0, 0, 0))
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        mono.paste(white, mask=alpha)

        return mono


class WindowsProcessor(PlatformProcessor):
    """Windows ICO file processor."""

    platform = Platform.WINDOWS

    def get_specs(self) -> List[IconSpec]:
        return [IconSpec(size[0], size[1]) for size in WINDOWS_ICO_SIZES]

    def process(self, source_image: Image.Image) -> Dict[str, Image.Image]:
        """
        Process source image for Windows.

        Creates a multi-size ICO file.
        """
        # Prepare sizes for ICO
        ico_images = []
        for size in WINDOWS_ICO_SIZES:
            resized = source_image.resize(size, Image.Resampling.LANCZOS)
            ico_images.append(resized)

        # Return the largest size as the main image
        # The actual ICO saving happens in save()
        return {"icon.ico": source_image}

    def save(self, processed_images: Dict[str, Image.Image]) -> List[Path]:
        """Save as multi-size ICO file."""
        platform_dir = self.output_dir / self.platform.value / self.variant
        platform_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        source = list(processed_images.values())[0]

        # Create all size variants
        ico_images = []
        for size in WINDOWS_ICO_SIZES:
            resized = source.resize(size, Image.Resampling.LANCZOS)
            ico_images.append(resized)

        # Save as ICO with all sizes
        ico_path = platform_dir / "icon.ico"
        ico_images[0].save(
            str(ico_path),
            format="ICO",
            sizes=WINDOWS_ICO_SIZES,
            append_images=ico_images[1:],
        )
        saved_paths.append(ico_path)
        logger.debug(f"Saved: {ico_path}")

        return saved_paths


class PWAProcessor(PlatformProcessor):
    """Progressive Web App icon processor."""

    platform = Platform.PWA

    def get_specs(self) -> List[IconSpec]:
        return PWA_ICON_SPECS

    def process(self, source_image: Image.Image) -> Dict[str, Image.Image]:
        """
        Process source image for PWA.

        Creates standard and maskable icon variants.
        """
        results = {}

        for spec in self.get_specs():
            resized = source_image.resize(spec.size, Image.Resampling.LANCZOS)

            if spec.purpose == "maskable":
                # Add padding for maskable icons
                resized = self._add_maskable_padding(resized)

            filename = f"icon-{spec.filename_suffix}.png"
            results[filename] = resized

        return results

    def _add_maskable_padding(self, img: Image.Image) -> Image.Image:
        """Add 10% padding for maskable PWA icons."""
        width, height = img.size
        padding = int(width * PWA_MASKABLE_PADDING_PERCENT)

        # Calculate the inner size
        inner_size = width - (2 * padding)

        # Resize the icon to fit in the inner area
        resized = img.resize((inner_size, inner_size), Image.Resampling.LANCZOS)

        # Create a new image with white background
        padded = Image.new("RGB", (width, height), (255, 255, 255))

        # Center the resized icon
        padded.paste(resized, (padding, padding))

        return padded

    def save(self, processed_images: Dict[str, Image.Image]) -> List[Path]:
        """Save icons and generate manifest-icons.json."""
        saved_paths = super().save(processed_images)

        # Generate manifest-icons.json
        manifest = self._generate_manifest()
        manifest_path = self.output_dir / self.platform.value / self.variant / "manifest-icons.json"

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        saved_paths.append(manifest_path)
        logger.debug(f"Saved: {manifest_path}")

        return saved_paths

    def _generate_manifest(self) -> List[Dict]:
        """Generate the icons array for web manifest."""
        icons = []
        for spec in self.get_specs():
            icons.append({
                "src": f"icon-{spec.filename_suffix}.png",
                "sizes": f"{spec.width}x{spec.height}",
                "type": "image/png",
                "purpose": spec.purpose,
            })
        return icons


def get_processor(
    platform: Platform,
    output_dir: Path,
    icon_type: IconType = IconType.APP_ICON,
    variant: str = "light",
    **kwargs,
) -> PlatformProcessor:
    """
    Factory function to get the appropriate processor for a platform.

    Args:
        platform: Target platform
        output_dir: Base output directory
        icon_type: Type of icon
        variant: Icon variant (light, dark, tinted)
        **kwargs: Additional processor-specific options

    Returns:
        Appropriate PlatformProcessor instance
    """
    processors = {
        Platform.IOS: IOSProcessor,
        Platform.MACOS: MacOSProcessor,
        Platform.ANDROID: AndroidProcessor,
        Platform.WINDOWS: WindowsProcessor,
        Platform.PWA: PWAProcessor,
    }

    processor_class = processors.get(platform)
    if not processor_class:
        raise ValueError(f"Unsupported platform: {platform}")

    return processor_class(output_dir, icon_type, variant, **kwargs)
