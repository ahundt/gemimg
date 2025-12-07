"""Constants and types for app icon generation."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class Platform(Enum):
    """Supported platforms for icon generation."""

    IOS = "ios"
    MACOS = "macos"
    ANDROID = "android"
    WINDOWS = "windows"
    PWA = "pwa"


class Preset(Enum):
    """Preset platform groups."""

    MOBILE = "mobile"  # iOS + Android
    DESKTOP = "desktop"  # macOS + Windows
    APPLE = "apple"  # iOS + macOS
    ALL = "all"  # All platforms


class IconType(Enum):
    """Types of icons with different requirements."""

    APP_ICON = "app-icon"  # Launcher icon, opaque background
    MENU_ICON = "menu-icon"  # In-app icons, transparency allowed
    FAVICON = "favicon"  # Website favicons


PRESET_PLATFORMS: Dict[Preset, List[Platform]] = {
    Preset.MOBILE: [Platform.IOS, Platform.ANDROID],
    Preset.DESKTOP: [Platform.MACOS, Platform.WINDOWS],
    Preset.APPLE: [Platform.IOS, Platform.MACOS],
    Preset.ALL: [Platform.IOS, Platform.MACOS, Platform.ANDROID, Platform.WINDOWS, Platform.PWA],
}


@dataclass
class IconSpec:
    """Specification for a single icon size."""

    width: int
    height: int
    scale: int = 1
    purpose: str = "any"  # For PWA: "any", "maskable"
    filename_suffix: str = ""

    @property
    def size(self) -> Tuple[int, int]:
        """Return the (width, height) tuple."""
        return (self.width, self.height)

    @property
    def scaled_size(self) -> Tuple[int, int]:
        """Return the scaled (width, height) tuple."""
        return (self.width * self.scale, self.height * self.scale)


# iOS icon sizes (App Store requires 1024x1024 master)
IOS_ICON_SPECS: List[IconSpec] = [
    IconSpec(1024, 1024, 1, filename_suffix="1024"),
    IconSpec(180, 180, 1, filename_suffix="180"),  # iPhone @3x
    IconSpec(167, 167, 1, filename_suffix="167"),  # iPad Pro @2x
    IconSpec(152, 152, 1, filename_suffix="152"),  # iPad @2x
    IconSpec(120, 120, 1, filename_suffix="120"),  # iPhone @2x
    IconSpec(87, 87, 1, filename_suffix="87"),  # Spotlight @3x
    IconSpec(80, 80, 1, filename_suffix="80"),  # Spotlight @2x
    IconSpec(76, 76, 1, filename_suffix="76"),  # iPad @1x
    IconSpec(60, 60, 1, filename_suffix="60"),  # iPhone @1x
    IconSpec(58, 58, 1, filename_suffix="58"),  # Settings @2x
    IconSpec(40, 40, 1, filename_suffix="40"),  # Spotlight @1x
    IconSpec(29, 29, 1, filename_suffix="29"),  # Settings @1x
    IconSpec(20, 20, 1, filename_suffix="20"),  # Notifications @1x
]

# macOS icon sizes (with @2x variants)
MACOS_ICON_SPECS: List[IconSpec] = [
    IconSpec(512, 512, 2, filename_suffix="512@2x"),  # 1024x1024
    IconSpec(512, 512, 1, filename_suffix="512"),
    IconSpec(256, 256, 2, filename_suffix="256@2x"),  # 512x512
    IconSpec(256, 256, 1, filename_suffix="256"),
    IconSpec(128, 128, 2, filename_suffix="128@2x"),  # 256x256
    IconSpec(128, 128, 1, filename_suffix="128"),
    IconSpec(32, 32, 2, filename_suffix="32@2x"),  # 64x64
    IconSpec(32, 32, 1, filename_suffix="32"),
    IconSpec(16, 16, 2, filename_suffix="16@2x"),  # 32x32
    IconSpec(16, 16, 1, filename_suffix="16"),
]

# Android icon sizes (adaptive icons use 512x512 master)
ANDROID_ICON_SPECS: List[IconSpec] = [
    IconSpec(512, 512, 1, filename_suffix="512"),  # Play Store
    IconSpec(192, 192, 1, filename_suffix="xxxhdpi"),
    IconSpec(144, 144, 1, filename_suffix="xxhdpi"),
    IconSpec(96, 96, 1, filename_suffix="xhdpi"),
    IconSpec(72, 72, 1, filename_suffix="hdpi"),
    IconSpec(48, 48, 1, filename_suffix="mdpi"),
]

# Windows ICO sizes (multi-size ICO file)
WINDOWS_ICO_SIZES: List[Tuple[int, int]] = [
    (256, 256),
    (128, 128),
    (64, 64),
    (48, 48),
    (32, 32),
    (16, 16),
]

# PWA icon sizes
PWA_ICON_SPECS: List[IconSpec] = [
    IconSpec(512, 512, 1, purpose="any", filename_suffix="512"),
    IconSpec(512, 512, 1, purpose="maskable", filename_suffix="512-maskable"),
    IconSpec(192, 192, 1, purpose="any", filename_suffix="192"),
    IconSpec(192, 192, 1, purpose="maskable", filename_suffix="192-maskable"),
    IconSpec(144, 144, 1, purpose="any", filename_suffix="144"),
    IconSpec(96, 96, 1, purpose="any", filename_suffix="96"),
    IconSpec(72, 72, 1, purpose="any", filename_suffix="72"),
    IconSpec(48, 48, 1, purpose="any", filename_suffix="48"),
]


# AI prompt templates
APP_ICON_PROMPT = """Generate an app icon with these MANDATORY requirements:
- MUST have a solid, opaque background color (no transparency)
- Background must completely fill the square canvas
- No semi-transparent or alpha channel elements
- sRGB color space
- Simple, bold design readable at 60x60px
- Centered composition with clear focal point
- No elements touching edges (platforms apply corner masks)

Icon subject: {user_prompt}"""

MENU_ICON_PROMPT = """Generate a UI icon with these requirements:
- Transparent background preferred
- Simple, recognizable silhouette
- Works at 24x24px to 48x48px
- Monochrome-friendly design

Icon subject: {user_prompt}"""

THEMED_ICON_PROMPT = """Generate THREE distinct app icon variants for the following concept:

{base_prompt}

OUTPUT REQUIREMENTS (3 images):

IMAGE 1 - LIGHT MODE:
- Opaque solid background (no transparency)
- Designed for light backgrounds
- sRGB color space, 1024x1024

IMAGE 2 - DARK MODE:
- Opaque solid background, dark/rich colors
- Optimized for dark backgrounds
- Same brand identity, adjusted for dark theme visibility
- sRGB color space, 1024x1024

IMAGE 3 - TINTED/MONOCHROME:
- Solid silhouette shape only
- Clean edges, simple recognizable form
- No internal details, just the outline/shape
- Works as Android themed icon (system will colorize)
- 1024x1024

All three icons must be visually consistent and recognizable as the same brand."""

# Android safe zone percentage (66% of total icon area)
ANDROID_SAFE_ZONE_PERCENT = 0.66

# PWA maskable icon padding (10% on each side)
PWA_MASKABLE_PADDING_PERCENT = 0.10
