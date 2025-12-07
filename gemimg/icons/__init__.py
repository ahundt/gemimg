"""App icon generation module for gemimg."""

from .constants import Platform, Preset, IconSpec, IconType, Variant, PRESET_PLATFORMS
from .generator import IconGenerator, IconVariants, IconGenerationResult, IconGeneratorConfig
from .platforms import (
    PlatformProcessor,
    IOSProcessor,
    MacOSProcessor,
    AndroidProcessor,
    WindowsProcessor,
    PWAProcessor,
)

__all__ = [
    "Platform",
    "Preset",
    "IconSpec",
    "IconType",
    "Variant",
    "PRESET_PLATFORMS",
    "IconGenerator",
    "IconGeneratorConfig",
    "IconVariants",
    "IconGenerationResult",
    "PlatformProcessor",
    "IOSProcessor",
    "MacOSProcessor",
    "AndroidProcessor",
    "WindowsProcessor",
    "PWAProcessor",
]
