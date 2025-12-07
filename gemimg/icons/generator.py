"""Icon generation orchestration."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from PIL import Image

from ..gemimg import GemImg
from .constants import (
    APP_ICON_PROMPT,
    MENU_ICON_PROMPT,
    PRESET_PLATFORMS,
    THEMED_ICON_PROMPT,
    IconType,
    Platform,
    Preset,
)
from .platforms import get_processor

logger = logging.getLogger(__name__)


def _load_image(path: Path) -> Image.Image:
    """Load image using context manager and return a copy to release file handle.

    This follows RAII pattern - the file is opened, copied, and closed immediately.
    The returned image is independent of the file handle.

    Args:
        path: Path to the image file

    Returns:
        A copy of the loaded image

    Raises:
        FileNotFoundError: If the image file does not exist
        PIL.UnidentifiedImageError: If the file is not a valid image
    """
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    with Image.open(path) as img:
        return img.copy()


@dataclass
class IconVariants:
    """Container for all icon variants."""

    light: Optional[Image.Image] = None
    dark: Optional[Image.Image] = None
    tinted: Optional[Image.Image] = None

    def has_any_variant(self) -> bool:
        """Check if at least one variant is present."""
        return any([self.light, self.dark, self.tinted])

    def count(self) -> int:
        """Return the number of non-None variants."""
        return sum(1 for v in [self.light, self.dark, self.tinted] if v is not None)


@dataclass
class IconGenerationResult:
    """Result of icon generation."""

    variants: IconVariants
    output_paths: Dict[str, List[Path]] = field(default_factory=dict)
    api_calls: int = 0

    def __post_init__(self):
        """Validate result fields."""
        if self.api_calls < 0:
            raise ValueError("api_calls cannot be negative")


@dataclass
class IconGeneratorConfig:
    """Configuration for icon generation."""

    icon_type: IconType = IconType.APP_ICON
    platforms: Set[Platform] = field(default_factory=lambda: set(PRESET_PLATFORMS[Preset.ALL]))
    themed: bool = False
    macos_shadow: bool = True
    validate_safe_zone: bool = True
    output_dir: Path = field(default_factory=lambda: Path("icons"))

    def __post_init__(self):
        """Validate configuration fields."""
        if not self.platforms:
            raise ValueError("At least one platform must be specified")


class IconGenerator:
    """
    End-to-end icon generation pipeline.

    Orchestrates AI generation and platform-specific post-processing.
    """

    def __init__(self, gemimg: Optional[GemImg] = None, config: Optional[IconGeneratorConfig] = None):
        """
        Initialize the IconGenerator.

        Args:
            gemimg: GemImg instance for AI generation. If None, will be created on demand.
            config: Configuration for icon generation.
        """
        self.gemimg = gemimg
        self.config = config or IconGeneratorConfig()

    def generate(
        self,
        prompt: Optional[str] = None,
        input_images: Optional[List[Path]] = None,
        light: Optional[Path] = None,
        dark: Optional[Path] = None,
        tinted: Optional[Path] = None,
        icon_type: Optional[IconType] = None,
        themed: Optional[bool] = None,
        output_dir: Optional[Path] = None,
        google_search: bool = False,
    ) -> IconGenerationResult:
        """
        End-to-end icon generation with flexible input handling.

        Input Image Handling:
        - 0 images: Generate from prompt (n=1 or n=3 if themed)
        - 1 image: Use as light variant OR as style reference
        - 2-3 images: Interpret as light/dark/tinted (skip generation)
        - 4+ images: Use all as style references for generation
          (Gemini 3 Pro supports up to 14 input images vs 6 for Flash)

        Explicit variant files (--light, --dark, --tinted) take precedence.

        Args:
            prompt: Text prompt for generation
            input_images: List of input image paths
            light: Explicit light variant file
            dark: Explicit dark variant file
            tinted: Explicit tinted variant file
            icon_type: Override config icon type
            themed: Override config themed setting
            output_dir: Override config output directory
            google_search: Enable Google Search grounding (Gemini 3 Pro only)

        Returns:
            IconGenerationResult with variants and output paths
        """
        # Apply overrides
        icon_type = icon_type or self.config.icon_type
        themed = themed if themed is not None else self.config.themed
        output_dir = output_dir or self.config.output_dir

        api_calls = 0
        input_images = input_images or []

        # 1. Resolve explicit variant files (highest priority)
        # Using _load_image() for RAII pattern - files are closed after copying
        variants = IconVariants(
            light=_load_image(light) if light else None,
            dark=_load_image(dark) if dark else None,
            tinted=_load_image(tinted) if tinted else None,
        )

        # 2. Handle input images based on count
        if len(input_images) == 1 and not variants.light:
            # Single input = light variant (unless used as reference)
            variants.light = _load_image(input_images[0])

        elif len(input_images) == 2 and not (variants.dark or variants.tinted):
            # Two inputs = light + dark
            if not variants.light:
                variants.light = _load_image(input_images[0])
            variants.dark = _load_image(input_images[1])

        elif len(input_images) == 3 and not (variants.dark or variants.tinted):
            # Three inputs = light + dark + tinted
            if not variants.light:
                variants.light = _load_image(input_images[0])
            variants.dark = _load_image(input_images[1])
            variants.tinted = _load_image(input_images[2])

        # 3. Determine what needs to be generated
        needs_light = variants.light is None
        needs_dark = themed and variants.dark is None
        needs_tinted = themed and variants.tinted is None

        # 4. Reference images for style consistency (4+ inputs, or inputs not used as variants)
        reference_images: List[Image.Image] = []
        if len(input_images) >= 4:
            reference_images = [_load_image(p) for p in input_images]
        elif len(input_images) > 0 and not needs_light:
            reference_images = [_load_image(p) for p in input_images]

        # 5. Generate missing variants (single API call)
        if needs_light or needs_dark or needs_tinted:
            if not prompt:
                raise ValueError("Prompt is required when generating icons")

            if not self.gemimg:
                raise ValueError("GemImg instance is required for generation")

            n_outputs = sum([needs_light, needs_dark, needs_tinted])
            gen_prompt = self._build_generation_prompt(
                icon_type, prompt, needs_light, needs_dark, needs_tinted
            )

            result = self.gemimg.generate(
                prompt=gen_prompt,
                imgs=reference_images if reference_images else None,
                n=n_outputs,
                aspect_ratio="1:1",
                image_size="1K",
                save=False,
                google_search=google_search,
            )

            if result and result.images:
                # Assign generated images to missing variants
                idx = 0
                if needs_light:
                    variants.light = result.images[idx]
                    idx += 1
                if needs_dark:
                    variants.dark = result.images[idx]
                    idx += 1
                if needs_tinted:
                    variants.tinted = result.images[idx]
                    idx += 1

                # Note: GemImg._generate_multiple makes n separate API calls
                api_calls = n_outputs
            else:
                raise RuntimeError("Failed to generate icon images")

        # 6. Post-process for all platforms (PIL only, 0 API calls)
        output_paths = self._postprocess_all_platforms(variants, output_dir, icon_type)

        return IconGenerationResult(
            variants=variants,
            output_paths=output_paths,
            api_calls=api_calls,
        )

    def _build_generation_prompt(
        self,
        icon_type: IconType,
        user_prompt: str,
        needs_light: bool,
        needs_dark: bool,
        needs_tinted: bool,
    ) -> str:
        """Build the appropriate prompt based on what needs to be generated."""
        # Get base prompt for icon type
        if icon_type == IconType.APP_ICON:
            base_prompt = APP_ICON_PROMPT.format(user_prompt=user_prompt)
        elif icon_type == IconType.MENU_ICON:
            base_prompt = MENU_ICON_PROMPT.format(user_prompt=user_prompt)
        else:
            base_prompt = user_prompt

        # If generating multiple variants, use themed prompt
        if sum([needs_light, needs_dark, needs_tinted]) > 1:
            return THEMED_ICON_PROMPT.format(base_prompt=base_prompt)

        return base_prompt

    def _postprocess_all_platforms(
        self,
        variants: IconVariants,
        output_dir: Path,
        icon_type: IconType,
    ) -> Dict[str, List[Path]]:
        """Post-process icons for all configured platforms."""
        output_paths: Dict[str, List[Path]] = {}

        # Process each variant
        variant_images = [
            ("light", variants.light),
            ("dark", variants.dark),
            ("tinted", variants.tinted),
        ]

        for variant_name, variant_image in variant_images:
            if variant_image is None:
                continue

            for platform in self.config.platforms:
                # Build platform-specific kwargs
                kwargs = {}
                if platform == Platform.MACOS:
                    kwargs["apply_shadow"] = self.config.macos_shadow
                elif platform == Platform.ANDROID:
                    kwargs["validate_safe_zone"] = self.config.validate_safe_zone

                processor = get_processor(
                    platform=platform,
                    output_dir=output_dir,
                    icon_type=icon_type,
                    variant=variant_name,
                    **kwargs,
                )

                processed = processor.process(variant_image)
                saved = processor.save(processed)

                key = f"{platform.value}/{variant_name}"
                output_paths[key] = saved

        return output_paths

    @classmethod
    def from_preset(
        cls,
        preset: Preset,
        gemimg: Optional[GemImg] = None,
        **kwargs,
    ) -> "IconGenerator":
        """
        Create an IconGenerator with a preset platform configuration.

        Args:
            preset: Platform preset (MOBILE, DESKTOP, APPLE, ALL)
            gemimg: GemImg instance for AI generation
            **kwargs: Additional config options

        Returns:
            Configured IconGenerator instance
        """
        platforms = set(PRESET_PLATFORMS[preset])
        config = IconGeneratorConfig(platforms=platforms, **kwargs)
        return cls(gemimg=gemimg, config=config)
