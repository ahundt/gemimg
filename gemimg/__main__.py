import argparse
import os
import sys
from pathlib import Path

from .gemimg import GemImg
from .grid import Grid
from .utils import save_image


def existing_file(path: str) -> str:
    """Argparse type validator that ensures file exists."""
    if not Path(path).exists():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    if not Path(path).is_file():
        raise argparse.ArgumentTypeError(f"Not a file: {path}")
    return path


def temperature_range(value: str) -> float:
    """Argparse type validator for temperature (0.0-2.0)."""
    try:
        temp = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid temperature: {value}")
    if not 0.0 <= temp <= 2.0:
        raise argparse.ArgumentTypeError(
            f"Temperature must be between 0.0 and 2.0, got {temp}"
        )
    return temp

# Model names for help text
MODELS = {
    "flash": "gemini-2.5-flash-image",
    "pro": "gemini-2.5-pro-image",
    "gemini3-flash": "gemini-3.0-flash-image",
    "gemini3-pro": "gemini-3.0-pro-image",
}
DEFAULT_MODEL = MODELS["flash"]


def add_api_args(parser: argparse.ArgumentParser) -> None:
    """Add API connection arguments."""
    group = parser.add_argument_group(
        "API Connection",
        "Authentication and endpoint configuration."
    )
    group.add_argument(
        "--api-key",
        default=os.getenv("GEMINI_API_KEY"),
        metavar="KEY",
        help="Gemini API key. Defaults to GEMINI_API_KEY env var.",
    )
    group.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="NAME",
        help=f"Model to use (default: {DEFAULT_MODEL}). "
             "Pro models support --image-size and --system-prompt. "
             "Gemini 3 models support --google-search.",
    )
    group.add_argument(
        "--base-url",
        default=os.getenv("GOOGLE_GEMINI_BASE_URL"),
        metavar="URL",
        help="Alternative API endpoint (defaults to GOOGLE_GEMINI_BASE_URL env var).",
    )


def add_generation_args(parser: argparse.ArgumentParser, for_icons: bool = False) -> None:
    """Add generation parameters shared between commands.

    Args:
        parser: The argument parser to add arguments to.
        for_icons: If True, used for icons subcommand context (currently unused).
    """
    group = parser.add_argument_group(
        "Generation Options",
        "Control image generation behavior. Some options require Pro or Gemini 3 models."
    )
    group.add_argument(
        "--temperature",
        type=temperature_range,
        default=1.0,
        metavar="FLOAT",
        help="Creativity level 0.0-2.0 (default: 1.0). "
             "Lower = more deterministic, higher = more varied.",
    )
    # image-size: both icons and general default to 2K for future-proofing
    default_size = "2K"
    group.add_argument(
        "--image-size",
        default=default_size,
        choices=["1K", "2K", "4K"],
        help=f"Output resolution (default: {default_size}). [Pro models only]",
    )
    group.add_argument(
        "--system-prompt",
        default=None,
        metavar="TEXT",
        help="Custom system instruction to guide generation style. [Pro models only]",
    )
    group.add_argument(
        "--google-search",
        action="store_true",
        help="Enable Google Search grounding for real-time information. [Gemini 3 only]",
    )


def generate_command(args: argparse.Namespace) -> None:
    """Handle the generate command (default behavior)."""
    if not args.api_key:
        print(
            "Error: API key is required. Provide it with --api-key or set the GEMINI_API_KEY environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    base_url = args.base_url or "https://generativelanguage.googleapis.com"
    gem_img = GemImg(api_key=args.api_key, model=args.model, base_url=base_url)

    # Parse grid dimensions if provided
    grid = None
    if args.grid:
        try:
            rows, cols = map(int, args.grid.lower().split("x"))
            grid = Grid(
                rows=rows,
                cols=cols,
                aspect_ratio=args.grid_aspect_ratio,
                image_size=args.grid_image_size,
                save_original_image=args.save_grid_original,
            )
        except ValueError:
            print(
                f"Error: Invalid grid format '{args.grid}'. Use ROWSxCOLS (e.g., 2x2).",
                file=sys.stderr,
            )
            sys.exit(1)

    # We call generate with save=False to handle file saving manually.
    result = gem_img.generate(
        prompt=args.prompt,
        imgs=args.input_images,
        aspect_ratio=args.aspect_ratio,
        resize_inputs=args.resize_inputs,
        save=False,
        temperature=args.temperature,
        webp=args.webp,
        n=args.n,
        store_prompt=args.store_prompt,
        image_size=args.image_size,
        system_prompt=args.system_prompt,
        grid=grid,
        google_search=args.google_search,
    )

    if result and result.images:
        ext = "webp" if args.webp else "png"

        output_path = Path(args.output_dir)
        if args.output_file:
            base_name = Path(args.output_file).stem
            if Path(args.output_file).suffix:
                ext = Path(args.output_file).suffix[1:]
        else:
            base_name = "output"

        output_path.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(result.images):
            if len(result.images) > 1:
                current_base = f"{base_name}-{i + 1}"
            else:
                current_base = base_name

            final_path = output_path / f"{current_base}.{ext}"

            if not args.force:
                counter = 1
                while final_path.exists():
                    final_path = output_path / f"{current_base}-{counter}.{ext}"
                    counter += 1

            save_image(img, str(final_path), args.store_prompt, args.prompt)
            print(f"Image saved to {final_path}")
    else:
        print("Failed to generate image.")


def icons_command(args: argparse.Namespace) -> None:
    """Handle the icons subcommand."""
    from .icons import IconGenerator, IconGeneratorConfig, IconType, Platform, Preset, PRESET_PLATFORMS

    if not args.api_key:
        # API key is only required if we need to generate
        has_all_variants = args.light and (not args.themed or (args.dark and args.tinted))
        has_enough_inputs = len(args.input_images) >= (3 if args.themed else 1)

        if not (has_all_variants or has_enough_inputs):
            print(
                "Error: API key is required for generation. Provide it with --api-key or set the GEMINI_API_KEY environment variable.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Determine platforms
    if args.platforms:
        platforms = set()
        for p in args.platforms:
            try:
                platforms.add(Platform(p.lower()))
            except ValueError:
                print(f"Error: Unknown platform '{p}'", file=sys.stderr)
                sys.exit(1)
    else:
        try:
            preset = Preset(args.preset.lower())
            platforms = set(PRESET_PLATFORMS[preset])
        except ValueError:
            print(f"Error: Unknown preset '{args.preset}'", file=sys.stderr)
            sys.exit(1)

    # Determine icon type
    if args.menu_icon:
        icon_type = IconType.MENU_ICON
    elif args.favicon:
        icon_type = IconType.FAVICON
    else:
        icon_type = IconType.APP_ICON

    # Create config with all overridable parameters
    config = IconGeneratorConfig(
        icon_type=icon_type,
        platforms=platforms,
        themed=args.themed,
        macos_shadow=not args.no_shadow,
        validate_safe_zone=not args.no_safe_zone_check,
        output_dir=Path(args.output_dir),
        image_size=args.image_size,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
    )

    # Create GemImg instance if needed
    gemimg = None
    if args.api_key:
        base_url = args.base_url or "https://generativelanguage.googleapis.com"
        gemimg = GemImg(api_key=args.api_key, model=args.model, base_url=base_url)

    # Create generator and run
    generator = IconGenerator(gemimg=gemimg, config=config)

    try:
        result = generator.generate(
            prompt=args.prompt,
            input_images=[Path(p) for p in args.input_images] if args.input_images else None,
            light=Path(args.light) if args.light else None,
            dark=Path(args.dark) if args.dark else None,
            tinted=Path(args.tinted) if args.tinted else None,
            google_search=args.google_search,
        )

        print(f"Generated icons with {result.api_calls} API call(s)")
        for key, paths in result.output_paths.items():
            print(f"  {key}:")
            for path in paths:
                print(f"    - {path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """CLI for generating images with GemImg."""
    # Check if first argument is 'icons' subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "icons":
        # Icons subcommand
        parser = argparse.ArgumentParser(
            prog="gemimg icons",
            description="""Generate production-ready app icons for iOS, macOS, Android, Windows, and PWA.

Examples:
  gemimg icons "a friendly robot mascot"              # Generate from prompt
  gemimg icons -i logo.png                            # Process existing image
  gemimg icons "gaming app" --platforms ios android   # Specific platforms
  gemimg icons -i icon.png --themed                   # Generate all 3 variants
  gemimg icons "tech startup" --image-size 2K         # Higher resolution

The icons subcommand handles all platform-specific requirements:
  - iOS: App Store sizes (1024px down to 20px)
  - macOS: ICNS format with optional drop shadow
  - Android: Adaptive icons with safe zone validation
  - Windows: Multi-size ICO file
  - PWA: Manifest with regular and maskable icons

Model Capabilities:
  Flash models  - Basic generation, up to 6 input images
  Pro models    - --image-size, --system-prompt, up to 6 input images
  Gemini 3      - --google-search, up to 14 input images""",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Input/Output group
        io_group = parser.add_argument_group(
            "Input/Output",
            "Source images and output destination."
        )
        io_group.add_argument(
            "prompt",
            nargs="?",
            default=None,
            help="Text description of the icon to generate (e.g., 'a minimalist coffee cup').",
        )
        io_group.add_argument(
            "-i",
            "--input-images",
            nargs="+",
            type=existing_file,
            default=[],
            metavar="FILE",
            help="Style reference images for AI generation. "
                 "Use --light/--dark/--tinted to provide pre-made variants.",
        )
        io_group.add_argument(
            "-o",
            "--output-dir",
            default="icons",
            metavar="DIR",
            help="Output directory (default: icons/).",
        )

        # Explicit variant files group
        variant_group = parser.add_argument_group(
            "Explicit Variants",
            "Provide pre-made icon variants (skips AI generation for that variant)."
        )
        variant_group.add_argument(
            "--light",
            type=existing_file,
            default=None,
            metavar="FILE",
            help="Light mode icon variant.",
        )
        variant_group.add_argument(
            "--dark",
            type=existing_file,
            default=None,
            metavar="FILE",
            help="Dark mode icon variant.",
        )
        variant_group.add_argument(
            "--tinted",
            type=existing_file,
            default=None,
            metavar="FILE",
            help="Tinted/monochrome variant (for Android themed icons).",
        )

        # Icon type group
        type_group = parser.add_argument_group(
            "Icon Type",
            "What kind of icon to generate (mutually exclusive)."
        )
        icon_type_mutex = type_group.add_mutually_exclusive_group()
        icon_type_mutex.add_argument(
            "--app-icon",
            action="store_true",
            default=True,
            help="Launcher icon with opaque background (default).",
        )
        icon_type_mutex.add_argument(
            "--menu-icon",
            action="store_true",
            help="In-app UI icon with transparency preserved.",
        )
        icon_type_mutex.add_argument(
            "--favicon",
            action="store_true",
            help="Website favicon (ICO + small PNGs).",
        )

        # Platform selection group
        platform_group = parser.add_argument_group(
            "Platform Selection",
            "Choose target platforms. --platforms overrides --preset."
        )
        platform_group.add_argument(
            "--preset",
            default="all",
            choices=["mobile", "desktop", "apple", "all"],
            help="Platform preset: mobile=iOS+Android, desktop=macOS+Windows, "
                 "apple=iOS+macOS, all=everything (default).",
        )
        platform_group.add_argument(
            "--platforms",
            nargs="+",
            choices=["ios", "macos", "android", "windows", "pwa"],
            metavar="PLATFORM",
            help="Specific platforms to generate for.",
        )

        # Icon processing group
        processing_group = parser.add_argument_group(
            "Icon Processing",
            "Post-processing options for platform-specific requirements."
        )
        processing_group.add_argument(
            "--themed",
            action="store_true",
            help="Generate all 3 variants (light+dark+tinted) for themed icon support.",
        )
        processing_group.add_argument(
            "--no-shadow",
            action="store_true",
            help="Skip macOS drop shadow effect.",
        )
        processing_group.add_argument(
            "--no-safe-zone-check",
            action="store_true",
            help="Skip Android adaptive icon safe zone validation.",
        )

        # Shared generation args and API args
        add_generation_args(parser, for_icons=True)
        add_api_args(parser)

        args = parser.parse_args(sys.argv[2:])
        icons_command(args)

    else:
        # Default generate command (backward compatible)
        parser = argparse.ArgumentParser(
            prog="gemimg",
            description="""Generate images using the Gemini API.

Examples:
  gemimg "a sunset over mountains"                    # Basic generation
  gemimg "oil painting style" -i photo.jpg            # Transform image
  gemimg "product photo" --aspect-ratio 16:9          # Custom aspect ratio
  gemimg "logo design" -n 4 --temperature 1.5         # Multiple variations
  gemimg "diagram" --grid 2x2                         # Generate 4-panel grid

Model Capabilities:
  Flash models  - Basic generation, up to 6 input images
  Pro models    - --image-size, --system-prompt, --grid, up to 6 input images
  Gemini 3      - --google-search, up to 14 input images""",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Input/Output group
        io_group = parser.add_argument_group(
            "Input/Output",
            "Source content and output destination."
        )
        io_group.add_argument(
            "prompt",
            help="Text description for image generation.",
        )
        io_group.add_argument(
            "-i",
            "--input-images",
            nargs="+",
            type=existing_file,
            default=[],
            metavar="FILE",
            help="Style reference images for AI generation context.",
        )
        io_group.add_argument(
            "-o",
            "--output-file",
            default=None,
            metavar="FILE",
            help="Output filename (default: output.png, output-2.png, ...).",
        )
        io_group.add_argument(
            "--output-dir",
            default="",
            metavar="DIR",
            help="Directory for output files.",
        )
        io_group.add_argument(
            "-f",
            "--force",
            action="store_true",
            help="Overwrite existing files without prompting.",
        )

        # Image format group
        format_group = parser.add_argument_group(
            "Image Format",
            "Control output image properties."
        )
        format_group.add_argument(
            "--aspect-ratio",
            default="1:1",
            metavar="RATIO",
            help="Output aspect ratio (default: 1:1). Examples: 16:9, 4:3, 3:4.",
        )
        format_group.add_argument(
            "--webp",
            action="store_true",
            help="Save as WebP instead of PNG.",
        )
        format_group.add_argument(
            "--no-resize",
            action="store_false",
            dest="resize_inputs",
            help="Don't resize input images (may cause API errors for large images).",
        )
        format_group.add_argument(
            "-n",
            type=int,
            default=1,
            metavar="COUNT",
            help="Number of images to generate (default: 1).",
        )
        format_group.add_argument(
            "--store-prompt",
            action="store_true",
            help="Embed the prompt in output image metadata.",
        )

        # Grid generation group (Pro only)
        grid_group = parser.add_argument_group(
            "Grid Generation [Pro models only]",
            "Generate multi-panel image grids."
        )
        grid_group.add_argument(
            "--grid",
            default=None,
            metavar="RxC",
            help="Grid dimensions as ROWSxCOLS (e.g., 2x2, 3x3).",
        )
        grid_group.add_argument(
            "--grid-aspect-ratio",
            default="1:1",
            metavar="RATIO",
            help="Aspect ratio for each grid cell (default: 1:1).",
        )
        grid_group.add_argument(
            "--grid-image-size",
            default="2K",
            choices=["1K", "2K", "4K"],
            help="Resolution for grid generation (default: 2K).",
        )
        grid_group.add_argument(
            "--save-grid-original",
            action="store_true",
            help="Save the full grid image before slicing into cells.",
        )

        # Shared generation args and API args
        add_generation_args(parser, for_icons=False)
        add_api_args(parser)

        args = parser.parse_args()
        generate_command(args)


if __name__ == "__main__":
    main()
