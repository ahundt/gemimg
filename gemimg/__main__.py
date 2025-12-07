import argparse
import os
import sys
from pathlib import Path

from .gemimg import GemImg
from .grid import Grid
from .utils import save_image


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared between commands."""
    parser.add_argument(
        "--api-key",
        default=os.getenv("GEMINI_API_KEY"),
        help="API key for the Gemini API. Defaults to the GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash-image", help="The model to use."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("GOOGLE_GEMINI_BASE_URL"),
        help="Alternative Gemini API endpoint for your organization.",
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

    # Create config
    config = IconGeneratorConfig(
        icon_type=icon_type,
        platforms=platforms,
        themed=args.themed,
        macos_shadow=not args.no_shadow,
        validate_safe_zone=not args.no_safe_zone_check,
        output_dir=Path(args.output_dir),
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
            description="Generate app icons for multiple platforms.",
        )

        parser.add_argument(
            "prompt",
            nargs="?",
            default=None,
            help="Text prompt for icon generation.",
        )
        parser.add_argument(
            "-i",
            "--input-images",
            nargs="+",
            default=[],
            help="Input images (0=prompt only, 1=light, 2-3=variants, 4+=style refs).",
        )
        parser.add_argument(
            "--light",
            default=None,
            help="Explicit light variant file.",
        )
        parser.add_argument(
            "--dark",
            default=None,
            help="Explicit dark variant file.",
        )
        parser.add_argument(
            "--tinted",
            default=None,
            help="Explicit tinted/monochrome variant file.",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            default="icons",
            help="Output directory for generated icons (default: icons/).",
        )

        # Icon type (mutually exclusive)
        icon_type_group = parser.add_mutually_exclusive_group()
        icon_type_group.add_argument(
            "--app-icon",
            action="store_true",
            default=True,
            help="Main launcher icon (default). Opaque background enforced.",
        )
        icon_type_group.add_argument(
            "--menu-icon",
            action="store_true",
            help="In-app UI icons. Transparency preserved.",
        )
        icon_type_group.add_argument(
            "--favicon",
            action="store_true",
            help="Website favicons only (ICO + small PNGs).",
        )

        # Platform selection
        parser.add_argument(
            "--preset",
            default="all",
            choices=["mobile", "desktop", "apple", "all"],
            help="Platform preset (default: all).",
        )
        parser.add_argument(
            "--platforms",
            nargs="+",
            choices=["ios", "macos", "android", "windows", "pwa"],
            help="Specific platforms (overrides --preset).",
        )

        # Themed variants
        parser.add_argument(
            "--themed",
            action="store_true",
            help="Generate all 3 variants (light+dark+tinted) in single call.",
        )

        # Post-processing options
        parser.add_argument(
            "--no-shadow",
            action="store_true",
            help="Skip macOS shadow template.",
        )
        parser.add_argument(
            "--no-safe-zone-check",
            action="store_true",
            help="Skip Android safe zone validation.",
        )

        add_common_args(parser)

        args = parser.parse_args(sys.argv[2:])
        icons_command(args)

    else:
        # Default generate command (backward compatible)
        parser = argparse.ArgumentParser(
            description="Generate images using the Gemini API."
        )

        parser.add_argument("prompt", help="The text prompt for image generation.")
        parser.add_argument(
            "-i",
            "--input-images",
            nargs="+",
            help="Optional paths to input images.",
            default=[],
        )
        parser.add_argument(
            "-o",
            "--output-file",
            help="Optional output filename. Defaults to output.png, output-2.png, etc.",
            default=None,
        )
        parser.add_argument(
            "--aspect-ratio", default="1:1", help="Aspect ratio of the generated image."
        )
        parser.add_argument(
            "--no-resize",
            action="store_false",
            dest="resize_inputs",
            help="Do not resize input images.",
        )
        parser.add_argument(
            "--output-dir", default="", help="Directory to save the generated images."
        )
        parser.add_argument(
            "--temperature", type=float, default=1.0, help="Generation temperature."
        )
        parser.add_argument(
            "--webp", action="store_true", help="Save as WEBP instead of PNG."
        )
        parser.add_argument("-n", type=int, default=1, help="Number of images to generate.")
        parser.add_argument(
            "--store-prompt",
            action="store_true",
            help="Store the prompt in the image metadata.",
        )
        parser.add_argument(
            "--image-size",
            default="2K",
            help="Image size for the generation (Pro models only).",
        )
        parser.add_argument(
            "--system-prompt",
            default=None,
            help="System prompt for the generation (Pro models only).",
        )
        parser.add_argument(
            "--grid",
            default=None,
            help="Grid dimensions as ROWSxCOLS (e.g., 2x2). Pro models only.",
        )
        parser.add_argument(
            "--google-search",
            action="store_true",
            help="Enable Google Search grounding for real-time data (Gemini 3 Pro only).",
        )
        parser.add_argument(
            "--grid-aspect-ratio",
            default="1:1",
            help="Aspect ratio for grid cells (default: 1:1).",
        )
        parser.add_argument(
            "--grid-image-size",
            default="2K",
            help="Image size for grid generation (default: 2K).",
        )
        parser.add_argument(
            "--save-grid-original",
            action="store_true",
            help="Save the original grid image before slicing.",
        )
        parser.add_argument(
            "-f",
            "--force",
            action="store_true",
            help="Force overwrite of existing files.",
        )

        add_common_args(parser)

        args = parser.parse_args()
        generate_command(args)


if __name__ == "__main__":
    main()
