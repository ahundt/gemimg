import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import httpx
from dotenv import load_dotenv
from PIL import Image

from .grid import Grid
from .utils import (
    _validate_aspect,
    b64_to_img,
    img_b64_part,
    img_to_b64,
    save_images_batch,
)

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class GemImg:
    api_key: str = field(default=os.getenv("GEMINI_API_KEY"), repr=False)
    client: httpx.Client = field(default_factory=httpx.Client, repr=False)
    model: str = "gemini-2.5-flash-image"
    base_url: str = field(
        default="https://generativelanguage.googleapis.com", repr=False
    )

    def __post_init__(self):
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. Pass it as `api_key`, set it as an environment variable or in .env file."
            )

    @property
    def is_pro(self) -> bool:
        """Check if the model is a pro variant."""
        return "-pro" in self.model

    @property
    def is_gemini3(self) -> bool:
        """Check if the model is a Gemini 3 variant."""
        return "gemini-3" in self.model

    def generate(
        self,
        prompt: Optional[str] = None,
        imgs: Optional[Union[str, Image.Image, List[str], List[Image.Image]]] = None,
        aspect_ratio: str = "1:1",
        resize_inputs: bool = True,
        save: bool = True,
        save_dir: str = "",
        temperature: float = 1.0,
        webp: bool = False,
        n: int = 1,
        store_prompt: bool = False,
        image_size: str = "2K",
        system_prompt: Optional[str] = None,
        grid: Optional[Grid] = None,
        google_search: bool = False,
    ) -> Optional["ImageGen"]:
        if not prompt and not imgs:
            raise ValueError("Either 'prompt' or 'imgs' must be provided")

        # Validate Google Search is only used with Gemini 3
        if google_search and not self.is_gemini3:
            raise ValueError("Google Search grounding requires a Gemini 3 model")

        # If grid is provided, use its aspect_ratio and image_size
        if grid is not None:
            if not self.is_pro:
                raise ValueError("Grid generation requires a Pro model")
            aspect_ratio = grid.aspect_ratio
            image_size = grid.image_size

        if n > 1:
            if temperature == 0:
                raise ValueError(
                    "Generating multiple images at temperature = 0.0 is redundant."
                )
            # Exclude 'self' from locals to avoid conflicts when passing as kwargs
            kwargs = {k: v for k, v in locals().items() if k != "self"}
            return self._generate_multiple(**kwargs)

        parts = []

        if imgs:
            # Ensure imgs is a list
            if isinstance(imgs, (str, Image.Image)):
                imgs = [imgs]

            # Validate input image count
            max_images = 14 if self.is_gemini3 else 6
            if len(imgs) > max_images:
                raise ValueError(
                    f"Maximum {max_images} input images for {self.model}"
                )

            img_b64_strings = [img_to_b64(img, resize_inputs) for img in imgs]
            parts.extend([img_b64_part(b64_str) for b64_str in img_b64_strings])

        if prompt:
            parts.append({"text": prompt.strip()})

        query_params = {
            "generationConfig": {
                "temperature": temperature,
                "imageConfig": {
                    "aspectRatio": _validate_aspect(aspect_ratio, self.is_pro)
                },
                "responseModalities": ["Image"],
            },
            "contents": [{"parts": parts}],
        }

        if self.is_pro:
            if image_size not in ["1K", "2K", "4K"]:
                raise ValueError("image_size must be one of '1K', '2K', or '4K'")
            query_params["generationConfig"]["imageConfig"]["imageSize"] = image_size
            if system_prompt:
                query_params["system_instruction"] = {
                    "parts": [{"text": system_prompt.strip()}]
                }

        # Add Google Search grounding for Gemini 3
        if google_search:
            query_params["tools"] = [{"googleSearch": {}}]

        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        api_url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"

        try:
            response = self.client.post(
                api_url, json=query_params, headers=headers, timeout=180
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            logger.error(
                "Request timed out after 180 seconds. "
                "The API may be experiencing high load. Please try again."
            )
            return None
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code == 401:
                logger.error("Authentication failed. Check your GEMINI_API_KEY.")
            elif status_code == 403:
                logger.error("Access forbidden. Check API quota or region restrictions.")
            elif status_code == 429:
                logger.error("Rate limit exceeded. Please wait before retrying.")
            elif status_code >= 500:
                logger.error(f"Server error ({status_code}). Please try again later.")
            else:
                logger.error(f"HTTP error {status_code}: {e.response.text[:200]}")
            return None

        try:
            response_data = response.json()
        except ValueError as e:
            logger.error(f"API returned invalid JSON response: {e}")
            return None

        if err := response_data.get("error"):
            logger.error(f"API Response Error: {err.get('code', 'unknown')} — {err.get('message', 'unknown error')}")
            return None

        # Defensive parsing with clear error messages
        if "usageMetadata" not in response_data:
            logger.error("API response missing 'usageMetadata' field.")
            return None
        usage_metadata = response_data["usageMetadata"]

        if "candidates" not in response_data or not response_data["candidates"]:
            logger.error("API response missing 'candidates' field.")
            return None
        candidates = response_data["candidates"][0]

        finish_reason = candidates.get("finishReason")
        if finish_reason in ["PROHIBITED_CONTENT", "NO_IMAGE"]:
            logger.error(f"Image was not generated due to {finish_reason}.")
            return None

        if "content" not in candidates:
            logger.error("No image is present in the response.")
            return None

        response_parts = candidates["content"].get("parts", [])

        output_images = [
            b64_to_img(part["inlineData"]["data"])
            for part in response_parts
            if "inlineData" in part
        ]

        # If grid is provided, slice the generated image(s) into subimages
        output_subimages = []
        if grid is not None:
            output_subimages = [
                sliced for img in output_images for sliced in grid.slice_image(img)
            ]

        output_image_paths = []
        output_subimage_paths = []
        if save:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            response_id = response_data["responseId"]
            file_extension = "webp" if webp else "png"
            save_kwargs = {
                "response_id": response_id,
                "save_dir": save_dir,
                "file_extension": file_extension,
                "store_prompt": store_prompt,
                "prompt": prompt,
            }

            if grid is not None:
                if grid.save_original_image:
                    output_image_paths = save_images_batch(output_images, **save_kwargs)
                output_subimage_paths = save_images_batch(
                    output_subimages, **save_kwargs
                )
            else:
                output_image_paths = save_images_batch(output_images, **save_kwargs)

        return ImageGen(
            images=output_images,
            image_paths=output_image_paths,
            usages=[
                Usage(
                    prompt_tokens=usage_metadata.get("promptTokenCount", -1),
                    completion_tokens=usage_metadata.get("candidatesTokenCount", -1),
                )
            ],
            subimages=output_subimages,
            subimage_paths=output_subimage_paths,
        )

    def _generate_multiple(self, n: int, **kwargs) -> Optional["ImageGen"]:
        """Helper to generate multiple images by accumulating results.

        Handles partial failures gracefully - if some API calls fail,
        returns successful results. Only returns None if all calls fail.
        """
        # Remove 'n' from kwargs if present to avoid duplication
        kwargs.pop("n", None)

        result = None
        failed_count = 0

        for i in range(n):
            gen_result = self.generate(n=1, **kwargs)
            if gen_result is None:
                failed_count += 1
                logger.warning(f"Image generation {i + 1}/{n} failed.")
                continue

            if result is None:
                result = gen_result
            else:
                result += gen_result

        if failed_count > 0 and result is not None:
            logger.warning(
                f"Generated {n - failed_count}/{n} images. "
                f"{failed_count} generation(s) failed."
            )

        return result


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ImageGen:
    images: List[Image.Image] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    usages: List[Usage] = field(default_factory=list)
    subimages: List[Image.Image] = field(default_factory=list)
    subimage_paths: List[str] = field(default_factory=list)

    @property
    def image(self) -> Optional[Image.Image]:
        return self.images[-1] if self.images else None

    @property
    def image_path(self) -> Optional[str]:
        return self.image_paths[-1] if self.image_paths else None

    @property
    def usage(self) -> Optional[Usage]:
        return self.usages[0] if self.usages else None

    def __add__(self, other: "ImageGen") -> "ImageGen":
        if isinstance(other, ImageGen):
            return ImageGen(
                images=self.images + other.images,
                image_paths=self.image_paths + other.image_paths,
                usages=self.usages + other.usages,
                subimages=self.subimages + other.subimages,
                subimage_paths=self.subimage_paths + other.subimage_paths,
            )
        raise TypeError("Can only add ImageGen instances.")

    def __repr__(self) -> str:
        img_info = f"images={len(self.images)}"
        if self.images:
            img = self.images[0]
            img_info += f" ({img.width}x{img.height})"
        subimg_info = ""
        if self.subimages:
            subimg = self.subimages[0]
            subimg_info = (
                f", subimages={len(self.subimages)} ({subimg.width}x{subimg.height})"
            )
        usage_info = ""
        if self.usages:
            total_tokens = sum(u.total_tokens for u in self.usages)
            usage_info = f", total_tokens={total_tokens}"
        return f"ImageGen({img_info}{subimg_info}{usage_info})"
