"""
Nano Banana Pro MCP Server - Image generation and editing via Vertex AI.

Provides MCP tools for generating and editing images using Google's
Gemini 3 Pro Image (Nano Banana Pro) model through the Vertex AI API.

Author: Terry.Kim <th.kim@lgdisplay.com>
Co-Author: Claudie
"""

import os
import json
import base64
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastmcp import FastMCP
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "gemini-3-pro-image-preview"

# Output directory for generated images (override with env var)
OUTPUT_DIR = Path(os.environ.get(
    "NANOBANANA_OUTPUT_DIR",
    str(Path.home() / "nanobanana_output"),
))

# Vertex AI settings (can be set via env vars)
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
USE_VERTEX_AI = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() == "true"

# Valid option values
VALID_ASPECT_RATIOS = {"1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"}
VALID_IMAGE_SIZES = {"1K", "2K", "4K"}
VALID_PERSON_GENERATION = {
    "DONT_ALLOW",
    "ALLOW_ADULT",
    "ALLOW_ALL",
}
VALID_SAFETY_LEVELS = {
    "BLOCK_LOW_AND_ABOVE",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_NONE",
}

MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------

_client = None


def _get_client() -> genai.Client:
    """Lazy-initialize the GenAI client."""
    global _client
    if _client is None:
        if USE_VERTEX_AI:
            _client = genai.Client(
                vertexai=True,
                project=VERTEX_PROJECT,
                location=VERTEX_LOCATION,
            )
        else:
            # Uses GEMINI_API_KEY env var automatically
            _client = genai.Client()
    return _client


def _ensure_output_dir() -> Path:
    """Ensure the output directory exists and return its path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _save_image(image_data: bytes, prefix: str = "generated") -> str:
    """Save image bytes to a file and return the absolute path."""
    out_dir = _ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    filename = f"{prefix}_{timestamp}_{short_id}.png"
    filepath = out_dir / filename
    filepath.write_bytes(image_data)
    return str(filepath)


def _build_config(
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    number_of_images: int = 1,
    person_generation: Optional[str] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    safety_level: Optional[str] = None,
) -> types.GenerateContentConfig:
    """Build GenerateContentConfig with full ImageConfig options."""
    image_cfg_kwargs = {
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
    }
    if person_generation is not None:
        image_cfg_kwargs["person_generation"] = person_generation

    config_kwargs = {
        "response_modalities": ["TEXT", "IMAGE"],
        "image_config": types.ImageConfig(**image_cfg_kwargs),
        "candidate_count": number_of_images,
    }

    if temperature is not None:
        config_kwargs["temperature"] = temperature
    if seed is not None:
        config_kwargs["seed"] = seed

    if safety_level is not None:
        config_kwargs["safety_settings"] = [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold=safety_level,
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold=safety_level,
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold=safety_level,
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold=safety_level,
            ),
        ]

    return types.GenerateContentConfig(**config_kwargs)


def _extract_results(response, output_format: str, prefix: str) -> dict:
    """Extract text and images from a generate_content response."""
    images = []
    text = ""

    for part in response.candidates[0].content.parts:
        if part.text:
            text += part.text
        elif part.inline_data:
            if output_format == "base64":
                b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                images.append({
                    "format": "base64",
                    "mime_type": part.inline_data.mime_type,
                    "data": b64,
                })
            else:
                filepath = _save_image(part.inline_data.data, prefix=prefix)
                images.append({
                    "format": "file",
                    "path": filepath,
                    "mime_type": part.inline_data.mime_type,
                })

    return {"images": images, "text": text}


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("nanobanana-pro")


@mcp.tool()
def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    number_of_images: int = 1,
    person_generation: Optional[str] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    safety_level: Optional[str] = None,
    output_format: str = "file",
) -> str:
    """Generate an image from a text prompt using Nano Banana Pro (Gemini 3 Pro Image).

    Creates high-fidelity images with reasoning-enhanced composition,
    legible text rendering, and up to 4K resolution.

    Args:
        prompt: Text description of the image to generate.
                Be specific about style, composition, lighting, and details.
        aspect_ratio: Aspect ratio of the output image.
                      Options: "1:1", "2:3", "3:2", "3:4", "4:3",
                      "9:16", "16:9", "21:9". Default: "1:1".
        image_size: Output resolution. Options: "1K", "2K", "4K".
                    Default: "1K".
        number_of_images: Number of images to generate (1-4). Default: 1.
        person_generation: Controls people generation.
                           Options: "DONT_ALLOW", "ALLOW_ADULT", "ALLOW_ALL".
                           Default: model default (None).
        temperature: Controls randomness (0.0-2.0). Google recommends 1.0
                     for image generation. Default: model default (None).
        seed: Random seed for reproducible results. Use the same seed
              with the same prompt to get similar outputs. Default: None.
        safety_level: Safety filter threshold applied to all harm categories.
                      Options: "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE",
                      "BLOCK_ONLY_HIGH", "BLOCK_NONE". Default: model default (None).
        output_format: "file" to save to disk and return path,
                       "base64" to return base64-encoded PNG data. Default: "file".

    Returns:
        JSON with generated image path(s) (or base64 data), model response text,
        and metadata.
    """
    client = _get_client()
    config = _build_config(
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        number_of_images=number_of_images,
        person_generation=person_generation,
        temperature=temperature,
        seed=seed,
        safety_level=safety_level,
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=config,
    )

    extracted = _extract_results(response, output_format, prefix="gen")
    result = {
        "prompt": prompt,
        "model": MODEL_ID,
        "settings": {
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "number_of_images": number_of_images,
        },
        **extracted,
    }
    if person_generation:
        result["settings"]["person_generation"] = person_generation
    if temperature is not None:
        result["settings"]["temperature"] = temperature
    if seed is not None:
        result["settings"]["seed"] = seed
    if safety_level:
        result["settings"]["safety_level"] = safety_level

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def edit_image(
    image_path: str,
    instruction: str,
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    person_generation: Optional[str] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    safety_level: Optional[str] = None,
    output_format: str = "file",
) -> str:
    """Edit an existing image using natural language instructions.

    Supports transformations like style changes, object removal/addition,
    background changes, color adjustments, and more.

    Args:
        image_path: Absolute path to the source image file.
        instruction: Natural language editing instruction.
                     Examples: "Make it look like a watercolor painting",
                     "Remove the background", "Change the sky to sunset colors"
        aspect_ratio: Aspect ratio of the output image.
                      Options: "1:1", "2:3", "3:2", "3:4", "4:3",
                      "9:16", "16:9", "21:9". Default: "1:1".
        image_size: Output resolution. Options: "1K", "2K", "4K".
                    Default: "1K".
        person_generation: Controls people generation.
                           Options: "DONT_ALLOW", "ALLOW_ADULT", "ALLOW_ALL".
                           Default: model default (None).
        temperature: Controls randomness (0.0-2.0). Default: model default (None).
        seed: Random seed for reproducible results. Default: None.
        safety_level: Safety filter threshold applied to all harm categories.
                      Options: "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE",
                      "BLOCK_ONLY_HIGH", "BLOCK_NONE". Default: model default (None).
        output_format: "file" to save to disk and return path,
                       "base64" to return base64-encoded PNG data. Default: "file".

    Returns:
        JSON with edited image path (or base64 data), model response text,
        and metadata.
    """
    src = Path(image_path)
    if not src.exists():
        return json.dumps({"error": f"Source image not found: {image_path}"})

    image_bytes = src.read_bytes()
    mime_type = MIME_MAP.get(src.suffix.lower(), "image/png")

    client = _get_client()
    config = _build_config(
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        number_of_images=1,
        person_generation=person_generation,
        temperature=temperature,
        seed=seed,
        safety_level=safety_level,
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            instruction,
        ],
        config=config,
    )

    extracted = _extract_results(response, output_format, prefix="edit")
    result = {
        "instruction": instruction,
        "source_image": image_path,
        "model": MODEL_ID,
        **extracted,
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def generate_with_references(
    prompt: str,
    reference_paths: list[str],
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    number_of_images: int = 1,
    person_generation: Optional[str] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    safety_level: Optional[str] = None,
    output_format: str = "file",
) -> str:
    """Generate an image using text prompt and reference images for consistency.

    Nano Banana Pro supports up to 14 reference inputs for maintaining
    character consistency, style matching, and compositional guidance.

    Args:
        prompt: Text description of the image to generate, referencing
                the provided images for style/character consistency.
        reference_paths: List of absolute paths to reference image files
                         (max 14 images).
        aspect_ratio: Aspect ratio of the output image.
                      Options: "1:1", "2:3", "3:2", "3:4", "4:3",
                      "9:16", "16:9", "21:9". Default: "1:1".
        image_size: Output resolution. Options: "1K", "2K", "4K".
                    Default: "1K".
        number_of_images: Number of images to generate (1-4). Default: 1.
        person_generation: Controls people generation.
                           Options: "DONT_ALLOW", "ALLOW_ADULT", "ALLOW_ALL".
                           Default: model default (None).
        temperature: Controls randomness (0.0-2.0). Default: model default (None).
        seed: Random seed for reproducible results. Default: None.
        safety_level: Safety filter threshold applied to all harm categories.
                      Options: "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE",
                      "BLOCK_ONLY_HIGH", "BLOCK_NONE". Default: model default (None).
        output_format: "file" to save to disk and return path,
                       "base64" to return base64-encoded PNG data. Default: "file".

    Returns:
        JSON with generated image path(s) (or base64 data), model response text,
        and metadata.
    """
    if len(reference_paths) > 14:
        return json.dumps({
            "error": "Nano Banana Pro supports a maximum of 14 reference images.",
        })

    # Build content parts: reference images + text prompt
    content_parts = []
    for ref_path in reference_paths:
        ref = Path(ref_path)
        if not ref.exists():
            return json.dumps({"error": f"Reference image not found: {ref_path}"})
        mime_type = MIME_MAP.get(ref.suffix.lower(), "image/png")
        content_parts.append(
            types.Part.from_bytes(data=ref.read_bytes(), mime_type=mime_type)
        )

    content_parts.append(prompt)

    client = _get_client()
    config = _build_config(
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        number_of_images=number_of_images,
        person_generation=person_generation,
        temperature=temperature,
        seed=seed,
        safety_level=safety_level,
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=content_parts,
        config=config,
    )

    extracted = _extract_results(response, output_format, prefix="ref")
    result = {
        "prompt": prompt,
        "reference_count": len(reference_paths),
        "model": MODEL_ID,
        **extracted,
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def list_generated_images(limit: int = 20) -> str:
    """List recently generated images in the output directory.

    Args:
        limit: Maximum number of files to return (default: 20, most recent first).

    Returns:
        JSON with list of image files sorted by modification time (newest first).
    """
    out_dir = _ensure_output_dir()
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}

    files = [
        f for f in out_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_exts
    ]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    files = files[:limit]

    result = {
        "output_dir": str(out_dir),
        "count": len(files),
        "files": [
            {
                "name": f.name,
                "path": str(f),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            }
            for f in files
        ],
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_supported_options() -> str:
    """Get all supported configuration options for Nano Banana Pro.

    Returns a reference of all available parameters, valid values,
    and their descriptions.
    """
    options = {
        "model": MODEL_ID,
        "output_dir": str(OUTPUT_DIR),
        "auth_mode": "vertex_ai" if USE_VERTEX_AI else "api_key",
        "parameters": {
            "aspect_ratio": {
                "values": sorted(VALID_ASPECT_RATIOS),
                "default": "1:1",
                "description": "Output image aspect ratio.",
            },
            "image_size": {
                "values": sorted(VALID_IMAGE_SIZES),
                "default": "1K",
                "description": "Output resolution. 4K for maximum quality.",
            },
            "number_of_images": {
                "range": "1-4",
                "default": 1,
                "description": "Number of images to generate per request.",
            },
            "person_generation": {
                "values": sorted(VALID_PERSON_GENERATION),
                "default": "model default",
                "description": "Controls generation of people/faces.",
            },
            "temperature": {
                "range": "0.0-2.0",
                "default": "model default (1.0 recommended)",
                "description": "Controls randomness. Higher = more creative.",
            },
            "seed": {
                "type": "integer",
                "default": "random",
                "description": "Fixed seed for reproducible results.",
            },
            "safety_level": {
                "values": sorted(VALID_SAFETY_LEVELS),
                "default": "model default",
                "description": "Safety filter threshold for all harm categories.",
            },
            "output_format": {
                "values": ["file", "base64"],
                "default": "file",
                "description": "'file' saves to disk, 'base64' returns encoded data.",
            },
        },
        "env_vars": {
            "NANOBANANA_OUTPUT_DIR": "Override output directory path.",
            "GOOGLE_CLOUD_PROJECT": "GCP project ID (Vertex AI mode).",
            "GOOGLE_CLOUD_LOCATION": "GCP region (default: global).",
            "GOOGLE_GENAI_USE_VERTEXAI": "Set 'true' for Vertex AI, 'false' for API key.",
            "GEMINI_API_KEY": "Gemini API key (when not using Vertex AI).",
        },
    }

    return json.dumps(options, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
