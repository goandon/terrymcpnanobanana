"""
Nano Banana MCP Server - Image generation and editing via Vertex AI.

Provides MCP tools for generating and editing images using Google's
Nano Banana models (Pro and Flash/Nano Banana 2) through the Vertex AI API.

Supported models:
  - Nano Banana 2 (Gemini 3.1 Flash Image) — fast, cost-effective (default)
  - Nano Banana Pro (Gemini 3 Pro Image) — highest quality

Author: Terry.Kim <goandonh@gmail.com>
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

# Model registry
MODELS = {
    "flash": "gemini-3.1-flash-image-preview",  # Nano Banana 2 (default)
    "pro": "gemini-3-pro-image-preview",         # Nano Banana Pro
}
DEFAULT_MODEL = "flash"

# Output directory for generated images (override with env var)
OUTPUT_DIR = Path(os.environ.get(
    "NANOBANANA_OUTPUT_DIR",
    str(Path.home() / "nanobanana_output"),
))

# Vertex AI settings (can be set via env vars)
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
USE_VERTEX_AI = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() == "true"

# Valid option values (model-specific)
VALID_ASPECT_RATIOS_FLASH = {
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4",
    "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
}
VALID_ASPECT_RATIOS_PRO = {"1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"}

VALID_IMAGE_SIZES_FLASH = {"512px", "1K", "2K", "4K"}
VALID_IMAGE_SIZES_PRO = {"1K", "2K", "4K"}

VALID_THINKING_LEVELS = {"minimal", "High"}

VALID_PERSON_GENERATION = {
    "DONT_ALLOW",
    "ALLOW_NONE",  # SDK alias for DONT_ALLOW
    "ALLOW_ADULT",
    "ALLOW_ALL",
}
VALID_PROMINENT_PEOPLE = {"ALLOW", "DENY"}
VALID_SAFETY_LEVELS = {
    "BLOCK_LOW_AND_ABOVE",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_NONE",
}
VALID_OUTPUT_FORMATS = {"file", "base64"}

# Image output defaults
OUTPUT_MIME_TYPE = "image/jpeg"
OUTPUT_COMPRESSION_QUALITY = 85

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


def _resolve_model(model: str) -> str:
    """Resolve a short model key ('flash'/'pro') to the full model ID."""
    key = model.lower().strip()
    if key not in MODELS:
        raise ValueError(
            f"Unknown model '{model}'. Valid options: {sorted(MODELS.keys())}"
        )
    return MODELS[key]


def _validate_params(
    model_key: str,
    aspect_ratio: str,
    image_size: str,
    output_format: str = "file",
    person_generation: Optional[str] = None,
    prominent_people: Optional[str] = None,
    safety_level: Optional[str] = None,
    thinking_level: Optional[str] = None,
    number_of_images: int = 1,
    temperature: Optional[float] = None,
) -> list[str]:
    """Validate parameters against allowed values. Returns list of errors."""
    errors = []
    is_flash = model_key.lower() == "flash"
    valid_ratios = VALID_ASPECT_RATIOS_FLASH if is_flash else VALID_ASPECT_RATIOS_PRO
    valid_sizes = VALID_IMAGE_SIZES_FLASH if is_flash else VALID_IMAGE_SIZES_PRO

    if aspect_ratio not in valid_ratios:
        errors.append(
            f"Invalid aspect_ratio '{aspect_ratio}' for {model_key}. "
            f"Valid: {sorted(valid_ratios)}"
        )
    if image_size not in valid_sizes:
        errors.append(
            f"Invalid image_size '{image_size}' for {model_key}. "
            f"Valid: {sorted(valid_sizes)}"
        )
    if output_format not in VALID_OUTPUT_FORMATS:
        errors.append(
            f"Invalid output_format '{output_format}'. Valid: {sorted(VALID_OUTPUT_FORMATS)}"
        )
    if person_generation is not None and person_generation not in VALID_PERSON_GENERATION:
        errors.append(
            f"Invalid person_generation '{person_generation}'. "
            f"Valid: {sorted(VALID_PERSON_GENERATION)}"
        )
    if prominent_people is not None and prominent_people not in VALID_PROMINENT_PEOPLE:
        errors.append(
            f"Invalid prominent_people '{prominent_people}'. "
            f"Valid: {sorted(VALID_PROMINENT_PEOPLE)}"
        )
    if safety_level is not None and safety_level not in VALID_SAFETY_LEVELS:
        errors.append(
            f"Invalid safety_level '{safety_level}'. Valid: {sorted(VALID_SAFETY_LEVELS)}"
        )
    if thinking_level is not None:
        if not is_flash:
            errors.append("thinking_level is only supported with the 'flash' model.")
        elif thinking_level not in VALID_THINKING_LEVELS:
            errors.append(
                f"Invalid thinking_level '{thinking_level}'. "
                f"Valid: {sorted(VALID_THINKING_LEVELS)}"
            )
    if not 1 <= number_of_images <= 4:
        errors.append(f"number_of_images must be 1-4, got {number_of_images}.")
    if temperature is not None and not 0.0 <= temperature <= 2.0:
        errors.append(f"temperature must be 0.0-2.0, got {temperature}.")

    return errors


def _ensure_output_dir() -> Path:
    """Ensure the output directory exists and return its path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _save_image(image_data: bytes, prefix: str = "generated") -> str:
    """Save image bytes to a file and return the absolute path."""
    out_dir = _ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    filename = f"{prefix}_{timestamp}_{short_id}.jpg"
    filepath = out_dir / filename
    filepath.write_bytes(image_data)
    return str(filepath)


def _build_config(
    model_key: str = DEFAULT_MODEL,
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    number_of_images: int = 1,
    person_generation: Optional[str] = None,
    prominent_people: Optional[str] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    safety_level: Optional[str] = None,
    thinking_level: Optional[str] = None,
    use_search: bool = False,
) -> types.GenerateContentConfig:
    """Build GenerateContentConfig with full ImageConfig options."""
    is_flash = model_key.lower() == "flash"

    image_cfg_kwargs = {
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
    }

    # These parameters are Vertex AI only — the SDK raises errors in API key mode.
    if USE_VERTEX_AI:
        image_cfg_kwargs["output_mime_type"] = OUTPUT_MIME_TYPE
        image_cfg_kwargs["output_compression_quality"] = OUTPUT_COMPRESSION_QUALITY
        if person_generation is not None:
            image_cfg_kwargs["person_generation"] = person_generation
        if prominent_people is not None:
            image_cfg_kwargs["prominent_people"] = prominent_people

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

    # Flash-only: thinking configuration
    if is_flash and thinking_level is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level,
            include_thoughts=True,
        )

    # Flash-only: Google Search grounding (web + image search)
    if is_flash and use_search:
        config_kwargs["tools"] = [types.Tool(
            google_search=types.GoogleSearch(
                search_types=types.SearchTypes(
                    web_search=types.WebSearch(),
                    image_search=types.ImageSearch(),
                )
            )
        )]

    return types.GenerateContentConfig(**config_kwargs)


def _extract_results(response, output_format: str, prefix: str) -> dict:
    """Extract text and images from a generate_content response."""
    images = []
    text = ""
    thought = ""

    for part in response.candidates[0].content.parts:
        # Skip thinking/thought parts (Flash model with thinking enabled)
        if hasattr(part, "thought") and part.thought:
            thought += part.text or ""
            continue
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

    result = {"images": images, "text": text}
    if thought:
        result["thinking"] = thought
    return result


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("nanobanana")


@mcp.tool()
def generate_image(
    prompt: str,
    model: str = DEFAULT_MODEL,
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    number_of_images: int = 1,
    person_generation: Optional[str] = None,
    prominent_people: Optional[str] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    safety_level: Optional[str] = None,
    thinking_level: Optional[str] = None,
    use_search: bool = False,
    output_format: str = "file",
) -> str:
    """Generate an image from a text prompt using Nano Banana models.

    Supports two models:
      - "flash" (default): Nano Banana 2 (Gemini 3.1 Flash Image) — fast, cost-effective
      - "pro": Nano Banana Pro (Gemini 3 Pro Image) — highest quality

    Args:
        prompt: Text description of the image to generate.
                Be specific about style, composition, lighting, and details.
        model: Model to use. "flash" (default, Nano Banana 2) or "pro" (Nano Banana Pro).
        aspect_ratio: Aspect ratio of the output image.
                      Common: "1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "21:9".
                      Flash also supports: "1:4", "4:1", "1:8", "8:1", "4:5", "5:4".
                      Default: "1:1".
        image_size: Output resolution. "1K", "2K", "4K".
                    Flash also supports "512px". Default: "1K".
        number_of_images: Number of images to generate (1-4). Default: 1.
        person_generation: Controls people generation.
                           Options: "DONT_ALLOW"/"ALLOW_NONE", "ALLOW_ADULT", "ALLOW_ALL".
                           Default: model default (None).
        prominent_people: Controls celebrity/prominent person generation separately.
                          Options: "ALLOW", "DENY". Overridden by person_generation
                          if both are set. Default: None.
        temperature: Controls randomness (0.0-2.0). Google recommends 1.0
                     for image generation. Default: model default (None).
        seed: Random seed for reproducible results. Use the same seed
              with the same prompt to get similar outputs. Default: None.
        safety_level: Safety filter threshold applied to all harm categories.
                      Options: "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE",
                      "BLOCK_ONLY_HIGH", "BLOCK_NONE". Default: model default (None).
        thinking_level: (Flash only) Thinking effort level.
                        "minimal" (default) or "High" for better composition.
                        Thinking tokens incur charges. Default: None (off).
        use_search: (Flash only) Enable Google Search grounding (web + image)
                    for accurate rendering of real subjects/places. Default: False.
        output_format: "file" to save to disk and return path,
                       "base64" to return base64-encoded data. Default: "file".

    Returns:
        JSON with generated image path(s) (or base64 data), model response text,
        and metadata.
    """
    model_id = _resolve_model(model)
    errors = _validate_params(
        model_key=model, aspect_ratio=aspect_ratio, image_size=image_size,
        output_format=output_format, person_generation=person_generation,
        prominent_people=prominent_people, safety_level=safety_level,
        thinking_level=thinking_level, number_of_images=number_of_images,
        temperature=temperature,
    )
    if errors:
        return json.dumps({"errors": errors})

    client = _get_client()
    config = _build_config(
        model_key=model,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        number_of_images=number_of_images,
        person_generation=person_generation,
        prominent_people=prominent_people,
        temperature=temperature,
        seed=seed,
        safety_level=safety_level,
        thinking_level=thinking_level,
        use_search=use_search,
    )

    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=config,
    )

    extracted = _extract_results(response, output_format, prefix="gen")
    result = {
        "prompt": prompt,
        "model": model_id,
        "settings": {
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "number_of_images": number_of_images,
        },
        **extracted,
    }
    if person_generation:
        result["settings"]["person_generation"] = person_generation
    if prominent_people:
        result["settings"]["prominent_people"] = prominent_people
    if temperature is not None:
        result["settings"]["temperature"] = temperature
    if seed is not None:
        result["settings"]["seed"] = seed
    if safety_level:
        result["settings"]["safety_level"] = safety_level
    if thinking_level:
        result["settings"]["thinking_level"] = thinking_level
    if use_search:
        result["settings"]["use_search"] = True

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def edit_image(
    image_path: str,
    instruction: str,
    model: str = DEFAULT_MODEL,
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    person_generation: Optional[str] = None,
    prominent_people: Optional[str] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    safety_level: Optional[str] = None,
    thinking_level: Optional[str] = None,
    use_search: bool = False,
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
        model: Model to use. "flash" (default, Nano Banana 2) or "pro" (Nano Banana Pro).
        aspect_ratio: Aspect ratio of the output image.
                      Common: "1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "21:9".
                      Flash also supports: "1:4", "4:1", "1:8", "8:1", "4:5", "5:4".
                      Default: "1:1".
        image_size: Output resolution. "1K", "2K", "4K".
                    Flash also supports "512px". Default: "1K".
        person_generation: Controls people generation.
                           Options: "DONT_ALLOW"/"ALLOW_NONE", "ALLOW_ADULT", "ALLOW_ALL".
                           Default: model default (None).
        prominent_people: Controls celebrity/prominent person generation.
                          Options: "ALLOW", "DENY". Default: None.
        temperature: Controls randomness (0.0-2.0). Default: model default (None).
        seed: Random seed for reproducible results. Default: None.
        safety_level: Safety filter threshold applied to all harm categories.
                      Options: "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE",
                      "BLOCK_ONLY_HIGH", "BLOCK_NONE". Default: model default (None).
        thinking_level: (Flash only) Thinking effort level.
                        "minimal" or "High". Default: None (off).
        use_search: (Flash only) Enable Google Search grounding (web + image).
                    Default: False.
        output_format: "file" to save to disk and return path,
                       "base64" to return base64-encoded data. Default: "file".

    Returns:
        JSON with edited image path (or base64 data), model response text,
        and metadata.
    """
    model_id = _resolve_model(model)
    errors = _validate_params(
        model_key=model, aspect_ratio=aspect_ratio, image_size=image_size,
        output_format=output_format, person_generation=person_generation,
        prominent_people=prominent_people, safety_level=safety_level,
        thinking_level=thinking_level, temperature=temperature,
    )
    if errors:
        return json.dumps({"errors": errors})

    src = Path(image_path)
    if not src.exists():
        return json.dumps({"error": f"Source image not found: {image_path}"})

    image_bytes = src.read_bytes()
    mime_type = MIME_MAP.get(src.suffix.lower(), "image/png")

    client = _get_client()
    config = _build_config(
        model_key=model,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        number_of_images=1,
        person_generation=person_generation,
        prominent_people=prominent_people,
        temperature=temperature,
        seed=seed,
        safety_level=safety_level,
        thinking_level=thinking_level,
        use_search=use_search,
    )

    response = client.models.generate_content(
        model=model_id,
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
        "model": model_id,
        "settings": {
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
        },
        **extracted,
    }
    if person_generation:
        result["settings"]["person_generation"] = person_generation
    if prominent_people:
        result["settings"]["prominent_people"] = prominent_people
    if temperature is not None:
        result["settings"]["temperature"] = temperature
    if seed is not None:
        result["settings"]["seed"] = seed
    if safety_level:
        result["settings"]["safety_level"] = safety_level
    if thinking_level:
        result["settings"]["thinking_level"] = thinking_level
    if use_search:
        result["settings"]["use_search"] = True

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def generate_with_references(
    prompt: str,
    reference_paths: list[str],
    model: str = DEFAULT_MODEL,
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    number_of_images: int = 1,
    person_generation: Optional[str] = None,
    prominent_people: Optional[str] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    safety_level: Optional[str] = None,
    thinking_level: Optional[str] = None,
    use_search: bool = False,
    output_format: str = "file",
) -> str:
    """Generate an image using text prompt and reference images for consistency.

    Use reference images for character consistency, style matching, and
    compositional guidance.

    Args:
        prompt: Text description of the image to generate, referencing
                the provided images for style/character consistency.
        reference_paths: List of absolute paths to reference image files.
                         Pro supports up to 14, Flash up to 10.
        model: Model to use. "flash" (default, Nano Banana 2) or "pro" (Nano Banana Pro).
        aspect_ratio: Aspect ratio of the output image.
                      Common: "1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "21:9".
                      Flash also supports: "1:4", "4:1", "1:8", "8:1", "4:5", "5:4".
                      Default: "1:1".
        image_size: Output resolution. "1K", "2K", "4K".
                    Flash also supports "512px". Default: "1K".
        number_of_images: Number of images to generate (1-4). Default: 1.
        person_generation: Controls people generation.
                           Options: "DONT_ALLOW"/"ALLOW_NONE", "ALLOW_ADULT", "ALLOW_ALL".
                           Default: model default (None).
        prominent_people: Controls celebrity/prominent person generation.
                          Options: "ALLOW", "DENY". Default: None.
        temperature: Controls randomness (0.0-2.0). Default: model default (None).
        seed: Random seed for reproducible results. Default: None.
        safety_level: Safety filter threshold applied to all harm categories.
                      Options: "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE",
                      "BLOCK_ONLY_HIGH", "BLOCK_NONE". Default: model default (None).
        thinking_level: (Flash only) Thinking effort level.
                        "minimal" or "High". Default: None (off).
        use_search: (Flash only) Enable Google Search grounding (web + image).
                    Default: False.
        output_format: "file" to save to disk and return path,
                       "base64" to return base64-encoded data. Default: "file".

    Returns:
        JSON with generated image path(s) (or base64 data), model response text,
        and metadata.
    """
    model_id = _resolve_model(model)
    errors = _validate_params(
        model_key=model, aspect_ratio=aspect_ratio, image_size=image_size,
        output_format=output_format, person_generation=person_generation,
        prominent_people=prominent_people, safety_level=safety_level,
        thinking_level=thinking_level, number_of_images=number_of_images,
        temperature=temperature,
    )
    if errors:
        return json.dumps({"errors": errors})

    max_refs = 10 if model.lower() == "flash" else 14
    if len(reference_paths) > max_refs:
        return json.dumps({
            "error": f"{MODELS[model.lower()]} supports a maximum of {max_refs} reference images.",
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
        model_key=model,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        number_of_images=number_of_images,
        person_generation=person_generation,
        prominent_people=prominent_people,
        temperature=temperature,
        seed=seed,
        safety_level=safety_level,
        thinking_level=thinking_level,
        use_search=use_search,
    )

    response = client.models.generate_content(
        model=model_id,
        contents=content_parts,
        config=config,
    )

    extracted = _extract_results(response, output_format, prefix="ref")
    result = {
        "prompt": prompt,
        "reference_count": len(reference_paths),
        "model": model_id,
        "settings": {
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "number_of_images": number_of_images,
        },
        **extracted,
    }
    if person_generation:
        result["settings"]["person_generation"] = person_generation
    if prominent_people:
        result["settings"]["prominent_people"] = prominent_people
    if temperature is not None:
        result["settings"]["temperature"] = temperature
    if seed is not None:
        result["settings"]["seed"] = seed
    if safety_level:
        result["settings"]["safety_level"] = safety_level
    if thinking_level:
        result["settings"]["thinking_level"] = thinking_level
    if use_search:
        result["settings"]["use_search"] = True

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
    """Get all supported configuration options for Nano Banana models.

    Returns a reference of all available parameters, valid values,
    and their descriptions for both Flash and Pro models.
    """
    options = {
        "models": {
            "flash": {
                "id": MODELS["flash"],
                "name": "Nano Banana 2 (Gemini 3.1 Flash Image)",
                "default": True,
                "aspect_ratios": sorted(VALID_ASPECT_RATIOS_FLASH),
                "image_sizes": sorted(VALID_IMAGE_SIZES_FLASH),
                "max_reference_images": 10,
                "exclusive_features": ["thinking_level", "use_search"],
            },
            "pro": {
                "id": MODELS["pro"],
                "name": "Nano Banana Pro (Gemini 3 Pro Image)",
                "default": False,
                "aspect_ratios": sorted(VALID_ASPECT_RATIOS_PRO),
                "image_sizes": sorted(VALID_IMAGE_SIZES_PRO),
                "max_reference_images": 14,
                "exclusive_features": [],
            },
        },
        "output_dir": str(OUTPUT_DIR),
        "auth_mode": "vertex_ai" if USE_VERTEX_AI else "api_key",
        "common_parameters": {
            "model": {
                "values": sorted(MODELS.keys()),
                "default": DEFAULT_MODEL,
                "description": "Model selection: 'flash' (fast, cheap) or 'pro' (highest quality).",
            },
            "aspect_ratio": {
                "default": "1:1",
                "description": "Output image aspect ratio. Flash supports more options.",
            },
            "image_size": {
                "default": "1K",
                "description": "Output resolution. Flash also supports '512px'.",
            },
            "number_of_images": {
                "range": "1-4",
                "default": 1,
                "description": "Number of images to generate per request.",
            },
            "person_generation": {
                "values": sorted(VALID_PERSON_GENERATION),
                "default": "model default",
                "description": "Controls generation of people/faces. ALLOW_NONE is an SDK alias for DONT_ALLOW.",
            },
            "prominent_people": {
                "values": sorted(VALID_PROMINENT_PEOPLE),
                "default": "model default",
                "description": "Controls generation of celebrity/prominent people. Overridden by person_generation if both set.",
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
        "flash_only_parameters": {
            "thinking_level": {
                "values": sorted(VALID_THINKING_LEVELS),
                "default": "None (off)",
                "description": "Thinking effort for better composition. Tokens incur charges.",
            },
            "use_search": {
                "type": "boolean",
                "default": False,
                "description": "Enable Google Search grounding for real subjects/places.",
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
    mcp.run(transport="stdio")
