"""
Nano Banana Pro MCP Server - Image generation and editing via Vertex AI.

Provides MCP tools for generating and editing images using Google's
Gemini 3 Pro Image (Nano Banana Pro) model through the Vertex AI API.

Author: Terry.Kim <th.kim@lgdisplay.com>
Co-Author: Claudie
"""

import os
import sys
import json
import base64
import uuid
from io import BytesIO
from pathlib import Path
from datetime import datetime

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


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("nanobanana-pro")


@mcp.tool()
def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    output_format: str = "file",
) -> str:
    """Generate an image from a text prompt using Nano Banana Pro (Gemini 3 Pro Image).

    Creates high-fidelity images with reasoning-enhanced composition,
    legible text rendering, and up to 4K resolution.

    Args:
        prompt: Text description of the image to generate.
                Be specific about style, composition, lighting, and details.
        aspect_ratio: Aspect ratio of the output image.
                      Options: "1:1", "3:4", "4:3", "9:16", "16:9"
        output_format: "file" to save to disk and return path,
                       "base64" to return base64-encoded PNG data.

    Returns:
        JSON with generated image path (or base64 data), model response text,
        and metadata.
    """
    client = _get_client()

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
            ),
        ),
    )

    result = {"prompt": prompt, "model": MODEL_ID, "images": [], "text": ""}

    for part in response.candidates[0].content.parts:
        if part.text:
            result["text"] += part.text
        elif part.inline_data:
            if output_format == "base64":
                b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                result["images"].append({
                    "format": "base64",
                    "mime_type": part.inline_data.mime_type,
                    "data": b64,
                })
            else:
                filepath = _save_image(part.inline_data.data, prefix="gen")
                result["images"].append({
                    "format": "file",
                    "path": filepath,
                    "mime_type": part.inline_data.mime_type,
                })

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def edit_image(
    image_path: str,
    instruction: str,
    aspect_ratio: str = "1:1",
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
                      Options: "1:1", "3:4", "4:3", "9:16", "16:9"
        output_format: "file" to save to disk and return path,
                       "base64" to return base64-encoded PNG data.

    Returns:
        JSON with edited image path (or base64 data), model response text,
        and metadata.
    """
    src = Path(image_path)
    if not src.exists():
        return json.dumps({"error": f"Source image not found: {image_path}"})

    image_bytes = src.read_bytes()

    # Determine MIME type from extension
    ext = src.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_map.get(ext, "image/png")

    client = _get_client()

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            instruction,
        ],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
            ),
        ),
    )

    result = {
        "instruction": instruction,
        "source_image": image_path,
        "model": MODEL_ID,
        "images": [],
        "text": "",
    }

    for part in response.candidates[0].content.parts:
        if part.text:
            result["text"] += part.text
        elif part.inline_data:
            if output_format == "base64":
                b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                result["images"].append({
                    "format": "base64",
                    "mime_type": part.inline_data.mime_type,
                    "data": b64,
                })
            else:
                filepath = _save_image(part.inline_data.data, prefix="edit")
                result["images"].append({
                    "format": "file",
                    "path": filepath,
                    "mime_type": part.inline_data.mime_type,
                })

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def generate_with_references(
    prompt: str,
    reference_paths: list[str],
    aspect_ratio: str = "1:1",
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
                      Options: "1:1", "3:4", "4:3", "9:16", "16:9"
        output_format: "file" to save to disk and return path,
                       "base64" to return base64-encoded PNG data.

    Returns:
        JSON with generated image path (or base64 data), model response text,
        and metadata.
    """
    if len(reference_paths) > 14:
        return json.dumps({
            "error": "Nano Banana Pro supports a maximum of 14 reference images.",
        })

    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }

    # Build content parts: reference images + text prompt
    content_parts = []
    for ref_path in reference_paths:
        ref = Path(ref_path)
        if not ref.exists():
            return json.dumps({"error": f"Reference image not found: {ref_path}"})
        ext = ref.suffix.lower()
        mime_type = mime_map.get(ext, "image/png")
        content_parts.append(
            types.Part.from_bytes(data=ref.read_bytes(), mime_type=mime_type)
        )

    content_parts.append(prompt)

    client = _get_client()

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=content_parts,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
            ),
        ),
    )

    result = {
        "prompt": prompt,
        "reference_count": len(reference_paths),
        "model": MODEL_ID,
        "images": [],
        "text": "",
    }

    for part in response.candidates[0].content.parts:
        if part.text:
            result["text"] += part.text
        elif part.inline_data:
            if output_format == "base64":
                b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                result["images"].append({
                    "format": "base64",
                    "mime_type": part.inline_data.mime_type,
                    "data": b64,
                })
            else:
                filepath = _save_image(part.inline_data.data, prefix="ref")
                result["images"].append({
                    "format": "file",
                    "path": filepath,
                    "mime_type": part.inline_data.mime_type,
                })

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


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
