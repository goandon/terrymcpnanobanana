# Nano Banana MCP Server

[한국어 문서 (Korean)](README_ko.md)

Image generation and editing MCP server powered by Google's **Nano Banana** models via Gemini API.

> **Nano Banana 2** (Gemini 3.1 Flash Image): `gemini-3.1-flash-image-preview` — fast, cost-effective (default)
> **Nano Banana Pro** (Gemini 3 Pro Image): `gemini-3-pro-image-preview` — highest quality

## Overview

This MCP (Model Context Protocol) server exposes Google's Nano Banana image generation models as tools for AI assistants such as Claude Desktop and Claude Code. Generate and edit images using natural language prompts.

### Key Features

- **Dual Model** — Switch between Flash (fast, cheap) and Pro (best quality) at runtime. Default: Flash
- **Text-to-Image** — Generate high-fidelity images up to 4K resolution
- **Image Editing** — Edit existing images with natural language instructions
- **Reference-based Generation** — Maintain character/style consistency with reference images (Flash: 10, Pro: 14)
- **Thinking Mode** — (Flash only) Enable reasoning for better composition quality
- **Search Grounding** — (Flash only) Google Search integration for accurate real-world subjects
- **Full Parameter Control** — Resolution, aspect ratio, temperature, seed, safety levels
- **Input Validation** — All parameters validated locally before API calls with clear error messages
- **Gemini API Key** — Simple API key authentication (no GCP project required)

## Model Comparison

| Feature | Flash (Nano Banana 2) | Pro (Nano Banana Pro) |
|---------|----------------------|----------------------|
| Model ID | `gemini-3.1-flash-image-preview` | `gemini-3-pro-image-preview` |
| Speed | ~4-6s | ~8-12s |
| Cost | ~$0.067/image | ~$0.13/image |
| Max Resolution | 4K | 4K |
| Aspect Ratios | 14 options | 8 options |
| Image Sizes | 512px, 1K, 2K, 4K | 1K, 2K, 4K |
| Reference Images | Up to 10 | Up to 14 |
| Thinking Mode | Yes | No |
| Search Grounding | Yes | No |

## Installation

```bash
cd terrymcpnanobanana
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastmcp` | >= 2.0.0 | MCP server framework |
| `google-genai` | >= 1.0.0 | Google Gen AI SDK |
| `Pillow` | >= 10.0.0 | Image processing |

## Authentication

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
export GEMINI_API_KEY=your-api-key-here
```

## MCP Client Configuration

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "nanobanana": {
      "command": "python",
      "args": ["/path/to/terrymcpnanobanana/server.py"],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here",
        "NANOBANANA_OUTPUT_DIR": "/path/to/output/folder"
      }
    }
  }
}
```

### Claude Code (`.mcp.json`)

```json
{
  "mcpServers": {
    "nanobanana": {
      "command": "python",
      "args": ["/path/to/terrymcpnanobanana/server.py"],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here",
        "NANOBANANA_OUTPUT_DIR": "/path/to/output/folder"
      }
    }
  }
}
```

### Windows

```json
{
  "mcpServers": {
    "nanobanana": {
      "command": "C:\\Users\\terry\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
      "args": ["C:\\path\\to\\terrymcpnanobanana\\server.py"],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here",
        "NANOBANANA_OUTPUT_DIR": "D:\\Images\\nanobanana"
      }
    }
  }
}
```

## Tools Reference

### 1. `generate_image` — Text to Image

Generate images from text prompts.

```
prompt: "A serene Japanese garden with cherry blossoms at sunset"
model: "flash"          # or "pro"
aspect_ratio: "16:9"
image_size: "4K"
number_of_images: 2
thinking_level: "High"  # Flash only: better composition
use_search: true        # Flash only: accurate real subjects
```

### 2. `edit_image` — Image Editing

Edit existing images with natural language instructions.

```
image_path: "/home/user/photo.jpg"
instruction: "Make it look like a watercolor painting"
model: "flash"
image_size: "2K"
use_search: true        # Flash only: reference real subjects
```

### 3. `generate_with_references` — Reference-based Generation

Generate images with character/style consistency using reference images.

```
prompt: "Same character sitting in a cafe, drinking coffee"
reference_paths: ["/home/user/char_ref1.png", "/home/user/char_ref2.png"]
model: "flash"           # max 10 refs (pro: max 14)
aspect_ratio: "3:4"
image_size: "2K"
```

### 4. `list_generated_images` — List Output

List recently generated images (most recent first).

```
limit: 10
```

### 5. `get_supported_options` — Parameter Reference

Returns all supported parameters and valid values for both models. No arguments required.

## Parameters

### Model Selection

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `model` | `flash`, `pro` | `flash` | Flash: fast & cheap. Pro: highest quality |

### Image Configuration

| Parameter | Flash | Pro | Default | Description |
|-----------|-------|-----|---------|-------------|
| `aspect_ratio` | `1:1` `1:4` `1:8` `2:3` `3:2` `3:4` `4:1` `4:3` `4:5` `5:4` `8:1` `9:16` `16:9` `21:9` | `1:1` `2:3` `3:2` `3:4` `4:3` `9:16` `16:9` `21:9` | `1:1` | Output aspect ratio |
| `image_size` | `512px` `1K` `2K` `4K` | `1K` `2K` `4K` | `1K` | Output resolution |
| `number_of_images` | 1-4 | 1-4 | 1 | Images per request |
| `output_format` | `file` `base64` | `file` `base64` | `file` | Save to disk or return encoded data |

### Generation Control

| Parameter | Range/Options | Default | Description |
|-----------|---------------|---------|-------------|
| `temperature` | 0.0 - 2.0 | model default | Randomness. Higher = more creative. Google recommends 1.0 |
| `seed` | integer | random | Fixed seed for reproducible results |

### Flash-only Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `thinking_level` | `minimal`, `High` | None (off) | Thinking mode for better composition. Incurs token charges |
| `use_search` | `true`, `false` | `false` | Google Search grounding for real subjects/places |

### Safety

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `safety_level` | `BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_ONLY_HIGH`, `BLOCK_NONE` | model default | Safety filter threshold (all harm categories) |

## Nano Banana 2 (Gemini 3.1 Flash Image) — Detailed Guide

Nano Banana 2 is the default model (`gemini-3.1-flash-image-preview`), based on **Gemini 3.1 Flash** with native image generation capabilities.

### Resolution & Output

| `image_size` | Pixel Resolution | Use Case |
|-------------|-----------------|----------|
| `512px` | 512 x 512 | Thumbnails, quick previews, emoji |
| `1K` | 1024 x 1024 | Standard quality (default) |
| `2K` | 2048 x 2048 | High quality prints, detailed work |
| `4K` | 4096 x 4096 | Maximum quality, large format |

Actual pixel dimensions vary by aspect ratio. The `image_size` controls the **long edge** resolution.

### Aspect Ratios (14 Options)

| Ratio | Orientation | Common Use |
|-------|-------------|------------|
| `1:1` | Square | Instagram, profile pictures |
| `3:2` | Landscape | Standard photo (35mm film ratio) |
| `4:3` | Landscape | Classic 4:3 display, tablets |
| `16:9` | Wide landscape | YouTube thumbnails, monitors |
| `21:9` | Ultra-wide | Cinematic, banner images |
| `2:3` | Portrait | Vertical photo |
| `3:4` | Portrait | Vertical display |
| `9:16` | Tall portrait | Instagram Stories, Reels, TikTok |
| `4:5` | Portrait | Instagram portrait post |
| `5:4` | Landscape | Slight landscape |
| `4:1` | Panoramic | Ultra-wide banner |
| `8:1` | Extreme panoramic | Website header strip |
| `1:4` | Tall vertical | Vertical banner |
| `1:8` | Extreme vertical | Vertical strip |

### Thinking Mode

Flash supports a built-in thinking/reasoning mode that improves compositional quality. The model "thinks" about layout, composition, and visual coherence before generating.

| `thinking_level` | Effect | Token Cost |
|-------------------|--------|-----------|
| `None` (default) | Standard generation — no reasoning step | No extra cost |
| `minimal` | Light reasoning — slight quality improvement | Low |
| `High` | Deep reasoning — best composition, coherent multi-object scenes | Higher |

**When to use thinking:**
- Complex scenes with multiple subjects or objects
- Precise spatial relationships ("cat on top of a bookshelf, next to a lamp")
- Photorealistic scenes requiring natural composition
- NOT needed for simple prompts ("a red apple on white background")

### Search Grounding

Flash can use Google Search (web + image) to ground generation in real-world knowledge.

```
use_search: true
```

**When to use search grounding:**
- Real landmarks: "Eiffel Tower at sunset" — gets accurate architecture
- Real products: "2024 MacBook Pro on a desk" — correct product details
- Real places: "Shibuya Crossing at night" — accurate scene
- Real animals/plants: "Golden Retriever puppy" — breed-accurate features
- NOT needed for fictional/abstract content

### Reference Images (Character Consistency)

Flash supports up to **10 reference images** for maintaining character/style consistency across multiple generations.

**Best practices:**
- Include multiple angles of the same character (front, side, 3/4)
- Use consistent lighting across references
- Higher quality references = better output consistency
- Character sheets (multi-angle on one image) work well
- Combine with a detailed prompt describing the character

**Reference image formats:** PNG, JPEG, WebP, GIF

### Prompting Tips for Flash

1. **Be specific about style:** "photorealistic 35mm film photograph" vs "watercolor painting" vs "anime illustration"
2. **Describe lighting:** "soft golden hour light", "harsh studio flash", "neon city lights"
3. **Camera details work:** "shot on Canon EOS R5, 85mm f/1.4, shallow depth of field"
4. **Negative instructions:** "no text, no watermark, no borders"
5. **Composition hints:** "rule of thirds", "centered composition", "bird's eye view"
6. **For people:** Include ethnicity, age range, clothing details, expression, pose
7. **Combine thinking + search:** For photorealistic real-world scenes, enable both for best results

### Limitations

- **Safety filters:** Gemini API applies content safety filters that may block certain prompts (bedroom + revealing clothing combinations, explicit violence). Use `safety_level: "BLOCK_ONLY_HIGH"` to relax, or adjust your prompt.
- **No `person_generation` / `prominent_people` control:** These parameters are only available in Vertex AI mode. API key mode uses model defaults.
- **No `output_mime_type` / compression quality control:** Output format is determined by the API in key mode. Images are saved as JPEG.
- **Text in images:** The model can render text but accuracy varies. Short words/titles work better than long sentences.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | — | Gemini API key from [AI Studio](https://aistudio.google.com/apikey) |
| `NANOBANANA_OUTPUT_DIR` | No | `~/nanobanana_output` | Image output directory |

## Output

### File Naming

```
{prefix}_{YYYYMMDD}_{HHMMSS}_{6char_uuid}.jpg
```

| Prefix | Source Tool |
|--------|------------|
| `gen_` | `generate_image` |
| `edit_` | `edit_image` |
| `ref_` | `generate_with_references` |

### Response Format

All tools return JSON:

```json
{
  "prompt": "A serene Japanese garden...",
  "model": "gemini-3.1-flash-image-preview",
  "settings": {
    "aspect_ratio": "16:9",
    "image_size": "4K",
    "number_of_images": 1,
    "thinking_level": "High"
  },
  "images": [
    {
      "format": "file",
      "path": "/home/user/nanobanana_output/gen_20260225_143052_a1b2c3.jpg",
      "mime_type": "image/jpeg"
    }
  ],
  "text": "Model's text response about the generated image",
  "thinking": "Model's reasoning process (when thinking_level is set)"
}
```

### Validation Errors

Invalid parameters return errors before making API calls:

```json
{
  "errors": [
    "Invalid aspect_ratio '99:1' for flash. Valid: ['1:1', '1:4', ...]",
    "temperature must be 0.0-2.0, got 5.0"
  ]
}
```

## Architecture

```
terrymcpnanobanana/
├── server.py           # MCP server (FastMCP + Google GenAI SDK)
├── requirements.txt    # Python dependencies
├── README.md           # Documentation (English)
└── README_ko.md        # Documentation (Korean)
```

### Design Decisions

- **Dual model support** — Runtime model switching via `model` parameter
- **Lazy client init** — GenAI client initialized on first tool call
- **Shared config builder** — `_build_config()` provides consistent config across all tools
- **Local validation** — `_validate_params()` checks all inputs before API calls
- **Model-specific constraints** — Aspect ratios, image sizes, reference limits per model
- **Centralized constants** — All valid values and defaults defined at module top
- **Async tool handlers** — Blocking API calls wrapped in `asyncio.to_thread()` for Windows compatibility

## Troubleshooting

### Authentication Error

```
google.api_core.exceptions.PermissionDenied
```

- Check `GEMINI_API_KEY` environment variable is set correctly
- Verify the API key is valid at [AI Studio](https://aistudio.google.com/apikey)

### Safety Filter Blocked

- Set `safety_level` to `BLOCK_ONLY_HIGH` to relax filtering
- Modify prompt to remove sensitive content
- Bedroom + revealing clothing combinations are commonly blocked

### Output Directory Permission

```
PermissionError: [Errno 13] Permission denied
```

- Verify write permissions on `NANOBANANA_OUTPUT_DIR` path
- Directory is auto-created if it doesn't exist (parent must be writable)

### Windows MCP Not Responding

If the MCP server connects but tool calls hang or fail silently on Windows:
- Ensure you're using the latest version with async tool handlers (`asyncio.to_thread()`)
- Use the full Python path in the MCP config
- Check that `GEMINI_API_KEY` is set (not Vertex AI mode)

## Changelog

### v0.2.0 (2026-03-06)
- **BREAKING**: Removed Vertex AI authentication — API key mode only
- Removed `person_generation` and `prominent_people` parameters (Vertex AI only)
- Removed `output_mime_type` and `output_compression_quality` (Vertex AI only)
- Added `__version__` tracking
- Async tool handlers for Windows compatibility (`asyncio.to_thread()`)

### v0.1.0
- Initial release with dual auth (Vertex AI + API Key)
- 5 tools: generate_image, edit_image, generate_with_references, list_generated_images, get_supported_options
- Dual model support (Flash + Pro)

---

Author: Terry.Kim <goandonh@gmail.com>
Co-Author: Claudie
