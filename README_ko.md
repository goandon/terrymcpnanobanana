# Nano Banana MCP Server

[English Documentation](README.md)

Vertex AI 기반 **Nano Banana** 이미지 생성/편집 MCP 서버.

> **Nano Banana 2** (Gemini 3.1 Flash Image): `gemini-3.1-flash-image-preview` — fast, cost-effective (default)
> **Nano Banana Pro** (Gemini 3 Pro Image): `gemini-3-pro-image-preview` — highest quality

## Overview

Google의 이미지 생성 모델 Nano Banana 시리즈를 MCP(Model Context Protocol) 서버로 제공합니다.
Claude Desktop, Claude Code 등 MCP 클라이언트에서 자연어로 이미지를 생성하고 편집할 수 있습니다.

### Key Features

- **Dual Model**: Flash (빠름, 저렴) / Pro (최고 품질) 선택 가능. 기본값: Flash
- **Text-to-Image**: 텍스트 프롬프트로 고품질 이미지 생성 (최대 4K)
- **Image Editing**: 자연어 명령으로 기존 이미지 편집
- **Reference-based Generation**: 참조 이미지로 캐릭터/스타일 일관성 유지 (Flash: 10장, Pro: 14장)
- **Thinking Mode**: (Flash only) 고품질 구도를 위한 사고 모드
- **Search Grounding**: (Flash only) Google Search 기반 실제 인물/장소 정확 렌더링
- **Full Parameter Control**: 해상도, 비율, 인물 생성, temperature, seed 등 전체 파라미터 지원
- **Input Validation**: API 호출 전 모든 파라미터 로컬 검증. 명확한 에러 메시지 제공
- **Dual Auth**: Vertex AI (ADC) 또는 Gemini API Key 두 가지 인증 방식

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
| `google-genai` | >= 1.0.0 | Google Gen AI SDK (Vertex AI / Gemini API) |
| `Pillow` | >= 10.0.0 | Image processing |

## Authentication

두 가지 인증 방식을 지원합니다.

### Option 1: Vertex AI (Production, recommended)

Google Cloud 프로젝트 + Application Default Credentials 사용:

```bash
# GCP 인증 설정
gcloud auth application-default login

# 환경변수 설정
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=true
```

### Option 2: Gemini API Key (Development)

Google AI Studio에서 발급한 API Key 사용:

```bash
export GOOGLE_GENAI_USE_VERTEXAI=false
export GEMINI_API_KEY=your-api-key-here
```

## MCP Client Configuration

### Claude Desktop (`claude_desktop_config.json`)

**Vertex AI mode:**
```json
{
  "mcpServers": {
    "nanobanana": {
      "command": "python",
      "args": ["/path/to/terrymcpnanobanana/server.py"],
      "env": {
        "GOOGLE_CLOUD_PROJECT": "your-project-id",
        "GOOGLE_CLOUD_LOCATION": "global",
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        "NANOBANANA_OUTPUT_DIR": "/path/to/output/folder"
      }
    }
  }
}
```

**API Key mode:**
```json
{
  "mcpServers": {
    "nanobanana": {
      "command": "python",
      "args": ["/path/to/terrymcpnanobanana/server.py"],
      "env": {
        "GOOGLE_GENAI_USE_VERTEXAI": "false",
        "GEMINI_API_KEY": "your-api-key-here",
        "NANOBANANA_OUTPUT_DIR": "/path/to/output/folder"
      }
    }
  }
}
```

### Claude Code (`.claude/settings.json`)

```json
{
  "mcpServers": {
    "nanobanana": {
      "command": "python",
      "args": ["/path/to/terrymcpnanobanana/server.py"],
      "env": {
        "GOOGLE_CLOUD_PROJECT": "your-project-id",
        "GOOGLE_GENAI_USE_VERTEXAI": "true"
      }
    }
  }
}
```

### Windows (Python full path)

```json
{
  "mcpServers": {
    "nanobanana": {
      "command": "C:\\Users\\terry\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
      "args": ["C:\\path\\to\\terrymcpnanobanana\\server.py"],
      "env": {
        "GOOGLE_CLOUD_PROJECT": "your-project-id",
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        "NANOBANANA_OUTPUT_DIR": "D:\\Images\\nanobanana"
      }
    }
  }
}
```

## Tools Reference

### 1. `generate_image` - Text to Image

텍스트 프롬프트로 이미지를 생성합니다.

```
prompt: "A serene Japanese garden with cherry blossoms at sunset"
model: "flash"          # or "pro"
aspect_ratio: "16:9"
image_size: "4K"
number_of_images: 2
thinking_level: "High"  # Flash only: better composition
use_search: true        # Flash only: accurate real subjects
```

### 2. `edit_image` - Image Editing

기존 이미지를 자연어 명령으로 편집합니다.

```
image_path: "/home/user/photo.jpg"
instruction: "Make it look like a watercolor painting"
model: "flash"
image_size: "2K"
use_search: true        # Flash only: reference real subjects
```

### 3. `generate_with_references` - Reference-based Generation

참조 이미지를 기반으로 일관된 스타일/캐릭터의 새 이미지를 생성합니다.

```
prompt: "Same character sitting in a cafe, drinking coffee"
reference_paths: ["/home/user/character_ref1.png", "/home/user/character_ref2.png"]
model: "flash"           # max 10 refs (pro: max 14)
aspect_ratio: "3:4"
image_size: "2K"
```

### 4. `list_generated_images` - List Output

생성된 이미지 목록을 최신순으로 조회합니다.

```
limit: 10
```

### 5. `get_supported_options` - Parameter Reference

사용 가능한 전체 파라미터와 유효 값을 반환합니다. 파라미터 없이 호출합니다.

## Parameters

### Model Selection

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `model` | `flash`, `pro` | `flash` | Flash: 빠름/저렴, Pro: 최고 품질 |

### Image Configuration

| Parameter | Flash | Pro | Default | Description |
|-----------|-------|-----|---------|-------------|
| `aspect_ratio` | `1:1`, `1:4`, `1:8`, `2:3`, `3:2`, `3:4`, `4:1`, `4:3`, `4:5`, `5:4`, `8:1`, `9:16`, `16:9`, `21:9` | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`, `16:9`, `21:9` | `1:1` | 출력 이미지 비율 |
| `image_size` | `512px`, `1K`, `2K`, `4K` | `1K`, `2K`, `4K` | `1K` | 출력 해상도 |
| `number_of_images` | 1 ~ 4 | 1 ~ 4 | 1 | 요청당 생성 이미지 수 |
| `output_format` | `file`, `base64` | `file`, `base64` | `file` | 파일 저장 또는 base64 반환 |

### Generation Control

| Parameter | Range/Options | Default | Description |
|-----------|---------------|---------|-------------|
| `temperature` | 0.0 ~ 2.0 | model default | 랜덤성 조절. 높을수록 창의적. Google 권장: 1.0 |
| `seed` | integer | random | 고정 시드로 재현 가능한 결과 생성 |

### Flash-only Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `thinking_level` | `minimal`, `High` | None (off) | 사고 모드. 더 나은 구도/품질. 토큰 비용 발생 |
| `use_search` | `true`, `false` | `false` | Google Search 그라운딩. 실제 인물/장소 정확 렌더링 |

### Safety & Content

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `person_generation` | `DONT_ALLOW`/`ALLOW_NONE`, `ALLOW_ADULT`, `ALLOW_ALL` | model default | 인물/얼굴 생성 제어 |
| `prominent_people` | `ALLOW`, `DENY` | model default | 유명인/저명인물 생성 제어 |
| `safety_level` | `BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_ONLY_HIGH`, `BLOCK_NONE` | model default | 4개 유해 카테고리 필터 임계값 |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT` | Vertex AI mode | `""` | GCP 프로젝트 ID |
| `GOOGLE_CLOUD_LOCATION` | No | `global` | GCP 리전 |
| `GOOGLE_GENAI_USE_VERTEXAI` | No | `true` | `true`: Vertex AI, `false`: API Key |
| `GEMINI_API_KEY` | API Key mode | - | Gemini API 키 |
| `NANOBANANA_OUTPUT_DIR` | No | `~/nanobanana_output` | 이미지 출력 디렉토리 |

## Output

### File Naming Convention

생성된 이미지는 다음 형식으로 저장됩니다:

```
{prefix}_{YYYYMMDD}_{HHMMSS}_{6char_uuid}.jpg
```

| Prefix | Source |
|--------|--------|
| `gen_` | `generate_image` |
| `edit_` | `edit_image` |
| `ref_` | `generate_with_references` |

Example: `gen_20260225_143052_a1b2c3.jpg`

### Response Format

모든 tool은 JSON 형식으로 응답합니다:

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

잘못된 파라미터는 API 호출 전에 에러를 반환합니다:

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

### Internal Structure (server.py)

```
Configuration          Model registry, env vars, model-specific valid option sets
Client                 _get_client() - lazy GenAI client init
Helpers                _resolve_model(), _save_image(), _build_config(), _extract_results()
MCP Tools              5 tools exposed via FastMCP
Entry Point            stdio transport for MCP communication
```

### Key Design Decisions

- **Dual model support**: `model` 파라미터로 Flash/Pro 런타임 전환
- **Lazy client init**: GenAI 클라이언트는 첫 tool 호출 시 초기화
- **Shared config builder**: `_build_config()` 헬퍼로 모든 tool이 동일한 설정 로직 공유
- **Local validation**: `_validate_params()` 로 API 호출 전 모든 입력값 사전 검증
- **Model-specific constraints**: aspect ratio, image size, reference limit 등 모델별 분리
- **Dual auth support**: 환경변수 하나로 Vertex AI / API Key 전환
- **SynthID watermarking**: 자동으로 SynthID 워터마크 적용

## Troubleshooting

### Authentication Error

```
google.auth.exceptions.DefaultCredentialsError
```

- Vertex AI: `gcloud auth application-default login` 실행
- API Key: `GEMINI_API_KEY` 환경변수 확인

### Model Not Found

```
Model gemini-3.1-flash-image-preview not found
```

- Vertex AI 프로젝트에서 Gemini API가 활성화되었는지 확인
- `GOOGLE_CLOUD_LOCATION`을 `global` 또는 `us-central1`로 설정
- Flash 모델이 아직 preview인 경우 `model="pro"`로 폴백

### Safety Filter Blocked

이미지 생성이 안전 필터에 의해 차단된 경우:
- `safety_level`을 `BLOCK_ONLY_HIGH`로 완화
- 프롬프트를 수정하여 민감한 내용 제거

### Output Directory Permission

```
PermissionError: [Errno 13] Permission denied
```

- `NANOBANANA_OUTPUT_DIR` 경로에 쓰기 권한이 있는지 확인
- 디렉토리가 없으면 자동 생성됨 (부모 디렉토리에 권한 필요)

---

Author: Terry.Kim <goandonh@gmail.com>
Co-Author: Claudie
