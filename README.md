# Nano Banana Pro MCP Server

Vertex AI 기반 **Gemini 3 Pro Image (Nano Banana Pro)** 이미지 생성/편집 MCP 서버.

> Model ID: `gemini-3-pro-image-preview`

## Overview

Google의 최신 이미지 생성 모델인 Nano Banana Pro를 MCP(Model Context Protocol) 서버로 제공합니다.
Claude Desktop, Claude Code 등 MCP 클라이언트에서 자연어로 이미지를 생성하고 편집할 수 있습니다.

### Key Features

- **Text-to-Image**: 텍스트 프롬프트로 고품질 이미지 생성 (최대 4K)
- **Image Editing**: 자연어 명령으로 기존 이미지 편집
- **Reference-based Generation**: 최대 14장 참조 이미지로 캐릭터/스타일 일관성 유지
- **Full Parameter Control**: 해상도, 비율, 인물 생성, temperature, seed 등 전체 파라미터 지원
- **Dual Auth**: Vertex AI (ADC) 또는 Gemini API Key 두 가지 인증 방식

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
    "nanobanana-pro": {
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
    "nanobanana-pro": {
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
    "nanobanana-pro": {
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
    "nanobanana-pro": {
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
aspect_ratio: "16:9"
image_size: "4K"
number_of_images: 2
```

### 2. `edit_image` - Image Editing

기존 이미지를 자연어 명령으로 편집합니다.

```
image_path: "/home/user/photo.jpg"
instruction: "Make it look like a watercolor painting"
image_size: "2K"
```

### 3. `generate_with_references` - Reference-based Generation

참조 이미지를 기반으로 일관된 스타일/캐릭터의 새 이미지를 생성합니다.
최대 14장의 참조 이미지를 지원합니다.

```
prompt: "Same character sitting in a cafe, drinking coffee"
reference_paths: ["/home/user/character_ref1.png", "/home/user/character_ref2.png"]
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

### Image Configuration

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `aspect_ratio` | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`, `16:9`, `21:9` | `1:1` | 출력 이미지 비율 |
| `image_size` | `1K`, `2K`, `4K` | `1K` | 출력 해상도. 4K는 최고 품질 |
| `number_of_images` | 1 ~ 4 | 1 | 요청당 생성 이미지 수 |
| `output_format` | `file`, `base64` | `file` | 파일 저장 또는 base64 데이터 반환 |

### Generation Control

| Parameter | Range/Options | Default | Description |
|-----------|---------------|---------|-------------|
| `temperature` | 0.0 ~ 2.0 | model default | 랜덤성 조절. 높을수록 창의적. Google 권장: 1.0 |
| `seed` | integer | random | 고정 시드로 재현 가능한 결과 생성 |

### Safety & Content

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `person_generation` | `DONT_ALLOW`, `ALLOW_ADULT`, `ALLOW_ALL` | model default | 인물/얼굴 생성 제어 |
| `safety_level` | `BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_ONLY_HIGH`, `BLOCK_NONE` | model default | 4개 유해 카테고리 필터 임계값 |

### Safety Level Details

| Level | Description |
|-------|-------------|
| `BLOCK_LOW_AND_ABOVE` | 가장 엄격 - 낮은 위험도부터 차단 |
| `BLOCK_MEDIUM_AND_ABOVE` | 중간 위험도부터 차단 |
| `BLOCK_ONLY_HIGH` | 높은 위험도만 차단 |
| `BLOCK_NONE` | 필터 없음 (주의 필요) |

### Person Generation Details

| Option | Description |
|--------|-------------|
| `DONT_ALLOW` | 인물 생성 불가 |
| `ALLOW_ADULT` | 성인 인물만 생성 허용 |
| `ALLOW_ALL` | 모든 인물 생성 허용 |

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
{prefix}_{YYYYMMDD}_{HHMMSS}_{6char_uuid}.png
```

| Prefix | Source |
|--------|--------|
| `gen_` | `generate_image` |
| `edit_` | `edit_image` |
| `ref_` | `generate_with_references` |

Example: `gen_20260225_143052_a1b2c3.png`

### Response Format

모든 tool은 JSON 형식으로 응답합니다:

```json
{
  "prompt": "A serene Japanese garden...",
  "model": "gemini-3-pro-image-preview",
  "settings": {
    "aspect_ratio": "16:9",
    "image_size": "4K",
    "number_of_images": 1
  },
  "images": [
    {
      "format": "file",
      "path": "/home/user/nanobanana_output/gen_20260225_143052_a1b2c3.png",
      "mime_type": "image/png"
    }
  ],
  "text": "Model's text response about the generated image"
}
```

## Architecture

```
terrymcpnanobanana/
├── server.py           # MCP server (FastMCP + Google GenAI SDK)
├── requirements.txt    # Python dependencies
└── README.md           # This documentation
```

### Internal Structure (server.py)

```
Configuration          Constants, env vars, valid option sets
Client                 _get_client() - lazy GenAI client init
Helpers                _save_image(), _build_config(), _extract_results()
MCP Tools              5 tools exposed via FastMCP
Entry Point            stdio transport for MCP communication
```

### Key Design Decisions

- **Lazy client init**: GenAI 클라이언트는 첫 tool 호출 시 초기화
- **Shared config builder**: `_build_config()` 헬퍼로 모든 tool이 동일한 설정 로직 공유
- **Dual auth support**: 환경변수 하나로 Vertex AI / API Key 전환
- **SynthID watermarking**: Vertex AI 사용 시 자동으로 SynthID 워터마크 적용

## Troubleshooting

### Authentication Error

```
google.auth.exceptions.DefaultCredentialsError
```

- Vertex AI: `gcloud auth application-default login` 실행
- API Key: `GEMINI_API_KEY` 환경변수 확인

### Model Not Found

```
Model gemini-3-pro-image-preview not found
```

- Vertex AI 프로젝트에서 Gemini API가 활성화되었는지 확인
- `GOOGLE_CLOUD_LOCATION`을 `global` 또는 `us-central1`로 설정

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
