# Transcript Extractor

A YouTube video transcription tool that integrates video downloading with speech-to-text conversion. This is a personal side project created to fulfill specific needs by combining multiple open-source projects.

## Project Architecture

This project combines the following technologies:

- **yt-dlp**: YouTube video downloading
- **WhisperX**: High-accuracy speech-to-text transcription
- **MCP (Model Context Protocol)**: Server mode support

## Features

- Support for multiple output formats (text, srt, vtt)
- Multiple Whisper model options (tiny to large-v3)
- Containerized deployment
- MCP server mode (streamable HTTP protocol)

## Usage

### 1. Command Line Mode

#### Local Execution
```bash
# Install dependencies
uv sync

# Basic usage
uv run transcript-extractor "https://youtube.com/watch?v=VIDEO_ID"

# Specify output format and model
uv run transcript-extractor "https://youtube.com/watch?v=VIDEO_ID" \
  --format srt \
  --model large-v3 \
  --language zh \
  --output output.srt
```

#### Docker Container Execution
```bash
# Build container
docker compose build

# Pure transcript output (recommended)
docker compose run --rm --quiet transcript-extractor "https://youtube.com/watch?v=VIDEO_ID" --model large-v3

# Output to file
docker compose run --rm transcript-extractor "https://youtube.com/watch?v=VIDEO_ID" \
  --format srt \
  --model large-v3 \
  --output /app/output/transcript.srt
```

#### Parameters
- `--format, -f`: Output format (text, srt, vtt)
- `--model, -m`: Model size (tiny, base, small, medium, large-v2, large-v3)
- `--language, -l`: Language code (zh, en, etc.)
- `--output, -o`: Output file path
- `--verbose, -v`: Show processing progress

### 2. MCP Server Mode

MCP server uses streamable HTTP protocol and requires its own authorization provider.

```bash
# Start MCP server
docker compose up -d mcp-server

# Check server status
docker compose ps
docker compose logs mcp-server

# Stop server
docker compose down mcp-server
```

#### Environment Variables
- `MCP_TRANSPORT=http`: Use HTTP transport
- `MCP_HOST`: Server host (default: 0.0.0.0)
- `MCP_PORT`: Server port (default: 8080)
- `AUTH_SERVER_URL`: Authorization server URL
- `MCP_JWT_AUDIENCE`: JWT audience (default: transcript-extractor)

## Output File Location

When using Docker, output files are stored at:
- Container path: `/app/output/`
- Local path: `./output/`

## System Requirements

- Python 3.11-3.12
- Docker & Docker Compose
- NVIDIA GPU (optional, for acceleration)
- FFmpeg (included in container)

## Development

This project uses `uv` for dependency management:

```bash
# Install development dependencies
uv sync

# Run tests
uv run python -m pytest

# Local development execution
uv run python main.py
```

---

*This is a personal side project that integrates multiple excellent open-source tools to meet daily video transcription needs.*