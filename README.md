# Transcript Extractor

A YouTube video transcription tool that integrates video downloading with speech-to-text conversion. This is a personal side project created to fulfill specific needs by combining multiple open-source projects.

## Project Architecture

This project combines the following technologies:

- **yt-dlp**: YouTube video downloading
- **WhisperX**: High-accuracy speech-to-text transcription
- **MCP (Model Context Protocol)**: Server mode support

## Features

- Support for multiple output formats (text, srt, vtt)
- Multiple model options (tiny, base, small, medium, large-v2, large-v3, breeze-asr-25)
- Speaker diarization support with configurable speaker count
- Redis/Valkey caching for downloaded audio files
- Containerized deployment with GPU support
- MCP server mode with HTTP transport and OAuth2 authentication

## Usage

### 1. Command Line Mode

#### Docker Container Execution

```bash
# Build and start MCP server container
docker compose build
docker compose up -d mcp-server

# Execute CLI commands in running container (reuses models for efficiency)
docker compose exec mcp-server uv run transcript-extractor "https://youtube.com/watch?v=VIDEO_ID" --model large-v3

# Use Breeze ASR 25 for Mandarin-English code-switching
docker compose exec mcp-server uv run transcript-extractor "https://youtube.com/watch?v=VIDEO_ID" --model breeze-asr-25

# Output to file
docker compose exec mcp-server uv run transcript-extractor "https://youtube.com/watch?v=VIDEO_ID" \
  --format srt \
  --model large-v3 \
  --output /app/output/transcript.srt
```

#### Parameters
- `--format, -f`: Output format (text, srt, vtt)
- `--model, -m`: Model name (tiny, base, small, medium, large-v2, large-v3, breeze-asr-25)
- `--language, -l`: Language code (zh, en, etc.)
- `--device`: Device to run transcription on (cpu, cuda)
- `--compute-type`: Compute precision (float16, float32, int8)
- `--diarize`: Enable speaker diarization (requires HF_TOKEN)
- `--num-speakers`: Number of speakers (if known)
- `--min-speakers`: Minimum number of speakers
- `--max-speakers`: Maximum number of speakers
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
**MCP Server Configuration:**
- `MCP_TRANSPORT=http`: Use HTTP transport
- `MCP_HOST`: Server host (default: 0.0.0.0)
- `MCP_PORT`: Server port (default: 8080)
- `AUTH_SERVER_URL`: OAuth2 authentication server URL
- `MCP_JWT_AUDIENCE`: JWT audience (default: transcript-extractor)
- `MAX_WHISPER_MODEL`: Maximum allowed model (server hardware limits)

**Storage and Caching:**
- `DOWNLOAD_DIR`: Directory for downloaded audio files (default: ./downloads)
- `MODEL_STORE_DIR`: Directory for caching Whisper models (default: ./models)
- `VALKEY_HOST/PORT/DB`: Redis/Valkey connection settings
- `VALKEY_PASSWORD/USERNAME`: Redis/Valkey authentication

**Additional Features:**
- `HF_TOKEN`: HuggingFace token (required for speaker diarization)
- `DEBUG_ENABLED/PORT`: Remote debugging configuration

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

# Local CLI execution
uv run transcript-extractor "https://youtube.com/watch?v=VIDEO_ID"

# Local MCP server
uv run transcript-extractor-mcp
```

---

*This is a personal side project that integrates multiple excellent open-source tools to meet daily video transcription needs.*