# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a YouTube transcript extraction tool that combines video downloading (yt-dlp) with speech-to-text transcription (WhisperX). The project supports both CLI and MCP (Model Context Protocol) server modes, with Docker containerization and Redis/Valkey caching.

## Development Setup

This project uses `uv` for dependency management. Core dependencies:
- WhisperX for speech recognition with PyTorch/torchaudio
- yt-dlp for video downloading  
- Click for CLI interface
- FastMCP for MCP server support with HTTP transport
- Redis for caching downloaded files
- Starlette/Uvicorn for web server components

## Common Commands

### Development Setup
```bash
# Install dependencies
uv sync

# Run tests (Docker-based integration tests with real YouTube videos)
./run_tests.sh

# Run individual test components in container
docker compose exec mcp-server uv run transcript-extractor "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --model tiny --format text
```

### CLI Usage (Local)
```bash
# Basic transcription
uv run transcript-extractor "https://youtube.com/watch?v=VIDEO_ID"

# Advanced options
uv run transcript-extractor "URL" --format srt --model large-v3 --verbose

# Use Breeze ASR 25 for Mandarin-English code-switching
uv run transcript-extractor "URL" --format srt --model breeze-asr-25

# Speaker diarization (requires HF_TOKEN environment variable)
uv run transcript-extractor "URL" --diarize --format srt --num-speakers 2
```

### Docker Development Workflow
```bash
# Build and start containerized environment (reuses models for efficiency)
docker compose build && docker compose up -d mcp-server

# Execute CLI in container (recommended for consistent environment)
docker compose exec mcp-server uv run transcript-extractor "URL" --format srt --model large-v3

# Check MCP server status
docker compose ps
docker compose logs mcp-server

# Stop services
docker compose down
```

### MCP Server Mode
```bash
# Start HTTP MCP server (requires AUTH_SERVER_URL configuration)
uv run transcript-extractor-mcp

# Or via Docker
docker compose up -d mcp-server
```

## Architecture

### Dual-Mode Architecture Overview

This project implements a **shared-core, dual-entry-point architecture** where both CLI and MCP server modes use identical business logic but different interfaces:

**Entry Points**:
- **CLI Mode** (`cli.py`): Direct command-line interface for immediate transcription
- **MCP Server Mode** (`mcp_server.py`): HTTP server with OAuth2 authentication for programmatic access

**Shared Core**: Both modes use the same `TranscriptionService` instance and components, ensuring consistent behavior and shared caching benefits.

### Core Components

**Service Layer** (`src/transcript_extractor/core/service.py`):
- `TranscriptionService`: Main orchestrator handling end-to-end transcription workflow
- `TranscriptionConfig`: Configuration dataclass for all transcription parameters  
- `TranscriptionResult`: Result container with all output formats and metadata
- **Key Pattern**: Single service instance can be reused across multiple requests (important for MCP server efficiency)

**Downloader** (`src/transcript_extractor/core/downloader.py`):
- `YouTubeDownloader`: Uses yt-dlp for audio extraction and YouTube transcript fetching
- Downloads audio in specified formats (wav, mp3, flac) with intelligent caching
- Extracts native YouTube transcripts/captions when available (faster than AI transcription)
- Handles file cleanup and temporary directory management

**Transcriber** (`src/transcript_extractor/core/transcriber.py`):
- `WhisperTranscriber`: Supports multiple transcription models with automatic device fallback
- **WhisperX models**: tiny, base, small, medium, large-v2, large-v3 (general multilingual)
- **Breeze ASR 25**: Specialized for Mandarin-English code-switching scenarios
- **Device Management**: Automatic CUDA→CPU fallback when GPU unavailable
- Generates text, SRT, and VTT output formats with precise timing alignment

**Cache Service** (`src/transcript_extractor/core/cache.py`):
- `CacheService`: Redis/Valkey-based caching for downloaded audio files
- Maps URLs to cached file paths with metadata for efficient reuse
- **Graceful Degradation**: Application works without Redis connection (cache disabled)
- Handles cache expiration and cleanup of missing files automatically

### Entry Points

**CLI Mode** (`src/transcript_extractor/cli.py`):
- Single command interface with comprehensive options
- Supports all model names, languages, and output formats
- Progress reporting with verbose mode
- **Usage Pattern**: Creates new `TranscriptionService` instance per execution

**MCP Server Mode** (`src/transcript_extractor/mcp_server.py`):
- HTTP-based MCP server with OAuth2 authentication and scope validation
- **Hardware-Aware Model Management**: Automatic model downgrading based on `MAX_WHISPER_MODEL` environment variable
- **Usage Pattern**: Single persistent service instance for efficiency across multiple requests
- Streaming support with detailed progress and error handling

### MCP Tools Integration

**Available Tools**:
1. **`extract_youtube_transcript`**: Full transcription with multiple format support
   - Model validation and potential downgrading
   - Progress streaming and comprehensive error handling
   - Returns all formats (text, SRT, VTT) plus metadata

2. **`get_youtube_transcripts`**: Fast retrieval of existing YouTube captions
   - No AI processing required - direct YouTube API access
   - Multiple language support detection
   - Useful for quick caption availability checking

3. **`list_whisper_models`**: Server capability discovery
   - Shows available models based on hardware limits
   - Model characteristics (speed vs accuracy tradeoffs)
   - Critical for client-side model selection logic

**Authentication Flow**:
- OAuth2 JWT token validation with configurable scopes
- Per-tool scope requirements (can be customized)
- Graceful fallback when authentication is disabled (CLI mode compatibility)

### Configuration

**Environment Variables**:
- `DOWNLOAD_DIR`: Directory for downloaded audio files (default: ./downloads)
- `MODEL_STORE_DIR`: Directory for caching models (WhisperX, Breeze ASR 25, alignment models) (default: ./models)
- `VALKEY_HOST/PORT/DB`: Redis/Valkey connection settings
- `MCP_TRANSPORT/HOST/PORT`: MCP server configuration
- `AUTH_SERVER_URL`: OAuth2 authentication server
- `MAX_WHISPER_MODEL`: Server hardware limits for model selection (Note: Breeze ASR 25 bypasses this limit)
- `DEBUG_ENABLED/PORT`: Remote debugging configuration
- `HF_TOKEN`: HuggingFace authentication token (required for speaker diarization, not supported with Breeze ASR 25)

**Docker Setup**:
- Multi-service compose with transcript-extractor and mcp-server
- GPU support with NVIDIA runtime
- Health checks and networking configuration

## Development Workflow and Debugging

### Recommended Development Environment

**Containerized Development** (Recommended):
- Use `docker compose exec mcp-server` for consistent model loading and GPU access
- Models persist between container restarts for faster iteration
- Eliminates environment-specific dependencies (ffmpeg, CUDA, etc.)

**Local Development**:
- Requires manual installation of ffmpeg and potential CUDA setup
- Use `tmux` for long-running transcription tasks (models can take time to load)
- Set `DOWNLOAD_DIR` and `MODEL_STORE_DIR` environment variables for persistent caching

### Debugging Common Issues

**Model Loading Problems**:
- **Symptom**: `libctranslate2` errors or CUDA initialization failures
- **Solution**: Code includes automatic CPU fallback - check logs for device switching
- **Debug**: Set `--device cpu` explicitly to bypass GPU issues

**Cache Issues**:
- **Redis Connection Failed**: Application continues without caching (degraded performance)
- **Debug Commands**: 
  ```bash
  docker compose logs mcp-server  # Check Redis connection status
  redis-cli ping  # Test Redis connectivity
  ```

**MCP Server Authentication**:
- **401 Errors**: Check `AUTH_SERVER_URL` configuration and JWT token validity  
- **Scope Errors**: Verify required scopes match OAuth2 provider configuration
- **Debug**: Authentication can be disabled for development (server runs without OAuth2)

**Model Downgrading**:
- **Server automatically downgrades models** based on `MAX_WHISPER_MODEL` environment variable
- **Client Choice vs Server Reality**: Check server response for actual model used
- **Breeze ASR 25 Exception**: Bypasses hardware limits (specialized model)

### Integration Testing Strategy

**Real-World Testing**: 
- `./run_tests.sh` uses actual YouTube videos for comprehensive validation
- Tests cover: download→transcription→formatting pipeline
- **Network Dependency**: Tests require internet access and working YouTube API

**Component Testing**:
```bash
# Test individual components in isolation
docker compose exec mcp-server uv run python -c "
from transcript_extractor.core.downloader import YouTubeDownloader
d = YouTubeDownloader()
print(d.get_youtube_transcripts('https://www.youtube.com/watch?v=dQw4w9WgXcQ'))
"
```

### Performance Optimization

**Model Caching**: Models download once to `MODEL_STORE_DIR` and persist across runs
**Audio Caching**: Redis caches downloaded audio files for faster re-transcription
**Container Reuse**: Keep `mcp-server` container running to avoid model reload overhead

## Development Notes

- Use `tmux` for long-running transcription tasks
- GPU acceleration requires NVIDIA Docker runtime  
- Model loading can fail with libctranslate2 issues - code includes CPU fallback
- Integration tests use real YouTube videos and require network access
- MCP server supports model downgrading based on hardware constraints
- Cache service is optional - application works without Redis connection