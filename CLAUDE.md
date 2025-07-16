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

```bash
# Install dependencies
uv sync

# Run CLI tool
uv run transcript-extractor "https://youtube.com/watch?v=VIDEO_ID"

# Run with specific options
uv run transcript-extractor "URL" --format srt --model large-v3

# Run MCP server
uv run transcript-extractor-mcp

# Run tests
uv run python -m pytest
./run_tests.sh  # Docker-based integration tests

# Docker usage
docker compose build
docker compose run --rm transcript-extractor "URL" --format srt
docker compose up -d mcp-server
```

## Architecture

### Core Components

**Service Layer** (`src/transcript_extractor/core/service.py`):
- `TranscriptionService`: Main orchestrator handling end-to-end transcription
- `TranscriptionConfig`: Configuration dataclass for all transcription parameters
- `TranscriptionResult`: Result container with all output formats and metadata
- Handles caching, progress callbacks, and error management

**Downloader** (`src/transcript_extractor/core/downloader.py`):
- `YouTubeDownloader`: Uses yt-dlp for audio extraction and YouTube transcript fetching
- Downloads audio in specified formats (wav, mp3, flac)
- Extracts native YouTube transcripts/captions when available
- Handles file cleanup and temporary directory management

**Transcriber** (`src/transcript_extractor/core/transcriber.py`):
- `WhisperTranscriber`: WhisperX wrapper with model loading and format conversion
- Supports all Whisper model sizes (tiny to large-v3)
- Handles device selection (CPU/CUDA) and compute precision
- Generates text, SRT, and VTT output formats with proper timing

**Cache Service** (`src/transcript_extractor/core/cache.py`):
- `CacheService`: Redis/Valkey-based caching for downloaded audio files
- Maps URLs to cached file paths with metadata
- Handles cache expiration and cleanup of missing files

### Entry Points

**CLI Mode** (`src/transcript_extractor/cli.py`):
- Single command interface with comprehensive options
- Supports all model sizes, languages, and output formats
- Progress reporting with verbose mode

**MCP Server Mode** (`src/transcript_extractor/mcp_server.py`):
- HTTP-based MCP server with OAuth2 authentication
- Server hardware limits with model downgrading
- Tools: `extract_youtube_transcript`, `get_youtube_transcripts`, `list_whisper_models`
- Streaming support with detailed progress and error handling

### Configuration

**Environment Variables**:
- `DOWNLOAD_DIR`: Directory for downloaded audio files (default: ./downloads)
- `MODEL_STORE_DIR`: Directory for caching Whisper and alignment models (default: ./models)
- `VALKEY_HOST/PORT/DB`: Redis/Valkey connection settings
- `MCP_TRANSPORT/HOST/PORT`: MCP server configuration
- `AUTH_SERVER_URL`: OAuth2 authentication server
- `MAX_WHISPER_MODEL`: Server hardware limits for model selection
- `DEBUG_ENABLED/PORT`: Remote debugging configuration

**Docker Setup**:
- Multi-service compose with transcript-extractor and mcp-server
- GPU support with NVIDIA runtime
- Health checks and networking configuration

## Development Notes

- Use `tmux` for long-running transcription tasks
- GPU acceleration requires NVIDIA Docker runtime
- Model loading can fail with libctranslate2 issues - code includes CPU fallback
- Integration tests use real YouTube videos and require network access
- MCP server supports model downgrading based on hardware constraints
- Cache service is optional - application works without Redis connection