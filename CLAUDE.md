# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a transcript extraction tool that combines YouTube video downloading (yt-dlp) with speech-to-text transcription (WhisperX). The project uses PyTorch for ML operations and Click for CLI interactions.

## Development Setup

This project uses `uv` for dependency management. The main dependencies include:
- WhisperX for speech recognition
- PyTorch and torchaudio for ML operations  
- yt-dlp for video downloading
- Click for CLI interface

## Common Commands

```bash
# Install dependencies
uv sync

# Run the CLI tool
uv run transcript-extractor

# Run the main module directly
uv run python main.py
```

## Architecture

The project structure indicates a CLI-based application:
- Entry point: `transcript_extractor.cli:main` (defined in pyproject.toml)
- Current implementation: Basic placeholder in `main.py`
- Target Python versions: 3.11-3.12

Note: The `src/transcript_extrator/` directory appears to be empty and may have a typo in the name (missing 'c' in 'extractor').