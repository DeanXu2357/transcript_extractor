#!/bin/bash

# Test script that builds and runs tests in Docker container

set -e

echo ""
echo "🚀 Starting MCP server container..."
docker compose up -d mcp-server --build --force-recreate

echo ""
echo "🧪 Running integration tests..."
echo "This will test:"
echo "  1. YouTube transcript download"
echo "  2. Audio download from YouTube"
echo "  3. Whisper transcription"
echo "  4. Full service integration"
echo ""

# Run integration tests in the running mcp-server container
docker compose exec mcp-server sh -c "cd /app && uv run python test_integration.py"

echo ""
echo "✅ Test completed!"
echo ""
echo "To run individual components:"
echo "  # Test CLI directly"
echo "  docker compose exec mcp-server uv run transcript-extractor \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\" --format text"
echo ""
echo "  # Test with SRT output"
echo "  docker compose exec mcp-server uv run transcript-extractor \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\" --format srt"
echo ""
echo "  # Quick test with tiny model"
echo "  docker compose exec mcp-server uv run transcript-extractor \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\" --model tiny --format text"
echo ""
