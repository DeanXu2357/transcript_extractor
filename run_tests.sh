#!/bin/bash

# Test script that builds and runs tests in Docker container

set -e

echo "🔧 Building Docker container..."
docker compose build

echo ""
echo "🧪 Running integration tests..."
echo "This will test:"
echo "  1. YouTube transcript download"
echo "  2. Audio download from YouTube"
echo "  3. Whisper transcription"
echo "  4. Full service integration"
echo ""

# Run integration tests by overriding entrypoint
docker compose run --rm --entrypoint="" transcript-extractor sh -c "cd /app && uv run python test_integration.py"

echo ""
echo "✅ Test completed!"
echo ""
echo "To run individual components:"
echo "  # Test CLI directly"
echo "  docker-compose run --rm transcript-extractor \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\" --format text"
echo ""
echo "  # Test with SRT output"
echo "  docker-compose run --rm transcript-extractor \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\" --format srt --output /app/output/test.srt"
echo ""
echo "  # Quick test with tiny model"
echo "  docker-compose run --rm transcript-extractor \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\" --model tiny --format text"
