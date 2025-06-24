#!/bin/bash

# Check MCP_TRANSPORT environment variable to determine startup mode
if [ "$MCP_TRANSPORT" = "http" ] || [ "$MCP_TRANSPORT" = "stdio" ]; then
    # Start MCP server mode
    echo "Starting MCP server with transport: $MCP_TRANSPORT"
    exec uv run transcript-extractor-mcp
else
    # Start CLI mode (default)
    echo "Starting CLI mode"
    exec uv run transcript-extractor "$@"
fi