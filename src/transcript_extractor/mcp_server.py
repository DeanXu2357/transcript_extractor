#!/usr/bin/env python3
"""MCP Server for YouTube Transcript Extraction with mcp-auth integration."""

import asyncio
import logging
import os
from typing import Any, Optional

import debugpy

from mcp.server.fastmcp import FastMCP
from mcpauth import MCPAuth
from mcpauth.config import AuthServerType
from mcpauth.utils import fetch_server_config
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Mount
import uvicorn
import traceback
import time

from .core.service import TranscriptionConfig, TranscriptionService
from .core.constants import ALL_MODELS, BREEZE_MODEL

# Environment variable constants
ENV_MAX_WHISPER_MODEL = "MAX_WHISPER_MODEL"
ENV_AUTH_SERVER_URL = "AUTH_SERVER_URL"
ENV_MCP_TRANSPORT = "MCP_TRANSPORT"
ENV_MCP_HOST = "MCP_HOST"
ENV_MCP_PORT = "MCP_PORT"
ENV_MCP_JWT_AUDIENCE = "MCP_JWT_AUDIENCE"
ENV_MCP_DEVICE = "MCP_DEVICE"
ENV_MCP_COMPUTE_TYPE = "MCP_COMPUTE_TYPE"

# Transport constants
TRANSPORT_STDIO = "stdio"
TRANSPORT_HTTP = "http"

# OAuth2 scope constants
SCOPE_OPENID = "openid"
SCOPE_OFFLINE = "offline"
SCOPE_TRANSCRIPT_READ = "transcript:read"
SCOPE_TRANSCRIPT_REALTIME = "transcript:realtime"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcript-extractor-mcp")


# Request logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()

        logger.info("=== MCP Request Start ===")
        logger.info(f"Method: {request.method}")
        logger.info(f"URL: {request.url}")
        logger.info(f"Headers: {dict(request.headers)}")
        logger.info(f"Client: {request.client}")

        stack = traceback.extract_stack()
        logger.info("=== Call Stack Before Response ===")
        for frame in stack[-10:]:  # Only show last 10 frames
            logger.info(f"{frame.filename}:{frame.lineno} in {frame.name}")
            if frame.line:
                logger.info(f"  {frame.line}")

        response = await call_next(request)

        processing_time = time.time() - start_time
        logger.info(f"=== MCP Request End ===")
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Processing time: {processing_time:.3f}s")
        logger.info("=" * 50)

        return response


mcp = FastMCP("transcript-extractor", stateless_http=True)

# Server hardware limits configuration
MODEL_HIERARCHY = ALL_MODELS
MAX_MODEL = os.getenv(ENV_MAX_WHISPER_MODEL, "large-v3")


def get_max_model_index():
    """Get the maximum allowed model index based on server configuration."""
    try:
        return MODEL_HIERARCHY.index(MAX_MODEL)
    except ValueError:
        logger.warning(f"Invalid MAX_WHISPER_MODEL '{MAX_MODEL}', defaulting to 'base'")
        return MODEL_HIERARCHY.index("base")


def validate_model_request(requested_model: str) -> tuple[str, bool]:
    """
    Validate if the requested model is within server hardware limits.
    Breeze ASR 25 bypasses hardware limits.

    Returns:
        tuple: (actual_model_to_use, is_downgraded)
    """
    # Breeze ASR 25 bypasses hardware limits
    if requested_model == BREEZE_MODEL:
        return requested_model, False

    max_index = get_max_model_index()

    try:
        requested_index = MODEL_HIERARCHY.index(requested_model)
    except ValueError:
        logger.warning(f"Unknown model '{requested_model}', using 'base'")
        return "base", True

    if requested_index <= max_index:
        return requested_model, False
    else:
        downgraded_model = MODEL_HIERARCHY[max_index]
        logger.warning(
            f"Model '{requested_model}' exceeds server limits, downgrading to '{downgraded_model}'"
        )
        return downgraded_model, True


logger.info(f"Server maximum model: {MAX_MODEL}")
logger.info(f"Available models: {MODEL_HIERARCHY[:get_max_model_index() + 1]}")


_transcription_service: Optional[TranscriptionService] = None


def get_transcription_service() -> TranscriptionService:
    """Get the transcription service instance."""
    if _transcription_service is None:
        raise RuntimeError(
            "TranscriptionService not initialized. "
            "This should not happen - service should be initialized in main()."
        )
    return _transcription_service


def init_mcp_auth():
    """Initialize MCP Auth with OIDC configuration."""
    auth_server_url = os.getenv(ENV_AUTH_SERVER_URL, "http://localhost:4444")

    try:
        server_config = fetch_server_config(auth_server_url, type=AuthServerType.OIDC)

        mcp_auth = MCPAuth(server=server_config)
        logger.info(f"MCP Auth initialized with OIDC server: {auth_server_url}")
        return mcp_auth

    except Exception as e:
        logger.warning(f"Failed to initialize MCP Auth: {e}")
        return None


mcp_auth = init_mcp_auth()


@mcp.tool()
def extract_youtube_transcript(
    url: str,
    model: str = "base",
    language: Optional[str] = None,
    format: str = "text",
    diarize: bool = False,
) -> dict[str, Any]:
    """Extract transcript from YouTube video using WhisperX.

    The MCP client has full control over model selection and transcription parameters.
    Use list_whisper_models() to see available models and their characteristics.

    Args:
        url: YouTube video URL (required)
        model: Whisper model size - client chooses based on speed/accuracy tradeoff
               Options: tiny, base, small, medium, large-v2, large-v3 (default: base)
        language: Language code for forced recognition (e.g., zh, en)
                 Leave None for automatic language detection (default: None)
        format: Output format preference - text, srt, or vtt (default: text)
        diarize: Enable speaker diarization to identify different speakers (default: false)

    Returns:
        Dictionary containing transcript data and metadata:
        {
            "success": bool,
            "transcript": str (in requested format),
            "detected_language": str,
            "model_used": str,
            "processing_time": float,
            "all_formats": {"text": str, "srt": str, "vtt": str},
            "youtube_transcripts": dict,
            "client_params": dict (echo of client parameters)
        }
    """
    try:
        if mcp_auth and mcp_auth.auth_info:
            logger.info(
                f"Authenticated request from: {mcp_auth.auth_info.claims.get('sub', 'unknown')}"
            )

        actual_model, was_downgraded = validate_model_request(model)

        config = TranscriptionConfig(
            url=url,
            model_name=actual_model,  # Use validated model
            language=language,
            diarize=diarize,
        )

        progress_messages = []
        start_time = __import__("time").time()

        def progress_callback(message: str) -> None:
            progress_messages.append(message)
            logger.info(message)

        progress_callback(f"Client requested model: {model}")
        if was_downgraded:
            progress_callback(
                f"⚠️  Model downgraded to '{actual_model}' due to server hardware limits (max: {MAX_MODEL})"
            )
        else:
            progress_callback(f"✅ Using requested model: {actual_model}")
        progress_callback("Starting transcription...")

        result = get_transcription_service().transcribe_youtube_video(
            config, progress_callback
        )

        if not result.success:
            return {
                "success": False,
                "error": result.error_message,
                "progress": progress_messages,
            }

        processing_time = __import__("time").time() - start_time

        transcript_data = {
            "text": result.transcript_text,
            "srt": result.transcript_srt,
            "vtt": result.transcript_vtt,
        }

        client_params = {
            "requested_model": model,
            "language": language,
            "format": format,
            "diarize": diarize,
        }

        server_info = {
            "max_allowed_model": MAX_MODEL,
            "available_models": MODEL_HIERARCHY[: get_max_model_index() + 1],
            "model_downgraded": was_downgraded,
            "downgrade_reason": (
                f"Client requested '{model}' but server max is '{MAX_MODEL}'"
                if was_downgraded
                else None
            ),
        }

        return {
            "success": True,
            "transcript": transcript_data.get(format, result.transcript_text),
            "detected_language": result.detected_language,
            "model_used": actual_model,  # Actual model that was used
            "model_requested": model,  # What client requested
            "processing_time": round(processing_time, 2),
            "youtube_transcripts": result.youtube_transcripts,
            "progress": progress_messages,
            "all_formats": transcript_data,
            "client_params": client_params,
            "server_info": server_info,
        }

    except Exception as e:
        logger.error(f"Error extracting transcript: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_youtube_transcripts(url: str) -> dict[str, Any]:
    """Get existing YouTube transcripts without audio processing.

    Args:
        url: YouTube video URL

    Returns:
        Dictionary containing available YouTube transcripts
    """
    try:
        from .core.downloader import YouTubeDownloader

        downloader = YouTubeDownloader()
        transcripts = downloader.get_youtube_transcripts(url)

        return {
            "success": True,
            "transcripts": transcripts,
            "languages": list(transcripts.keys()) if transcripts else [],
        }

    except Exception as e:
        logger.error(f"Error getting YouTube transcripts: {e}")
        return {"success": False, "error": str(e), "transcripts": {}, "languages": []}


@mcp.tool()
def list_whisper_models() -> dict[str, Any]:
    """List available Whisper models.

    Returns:
        Dictionary containing model information
    """
    models = {
        "tiny": {
            "speed": "Fastest",
            "accuracy": "Basic",
        },
        "base": {
            "speed": "Fast",
            "accuracy": "Good",
        },
        "small": {
            "speed": "Medium",
            "accuracy": "Better",
        },
        "medium": {
            "speed": "Slow",
            "accuracy": "High",
        },
        "large-v2": {
            "speed": "Very Slow",
            "accuracy": "Excellent",
        },
        "large-v3": {
            "speed": "Very Slow",
            "accuracy": "Best",
        },
    }

    max_index = get_max_model_index()
    available_models = {
        k: v for i, (k, v) in enumerate(models.items()) if i <= max_index
    }

    return {
        "available_models": list(available_models.keys()),
        "models": available_models,
        "max_model": MAX_MODEL,
    }


async def main():
    """Run the MCP server with optional authentication."""
    global _transcription_service

    def service_progress_callback(message: str) -> None:
        logger.info(f"Service: {message}")

    _transcription_service = TranscriptionService(
        progress_callback=service_progress_callback,
        device=os.getenv(ENV_MCP_DEVICE),
        compute_type=os.getenv(ENV_MCP_COMPUTE_TYPE, "float16"),
    )

    debug_port = int(os.getenv("DEBUG_PORT", "5678"))
    if os.getenv("DEBUG_ENABLED", "false").lower() == "true":
        debugpy.listen(("0.0.0.0", debug_port))
        logger.info(f"Debugpy listening on port {debug_port}")
        if os.getenv("DEBUG_WAIT", "false").lower() == "true":
            logger.info("Waiting for debugger to attach...")
            debugpy.wait_for_client()
            logger.info("Debugger attached!")

    transport = os.getenv(ENV_MCP_TRANSPORT, TRANSPORT_STDIO)

    if transport == TRANSPORT_HTTP:
        host = os.getenv(ENV_MCP_HOST, "0.0.0.0")
        port = int(os.getenv(ENV_MCP_PORT, "8080"))

        logger.info(f"Starting MCP HTTP server on {host}:{port}")

        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def health_check(_request):
            return JSONResponse({"status": "ok", "service": "transcript-extractor-mcp"})

        if mcp_auth:
            from starlette.middleware import Middleware

            bearer_auth_params = {
                # "audience": os.getenv(ENV_MCP_JWT_AUDIENCE, "transcript-extractor"),
                "required_scopes": None,
                "leeway": 60,
                "show_error_details": True,  # Enable detailed error information
            }

            bearer_auth = Middleware(
                mcp_auth.bearer_auth_middleware("jwt", **bearer_auth_params)
            )

            mcp_app = mcp.streamable_http_app()
            logger.info(f"MCP app type: {type(mcp_app)}")

            app = Starlette(
                routes=[
                    Route("/health", health_check),
                    mcp_auth.metadata_route(),
                    Mount(
                        "/",
                        app=mcp_app,
                        middleware=[
                            bearer_auth,
                            # Middleware(RequestLoggingMiddleware)
                        ],
                    ),
                ],
                lifespan=(
                    mcp_app.router.lifespan_context
                    if hasattr(mcp_app.router, "lifespan_context")
                    else None
                ),
            )
        else:
            logger.error(
                "HTTP transport requires authentication. Please configure AUTH_SERVER_URL environment variables."
            )
            raise RuntimeError(
                "HTTP transport mode requires MCP Auth configuration. Authentication failed to initialize."
            )

        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    else:
        from mcp.server.stdio import stdio_server

        logger.info("Starting MCP stdio server")
        async with stdio_server() as (read_stream, write_stream):
            await mcp.run(
                read_stream, write_stream, mcp.create_initialization_options()
            )


def main_mcp():
    asyncio.run(main())


if __name__ == "__main__":
    print("Please use one of the following entry points:")
    print("  transcript-extractor       # CLI mode")
    print(
        "  transcript-extractor-mcp   # MCP server mode (only support http transport)"
    )
