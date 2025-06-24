#!/usr/bin/env python3
"""MCP Server for YouTube Transcript Extraction with mcp-auth integration."""

import asyncio
import logging
import os
from functools import wraps
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import Tool
from mcpauth import MCPAuth
from mcpauth.config import AuthServerType
from mcpauth.utils import fetch_server_config
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Mount
import uvicorn

from .core.service import TranscriptionConfig, transcribe_youtube_video

# Environment variable constants
ENV_MAX_WHISPER_MODEL = "MAX_WHISPER_MODEL"
ENV_AUTH_SERVER_URL = "AUTH_SERVER_URL"
ENV_MCP_TRANSPORT = "MCP_TRANSPORT"
ENV_MCP_HOST = "MCP_HOST"
ENV_MCP_PORT = "MCP_PORT"
ENV_MCP_JWT_AUDIENCE = "MCP_JWT_AUDIENCE"

# Transport constants
TRANSPORT_STDIO = "stdio"
TRANSPORT_HTTP = "http"


# OAuth2 scope constants
SCOPE_OPENID = "openid"
SCOPE_OFFLINE = "offline"
SCOPE_TRANSCRIPT_READ = "transcript:read"
SCOPE_TRANSCRIPT_REALTIME = "transcript:realtime"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcript-extractor-mcp")

# Create MCP server
mcp = FastMCP("transcript-extractor")

# Server hardware limits configuration
MODEL_HIERARCHY = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
MAX_MODEL = os.getenv(ENV_MAX_WHISPER_MODEL, "large-v3")  # Default to allow all models


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

    Returns:
        tuple: (actual_model_to_use, is_downgraded)
    """
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


# Log server configuration at startup
logger.info(f"Server maximum model: {MAX_MODEL}")
logger.info(f"Available models: {MODEL_HIERARCHY[:get_max_model_index() + 1]}")


# Initialize MCP Auth
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


# Initialize auth (optional)
mcp_auth = init_mcp_auth()


def require_scope(required_scope: str):
    """
    Decorator to check if the current authenticated user has the required OAuth2 scope.

    Args:
        required_scope: The OAuth2 scope required to access this tool

    Returns:
        Decorator function that checks scope before executing the tool
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # If no authentication is configured, allow access (for CLI mode)
            if not mcp_auth:
                logger.info(
                    f"No authentication configured, allowing access to {func.__name__}"
                )
                return func(*args, **kwargs)

            # Access auth_info during request execution (per-request context)
            auth_info = mcp_auth.auth_info
            if not auth_info:
                logger.warning(f"No authentication info available for {func.__name__}")
                return {
                    "success": False,
                    "error": "Authentication required",
                    "required_scope": required_scope,
                }

            # Check if user has the required scope
            user_scopes = auth_info.claims.get("scope", "").split()
            if required_scope not in user_scopes:
                logger.warning(
                    f"Access denied to {func.__name__}: required scope '{required_scope}' not found in user scopes: {user_scopes}"
                )
                return {
                    "success": False,
                    "error": f"Insufficient permissions. Required scope: {required_scope}",
                    "required_scope": required_scope,
                    "user_scopes": user_scopes,
                }

            logger.info(
                f"Access granted to {func.__name__}: user has required scope '{required_scope}'"
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


@require_scope(SCOPE_TRANSCRIPT_REALTIME)
@mcp.tool()
def extract_youtube_transcript(
    url: str,
    model: str = "base",
    language: Optional[str] = None,
    format: str = "text",
    audio_format: str = "wav",
    keep_audio: bool = False,
    device: Optional[str] = None,
    compute_type: str = "float16",
    align: bool = True,
    temp_dir: Optional[str] = None,
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
        audio_format: Audio download format - wav, mp3, or flac (default: wav)
        keep_audio: Whether to preserve downloaded audio file (default: false)
        device: Compute device preference - cpu or cuda (default: auto-detect)
        compute_type: Precision mode - float16, float32, or int8 (default: float16)
        align: Enable word-level timestamp alignment (default: true)
        temp_dir: Custom temporary directory path (default: system temp)

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
            "audio_path": str (if keep_audio=true),
            "client_params": dict (echo of client parameters)
        }
    """
    try:
        # Check authentication if mcp_auth is available
        if mcp_auth and mcp_auth.auth_info:
            logger.info(
                f"Authenticated request from: {mcp_auth.auth_info.claims.get('sub', 'unknown')}"
            )

        # Validate and potentially downgrade model based on server limits
        actual_model, was_downgraded = validate_model_request(model)

        # Create configuration
        config = TranscriptionConfig(
            url=url,
            model_size=actual_model,  # Use validated model
            language=language,
            audio_format=audio_format,
            device=device,
            compute_type=compute_type,
            align=align,
            temp_dir=Path(temp_dir) if temp_dir else None,
            keep_audio=keep_audio,
        )

        # Progress callback
        progress_messages = []
        start_time = __import__("time").time()

        def progress_callback(message: str) -> None:
            progress_messages.append(message)
            logger.info(message)

        # Log client's model choice and server decision
        progress_callback(f"Client requested model: {model}")
        if was_downgraded:
            progress_callback(
                f"⚠️  Model downgraded to '{actual_model}' due to server hardware limits (max: {MAX_MODEL})"
            )
        else:
            progress_callback(f"✅ Using requested model: {actual_model}")
        progress_callback(
            f"Client parameters: device={device}, compute_type={compute_type}, align={align}"
        )

        # Run transcription
        result = transcribe_youtube_video(config, progress_callback)

        if not result.success:
            return {
                "success": False,
                "error": result.error_message,
                "progress": progress_messages,
            }

        # Calculate processing time
        processing_time = __import__("time").time() - start_time

        # Return result based on requested format
        transcript_data = {
            "text": result.transcript_text,
            "srt": result.transcript_srt,
            "vtt": result.transcript_vtt,
        }

        # Echo client parameters for transparency
        client_params = {
            "requested_model": model,
            "language": language,
            "format": format,
            "audio_format": audio_format,
            "keep_audio": keep_audio,
            "device": device,
            "compute_type": compute_type,
            "align": align,
            "temp_dir": temp_dir,
        }

        # Server decision information
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
            "audio_path": str(result.audio_path) if result.audio_path else None,
            "progress": progress_messages,
            "all_formats": transcript_data,
            "client_params": client_params,
            "server_info": server_info,
        }

    except Exception as e:
        logger.error(f"Error extracting transcript: {e}")
        return {"success": False, "error": str(e)}


@require_scope(SCOPE_TRANSCRIPT_READ)
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


@require_scope(SCOPE_TRANSCRIPT_READ)
@mcp.tool()
def list_whisper_models() -> dict[str, Any]:
    """List available Whisper models for client selection.

    This tool helps MCP clients understand the available models and make informed
    choices based on their speed/accuracy/resource requirements.

    Returns:
        Dictionary containing detailed model information for client decision-making
    """
    models = {
        "tiny": {
            "parameters": "39M",
            "vram": "~1GB",
            "relative_speed": "32x",
            "accuracy": "⭐",
            "use_case": "Quick testing, real-time transcription",
            "pros": ["Fastest", "Lowest memory", "Good for long videos"],
            "cons": ["Lower accuracy", "May miss complex words"],
        },
        "base": {
            "parameters": "74M",
            "vram": "~1GB",
            "relative_speed": "16x",
            "accuracy": "⭐⭐",
            "use_case": "Balanced performance, general purpose",
            "pros": ["Good speed/accuracy balance", "Reasonable memory usage"],
            "cons": ["May struggle with accents", "Not best for technical content"],
        },
        "small": {
            "parameters": "244M",
            "vram": "~2GB",
            "relative_speed": "6x",
            "accuracy": "⭐⭐⭐",
            "use_case": "Quality transcription with acceptable speed",
            "pros": ["Better accuracy", "Handles multiple languages well"],
            "cons": ["Slower than base", "More memory required"],
        },
        "medium": {
            "parameters": "769M",
            "vram": "~5GB",
            "relative_speed": "2x",
            "accuracy": "⭐⭐⭐⭐",
            "use_case": "High-quality transcription, professional use",
            "pros": ["High accuracy", "Good with technical terms", "Multi-language"],
            "cons": ["Significantly slower", "Requires more VRAM"],
        },
        "large-v2": {
            "parameters": "1550M",
            "vram": "~10GB",
            "relative_speed": "1x",
            "accuracy": "⭐⭐⭐⭐⭐",
            "use_case": "Best accuracy, research/archival work",
            "pros": [
                "Excellent accuracy",
                "Handles difficult audio",
                "Multi-language expert",
            ],
            "cons": ["Very slow", "High memory requirements"],
        },
        "large-v3": {
            "parameters": "1550M",
            "vram": "~10GB",
            "relative_speed": "1x",
            "accuracy": "⭐⭐⭐⭐⭐",
            "use_case": "State-of-the-art accuracy, latest model",
            "pros": [
                "Best available accuracy",
                "Latest improvements",
                "Superior multi-language",
            ],
            "cons": ["Very slow", "High memory requirements", "Requires powerful GPU"],
        },
    }

    # Filter models based on server hardware limits
    max_index = get_max_model_index()
    available_models = {
        k: v for i, (k, v) in enumerate(models.items()) if i <= max_index
    }

    # Filter recommendations based on available models
    available_model_names = list(available_models.keys())
    filtered_recommendations = {}

    for scenario, rec in {
        "quick_preview": {
            "model": "tiny",
            "reason": "Fastest processing, good for testing",
        },
        "daily_use": {"model": "base", "reason": "Best balance of speed and accuracy"},
        "high_quality": {
            "model": "medium",
            "reason": "Professional quality with reasonable speed",
        },
        "best_accuracy": {"model": "large-v3", "reason": "State-of-the-art accuracy"},
        "limited_memory": {"model": "tiny", "reason": "Only requires 1GB VRAM"},
        "gpu_accelerated": {"model": "small", "reason": "Good GPU utilization balance"},
    }.items():
        if rec["model"] in available_model_names:
            filtered_recommendations[scenario] = rec
        else:
            # Find the best available alternative
            for alt_model in reversed(available_model_names):
                if alt_model in available_model_names:
                    filtered_recommendations[scenario] = {
                        "model": alt_model,
                        "reason": f"Best available model on this server (requested {rec['model']} not available)",
                    }
                    break

    return {
        "success": True,
        "message": f"Server allows models up to '{MAX_MODEL}'. Choose based on your requirements.",
        "server_limits": {
            "max_model": MAX_MODEL,
            "available_models": available_model_names,
            "unavailable_models": [
                m for m in MODEL_HIERARCHY if m not in available_model_names
            ],
        },
        "models": available_models,  # Only show available models
        "recommendations": filtered_recommendations,
        "selection_guide": {
            "consider_video_length": "Longer videos may benefit from faster models",
            "consider_audio_quality": "Poor audio quality needs larger models",
            "consider_language": "Non-English content may need medium+ models",
            "consider_technical_content": "Technical/medical terms need larger models",
            "server_constraint": f"This server is limited to models up to '{MAX_MODEL}'",
        },
    }


async def main():
    """Run the MCP server with optional authentication."""
    transport = os.getenv(ENV_MCP_TRANSPORT, TRANSPORT_STDIO)

    if transport == TRANSPORT_HTTP:
        # HTTP transport with Starlette and mcp-auth
        host = os.getenv(ENV_MCP_HOST, "0.0.0.0")
        port = int(os.getenv(ENV_MCP_PORT, "8080"))

        logger.info(f"Starting MCP HTTP server on {host}:{port}")

        from starlette.responses import JSONResponse
        from starlette.routing import Route

        # Health check endpoint
        async def health_check(request):
            return JSONResponse({"status": "ok", "service": "transcript-extractor-mcp"})

        if mcp_auth:
            # Setup authenticated HTTP server with per-tool scope validation
            from starlette.middleware import Middleware

            # Configure bearer auth middleware for JWT token validation
            bearer_auth_params = {
                "audience": os.getenv(ENV_MCP_JWT_AUDIENCE, "transcript-extractor"),
                "required_scopes": None,  # Handle scopes per-tool
                "leeway": 60,
            }

            bearer_auth = Middleware(
                mcp_auth.bearer_auth_middleware("jwt", **bearer_auth_params)
            )

            app = Starlette(
                routes=[
                    Route("/health", health_check),
                    mcp_auth.metadata_route(),
                    Mount(
                        "/mcp", app=mcp.streamable_http_app(), middleware=[bearer_auth]
                    ),
                ]
            )
        else:
            # HTTP transport requires authentication
            logger.error(
                "HTTP transport requires authentication. Please configure AUTH_SERVER_URL and AUTH_SERVER_TYPE environment variables."
            )
            raise RuntimeError(
                "HTTP transport mode requires MCP Auth configuration. Authentication failed to initialize."
            )

        # Run server
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    else:
        # Standard stdio transport for local usage
        from mcp.server.stdio import stdio_server

        logger.info("Starting MCP stdio server")
        async with stdio_server() as (read_stream, write_stream):
            await mcp.run(
                read_stream, write_stream, mcp.create_initialization_options()
            )


def main_mcp():
    """Entry point for MCP server."""
    asyncio.run(main())


if __name__ == "__main__":
    print("Please use one of the following entry points:")
    print("  transcript-extractor       # CLI mode")
    print(
        "  transcript-extractor-mcp   # MCP server mode (only support http transport)"
    )
