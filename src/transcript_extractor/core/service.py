import tempfile
import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from .downloader import YouTubeDownloader
from .transcriber import WhisperTranscriber
from .cache import CacheService, with_cache


@dataclass
class TranscriptionConfig:
    """Configuration for transcription service."""

    url: str
    model_size: str = "base"
    language: Optional[str] = None
    audio_format: str = "wav"
    device: Optional[str] = None
    compute_type: str = "float32"
    align: bool = True
    temp_dir: Optional[Path] = None
    keep_audio: bool = False


@dataclass
class TranscriptionResult:
    """Result of transcription service."""

    transcript_text: str
    transcript_srt: str
    transcript_vtt: str
    raw_result: Dict[str, Any]
    audio_path: Optional[Path]
    detected_language: str
    youtube_transcripts: Dict[
        str, str
    ]  # Language code -> transcript content from YouTube
    success: bool = True
    error_message: Optional[str] = None


class TranscriptionService:
    """Core transcription service that handles YouTube download and transcription."""

    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None):
        """Initialize the service.

        Args:
            progress_callback: Optional callback function for progress updates
        """
        self.progress_callback = progress_callback or (lambda x: None)

        self.download_dir = Path(os.getenv("DOWNLOAD_DIR", "./downloads"))
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.downloader = YouTubeDownloader(output_dir=str(self.download_dir))

        try:
            self.cache_service = CacheService()

            self.downloader.download_audio = with_cache(
                cache_service=self.cache_service,
                progress_callback=self.progress_callback,
            )(self.downloader.download_audio)
            self.progress_callback("Cache service initialized successfully")
        except Exception as e:
            self.progress_callback(f"Warning: Cache service unavailable: {e}")
            self.cache_service = None

    def transcribe_youtube_video(
        self, config: TranscriptionConfig
    ) -> TranscriptionResult:
        """Transcribe a YouTube video.

        Args:
            config: Transcription configuration

        Returns:
            TranscriptionResult containing all formats and metadata
        """
        try:
            self.progress_callback(f"Using download directory: {self.download_dir}")
            self.progress_callback(f"Model: {config.model_size}")
            self.progress_callback(f"Language: {config.language or 'auto-detect'}")

            # Get YouTube transcripts first
            self.progress_callback("Fetching YouTube transcripts...")
            youtube_transcripts = self.downloader.get_youtube_transcripts(config.url)
            if youtube_transcripts:
                self.progress_callback(
                    f"Found YouTube transcripts in {len(youtube_transcripts)} languages"
                )
            else:
                self.progress_callback("No YouTube transcripts available")

            # Download audio
            self.progress_callback("Downloading audio...")
            audio_path = self.downloader.download_audio(
                config.url, format=config.audio_format
            )
            self.progress_callback(f"Audio downloaded to: {audio_path}")

            # Initialize transcriber
            transcriber = WhisperTranscriber(
                model_size=config.model_size,
                device=config.device,
                compute_type=config.compute_type,
            )

            # Transcribe audio
            self.progress_callback("Loading model and transcribing...")
            raw_result = transcriber.transcribe_audio(
                audio_path, language=config.language, align=config.align
            )

            detected_language = raw_result.get("language", "unknown")
            self.progress_callback(f"Detected language: {detected_language}")

            # Generate all formats
            transcript_text = transcriber.format_transcript(
                raw_result, format_type="text"
            )
            transcript_srt = transcriber.format_transcript(
                raw_result, format_type="srt"
            )
            transcript_vtt = transcriber.format_transcript(
                raw_result, format_type="vtt"
            )

            return TranscriptionResult(
                transcript_text=transcript_text,
                transcript_srt=transcript_srt,
                transcript_vtt=transcript_vtt,
                raw_result=raw_result,
                audio_path=audio_path if config.keep_audio else None,
                detected_language=detected_language,
                youtube_transcripts=youtube_transcripts,
                success=True,
            )

        except Exception as e:
            return TranscriptionResult(
                transcript_text="",
                transcript_srt="",
                transcript_vtt="",
                raw_result={},
                audio_path=None,
                detected_language="unknown",
                youtube_transcripts={},
                success=False,
                error_message=str(e),
            )


def transcribe_youtube_video(
    config: TranscriptionConfig,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> TranscriptionResult:
    """Convenience function for transcribing YouTube videos.

    Args:
        config: Transcription configuration
        progress_callback: Optional callback for progress updates

    Returns:
        TranscriptionResult containing all formats and metadata
    """
    service = TranscriptionService(progress_callback)
    return service.transcribe_youtube_video(config)
