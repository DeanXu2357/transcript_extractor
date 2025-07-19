import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from .downloader import YouTubeDownloader
from .transcriber import WhisperTranscriber
from .breeze_transcriber import BreezeTranscriber
from .cache import CacheService, with_cache
from .constants import WHISPER_MODELS, BREEZE_MODEL
from .base_transcriber import BaseTranscriber


@dataclass
class TranscriptionConfig:
    """Configuration for transcription service."""

    url: str
    model_name: str = "base"
    language: Optional[str] = None
    device: Optional[str] = None
    compute_type: str = "float16"
    diarize: bool = False
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


@dataclass
class TranscriptionResult:
    """Result of transcription service."""

    transcript_text: str
    transcript_srt: str
    transcript_vtt: str
    raw_result: Dict[str, Any]
    detected_language: str
    youtube_transcripts: Dict[str, str]
    success: bool = True
    error_message: Optional[str] = None


class TranscriptionService:
    """Core transcription service that handles YouTube download and transcription."""

    def __init__(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
        device: Optional[str] = None,
        compute_type: str = "float16",
    ):
        """Initialize the service.

        Args:
            progress_callback: Optional callback function for progress updates
            device: Device to run transcription on (auto-detect if None)
            compute_type: Compute precision for transcription models
        """
        self.progress_callback = progress_callback or (lambda _: None)

        self.device = device
        self.compute_type = compute_type

        self.download_dir = Path(os.getenv("DOWNLOAD_DIR", "./downloads"))
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.model_store_dir = Path(os.getenv("MODEL_STORE_DIR", "./models"))
        self.model_store_dir.mkdir(parents=True, exist_ok=True)

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

        self.transcribers: Dict[str, BaseTranscriber] = {}
        self._initialize_transcribers()

    def _initialize_transcribers(self):
        """Initialize all supported transcriber instances."""
        self.progress_callback("Initializing transcriber instances...")
        self.progress_callback(f"System device: {self.device or 'auto-detect'}")
        self.progress_callback(f"System compute_type: {self.compute_type}")

        try:
            for model_name in WHISPER_MODELS:
                self.transcribers[model_name] = WhisperTranscriber(
                    model_name="base",
                    device=self.device,
                    compute_type=self.compute_type,
                    model_store_dir=self.model_store_dir,
                )
                self.progress_callback(
                    f"Initialized shared WhisperX transcriber for models: {WHISPER_MODELS}"
                )

            self.transcribers[BREEZE_MODEL] = BreezeTranscriber(
                device=self.device,
                model_store_dir=self.model_store_dir,
            )
            self.progress_callback(f"Initialized Breeze transcriber: {BREEZE_MODEL}")
        except Exception as e:
            self.progress_callback(f"Failed to initialize Breeze transcriber: {e}")

        self.progress_callback(
            f"Initialized transcriber instances for {len(set(self.transcribers.values()))} unique transcribers"
        )

    def transcribe_youtube_video(
        self,
        config: TranscriptionConfig,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TranscriptionResult:
        """Transcribe a YouTube video.

        Args:
            config: Transcription configuration
            progress_callback: Optional callback for progress updates (overrides instance callback)

        Returns:
            TranscriptionResult containing all formats and metadata
        """
        callback = progress_callback or self.progress_callback

        try:
            callback(f"Using download directory: {self.download_dir}")
            callback(f"Model: {config.model_name}")
            callback(f"Language: {config.language or 'auto-detect'}")

            callback("Fetching YouTube transcripts...")
            youtube_transcripts = self.downloader.get_youtube_transcripts(config.url)
            if youtube_transcripts:
                callback(
                    f"Found YouTube transcripts in {len(youtube_transcripts)} languages"
                )
            else:
                callback("No YouTube transcripts available")

            callback("Downloading audio...")
            audio_path = self.downloader.download_audio(config.url, format="wav")
            callback(f"Audio downloaded to: {audio_path}")

            transcriber = self._get_transcriber(config)

            callback("Loading model and transcribing...")
            raw_result = transcriber.transcribe_audio(
                audio_path,
                language=config.language,
                diarize=config.diarize,
                num_speakers=config.num_speakers,
                min_speakers=config.min_speakers,
                max_speakers=config.max_speakers,
            )

            detected_language = raw_result.get("language", "unknown")
            callback(f"Detected language: {detected_language}")

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
                detected_language="unknown",
                youtube_transcripts={},
                success=False,
                error_message=str(e),
            )

    def _get_transcriber(self, config: TranscriptionConfig) -> BaseTranscriber:
        """Get the appropriate transcriber instance based on model name.

        Args:
            config: Transcription configuration

        Returns:
            Pre-initialized transcriber instance

        Raises:
            ValueError: If the requested model is not available
        """
        if config.model_name not in self.transcribers:
            available_models = list(self.transcribers.keys())
            raise ValueError(
                f"Model '{config.model_name}' not available. "
                f"Available models: {', '.join(available_models)}"
            )

        return self.transcribers[config.model_name]
