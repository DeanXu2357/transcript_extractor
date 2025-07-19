"""Base transcriber interface for all transcription implementations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union


class BaseTranscriber(ABC):
    """Abstract base class for all transcriber implementations."""

    def __init__(self, device: Optional[str] = None, model_store_dir: Optional[Path] = None):
        """Initialize the transcriber.

        Args:
            device: Device to run on ("cpu", "cuda"). Auto-detect if None.
            model_store_dir: Directory to store downloaded models (default: ./models)
        """
        self.device = device
        self.model_store_dir = model_store_dir or Path("./models")
        self.model_store_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        diarize: bool = False,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Dict:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., "zh", "en"). Auto-detect if None.
            diarize: Whether to perform speaker diarization
            num_speakers: Number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            Dictionary with transcription results including segments and metadata

        Raises:
            Exception: If transcription fails
        """
        pass

    @abstractmethod
    def format_transcript(self, result: Dict, format_type: str = "text") -> str:
        """Format transcription result.

        Args:
            result: Transcription result from transcribe_audio
            format_type: Output format ("text", "srt", "vtt")

        Returns:
            Formatted transcript string

        Raises:
            ValueError: If format_type is not supported
        """
        pass

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to VTT time format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"