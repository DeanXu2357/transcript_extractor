from pathlib import Path
from typing import Dict, Optional, Union
import whisperx
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class WhisperTranscriber:
    """Speech-to-text transcriber using WhisperX."""

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: str = "float16",
        model_store_dir: Optional[Path] = None,
    ):
        """Initialize the transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Device to run on ("cpu", "cuda"). Auto-detect if None.
            compute_type: Compute precision ("float16", "int8", "float32")
            model_store_dir: Directory to store downloaded models (default: ./models)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type
        self.model_store_dir = model_store_dir or Path("./models")
        self.model_store_dir.mkdir(parents=True, exist_ok=True)
        
        # WhisperX configuration from environment
        import os
        self.batch_size = int(os.getenv("WHISPERX_BATCH_SIZE", "16"))
        self.compute_type = os.getenv("WHISPERX_COMPUTE_TYPE", compute_type)
        
        # Create whisper cache directory
        self.whisper_cache_dir = self.model_store_dir / "whisperx"
        self.whisper_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create alignment cache directory
        self.alignment_cache_dir = self.model_store_dir / "alignment"
        self.alignment_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_alignment_cache_dir(self, language: str) -> str:
        """Get alignment cache directory for specific language, creating if needed."""
        lang_cache_dir = self.alignment_cache_dir / language
        lang_cache_dir.mkdir(parents=True, exist_ok=True)
        return str(lang_cache_dir)






    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> Dict:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., "zh", "en"). Auto-detect if None.

        Returns:
            Dictionary with transcription results including segments and word timings

        Raises:
            Exception: If transcription fails
        """
        try:
            # Load model
            model = whisperx.load_model(
                self.model_size, 
                self.device, 
                compute_type=self.compute_type,
                download_root=str(self.whisper_cache_dir)
            )
            
            # Load and transcribe audio
            audio = whisperx.load_audio(str(audio_path))
            result = model.transcribe(audio, batch_size=self.batch_size, language=language)

            # Load alignment model and align output
            align_model, metadata = whisperx.load_align_model(
                language_code=result["language"], 
                device=self.device,
                model_dir=self._get_alignment_cache_dir(result["language"])
            )

            # Align whisper output
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

            return result

        except Exception as e:
            raise Exception(f"Failed to transcribe audio {audio_path}: {str(e)}")

    def format_transcript(self, result: Dict, format_type: str = "text") -> str:
        """Format transcription result.

        Args:
            result: Transcription result from transcribe_audio
            format_type: Output format ("text", "srt", "vtt")

        Returns:
            Formatted transcript string
        """
        if format_type == "text":
            return self._format_text(result)
        elif format_type == "srt":
            return self._format_srt(result)
        elif format_type == "vtt":
            return self._format_vtt(result)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _format_text(self, result: Dict) -> str:
        """Format as plain text."""
        segments = result.get("segments", [])
        return "\n".join(segment["text"].strip() for segment in segments)

    def _format_srt(self, result: Dict) -> str:
        """Format as SRT subtitle file."""
        segments = result.get("segments", [])
        srt_content = []

        for i, segment in enumerate(segments, 1):
            start_time = self._seconds_to_srt_time(segment["start"])
            end_time = self._seconds_to_srt_time(segment["end"])

            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(segment["text"].strip())
            srt_content.append("")

        return "\n".join(srt_content)

    def _format_vtt(self, result: Dict) -> str:
        """Format as WebVTT file."""
        segments = result.get("segments", [])
        vtt_content = ["WEBVTT", ""]

        for segment in segments:
            start_time = self._seconds_to_vtt_time(segment["start"])
            end_time = self._seconds_to_vtt_time(segment["end"])

            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(segment["text"].strip())
            vtt_content.append("")

        return "\n".join(vtt_content)

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
