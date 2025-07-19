from pathlib import Path
from typing import Dict, Optional, Union
import torch

from .base_transcriber import BaseTranscriber


class BreezeTranscriber(BaseTranscriber):
    """Speech-to-text transcriber using Breeze ASR 25 for Mandarin-English code-switching."""

    def __init__(
        self,
        device: Optional[str] = None,
        model_store_dir: Optional[Path] = None,
    ):
        """Initialize the Breeze transcriber.

        Args:
            device: Device to run on ("cpu", "cuda"). Auto-detect if None.
            model_store_dir: Directory to store downloaded models (default: ./models)
        """
        super().__init__(device=device, model_store_dir=model_store_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        diarize: bool = False,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Dict:
        """Transcribe audio file to text using Breeze ASR 25.

        Args:
            audio_path: Path to audio file
            language: Language code (currently not used by Breeze, auto-detects)
            diarize: Whether to perform speaker diarization (not supported)
            num_speakers: Number of speakers (not supported)
            min_speakers: Minimum number of speakers (not supported)
            max_speakers: Maximum number of speakers (not supported)

        Returns:
            Dictionary with transcription results in WhisperX-compatible format

        Raises:
            Exception: If transcription fails
        """
        try:
            from transformers import (
                WhisperProcessor,
                WhisperForConditionalGeneration,
                AutomaticSpeechRecognitionPipeline,
            )
            import librosa
            import gc

            model_name = "MediaTek-Research/Breeze-ASR-25"
            breeze_cache_dir = self.model_store_dir / "breeze-asr-25"
            breeze_cache_dir.mkdir(parents=True, exist_ok=True)

            processor = WhisperProcessor.from_pretrained(
                model_name, cache_dir=str(breeze_cache_dir)
            )
            model = WhisperForConditionalGeneration.from_pretrained(
                model_name, cache_dir=str(breeze_cache_dir)
            )
            model.to(self.device)
            model.eval()

            audio, _ = librosa.load(str(audio_path), sr=16000)

            asr_pipeline = AutomaticSpeechRecognitionPipeline(
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=0,  # Allows processing of long audio files
                device=self.device,
            )

            # Perform transcription with timestamps
            output = asr_pipeline(audio, return_timestamps=True)
            transcription = output["text"]

            if "chunks" in output and output["chunks"]:
                segments = []
                for chunk in output["chunks"]:
                    timestamp = chunk.get("timestamp", (0.0, 0.0))
                    text = chunk.get("text", "").strip()

                    # Skip empty chunks or invalid timestamps
                    if text and timestamp[0] < timestamp[1]:
                        segments.append(
                            {"start": timestamp[0], "end": timestamp[1], "text": text}
                        )

                result = {"segments": segments, "language": language or "zh"}
            else:
                result = {
                    "segments": [{"text": transcription}],
                    "language": language or "zh",
                }

            # Speaker diarization is not supported with Breeze ASR 25
            # due to lack of word-level timestamps
            if diarize:
                import warnings

                warnings.warn(
                    "Speaker diarization is not supported with Breeze ASR 25 due to lack of word-level timestamps. "
                    "Please use a WhisperX model (tiny, base, small, medium, large-v2, large-v3) for diarization. "
                    "Continuing without diarization...",
                    UserWarning,
                )

            return result

        except Exception as e:
            raise Exception(
                f"Failed to transcribe audio with Breeze ASR 25 {audio_path}: {str(e)}"
            )
        finally:
            # Manually clean up memory resources. The frameworks  may not immediately release GPU memory after use.
            # This explicit cleanup is necessary to prevent memory leaks or out-of-memory
            # errors, especially when processing multiple files in a sequence.
            if "asr_pipeline" in locals():
                del asr_pipeline
            if "model" in locals():
                del model
            if "processor" in locals():
                del processor

            # Trigger Python's garbage collector to reclaim memory from the deleted objects.
            gc.collect()

            # If using a GPU, clear the CUDA cache to free up unused memory held by PyTorch.
            if self.device.startswith("cuda"):
                import torch  # Ensure torch is imported here if not already

                torch.cuda.empty_cache()

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
        lines = []

        for segment in segments:
            text = segment["text"].strip()
            if text:
                lines.append(text)

        return "\n".join(lines)

    def _format_srt(self, result: Dict) -> str:
        """Format Breeze ASR 25 results as SRT subtitle file."""
        segments = result.get("segments", [])
        srt_content = []
        subtitle_index = 1

        for segment in segments:
            if "start" in segment and "end" in segment:
                start_time = self._seconds_to_srt_time(segment["start"])
                end_time = self._seconds_to_srt_time(segment["end"])
                text = segment["text"].strip()

                if text:  # Only add non-empty segments
                    srt_content.append(f"{subtitle_index}")
                    srt_content.append(f"{start_time} --> {end_time}")
                    srt_content.append(text)
                    srt_content.append("")
                    subtitle_index += 1
            else:
                # No timing information, return as plain text
                return self._format_text(result)

        return "\n".join(srt_content)

    def _format_vtt(self, result: Dict) -> str:
        """Format Breeze ASR 25 results as WebVTT file."""
        segments = result.get("segments", [])
        vtt_content = ["WEBVTT", ""]

        for segment in segments:
            if "start" in segment and "end" in segment:
                start_time = self._seconds_to_vtt_time(segment["start"])
                end_time = self._seconds_to_vtt_time(segment["end"])
                text = segment["text"].strip()

                if text:  # Only add non-empty segments
                    vtt_content.append(f"{start_time} --> {end_time}")
                    vtt_content.append(text)
                    vtt_content.append("")
            else:
                # No timing information, return as plain text
                return self._format_text(result)

        return "\n".join(vtt_content)
