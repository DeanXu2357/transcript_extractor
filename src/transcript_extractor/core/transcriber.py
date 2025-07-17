from pathlib import Path
from typing import Dict, Optional, Union
import whisperx
import torch


class WhisperTranscriber:
    """Speech-to-text transcriber using WhisperX."""

    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        compute_type: str = "float16",
        model_store_dir: Optional[Path] = None,
    ):
        """Initialize the transcriber.

        Args:
            model_name: Model name (tiny, base, small, medium, large-v2, large-v3, breeze-asr-25)
            device: Device to run on ("cpu", "cuda"). Auto-detect if None.
            compute_type: Compute precision ("float16", "int8", "float32")
            model_store_dir: Directory to store downloaded models (default: ./models)
        """
        self.model_name = model_name
        self.is_breeze_model = model_name == "breeze-asr-25"
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
            Dictionary with transcription results including segments and word timings

        Raises:
            Exception: If transcription fails
        """
        try:
            # Load model based on model type
            if self.is_breeze_model:
                return self._transcribe_with_breeze(
                    audio_path, language, diarize, num_speakers, min_speakers, max_speakers
                )
            
            # Load WhisperX model
            model = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type,
                download_root=str(self.whisper_cache_dir),
            )

            # Load and transcribe audio
            audio = whisperx.load_audio(str(audio_path))
            result = model.transcribe(
                audio, batch_size=self.batch_size, language=language
            )

            # Load alignment model and align output
            align_model, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=self.device,
                model_dir=self._get_alignment_cache_dir(result["language"]),
            )

            # Store language before alignment
            detected_language = result["language"]

            # Align whisper output
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

            # Preserve language information
            result["language"] = detected_language

            # Perform speaker diarization if requested
            if diarize:
                import os

                hf_token = os.getenv("HF_TOKEN")
                if hf_token:
                    diarize_model = whisperx.diarize.DiarizationPipeline(
                        use_auth_token=hf_token, device=self.device
                    )

                    # Build parameters for diarization
                    diarize_params = {}
                    if num_speakers is not None:
                        diarize_params["num_speakers"] = num_speakers
                    if min_speakers is not None:
                        diarize_params["min_speakers"] = min_speakers
                    if max_speakers is not None:
                        diarize_params["max_speakers"] = max_speakers

                    diarize_segments = diarize_model(audio, **diarize_params)
                    result = whisperx.assign_word_speakers(
                        diarize_segments, result, fill_nearest=False
                    )
                else:
                    raise Exception(
                        "HF_TOKEN environment variable required for speaker diarization"
                    )

            return result

        except Exception as e:
            raise Exception(f"Failed to transcribe audio {audio_path}: {str(e)}")

    def _transcribe_with_breeze(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        diarize: bool = False,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Dict:
        """Transcribe audio using Breeze ASR 25 model.
        
        Args:
            audio_path: Path to audio file
            language: Language code (currently not used by Breeze, auto-detects)
            diarize: Whether to perform speaker diarization
            num_speakers: Number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            
        Returns:
            Dictionary with transcription results in WhisperX-compatible format
        """
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutomaticSpeechRecognitionPipeline
            import librosa
            
            # Load Breeze ASR 25 model with caching
            model_name = "MediaTek-Research/Breeze-ASR-25"
            breeze_cache_dir = self.model_store_dir / "breeze-asr-25"
            breeze_cache_dir.mkdir(parents=True, exist_ok=True)
            
            processor = WhisperProcessor.from_pretrained(
                model_name, 
                cache_dir=str(breeze_cache_dir)
            )
            model = WhisperForConditionalGeneration.from_pretrained(
                model_name, 
                cache_dir=str(breeze_cache_dir)
            )
            model.to(self.device)
            model.eval()
            
            # Load and preprocess audio
            audio, _ = librosa.load(str(audio_path), sr=16000)
            
            # Create ASR pipeline for long audio processing
            asr_pipeline = AutomaticSpeechRecognitionPipeline(
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=0,  # Allows processing of long audio files
                device=self.device
            )
            
            # Perform transcription with timestamps
            output = asr_pipeline(audio, return_timestamps=True)
            transcription = output["text"]
            
            # Convert to WhisperX-compatible format using chunks
            if "chunks" in output and output["chunks"]:
                # Process chunks with timestamps
                segments = []
                for chunk in output["chunks"]:
                    timestamp = chunk.get("timestamp", (0.0, 0.0))
                    text = chunk.get("text", "").strip()
                    
                    # Skip empty chunks or invalid timestamps
                    if text and timestamp[0] < timestamp[1]:
                        segments.append({
                            "start": timestamp[0],
                            "end": timestamp[1],
                            "text": text
                        })
                
                result = {
                    "segments": segments,
                    "language": language or "zh"
                }
            else:
                # Fallback to single segment
                result = {
                    "segments": [{
                        "text": transcription
                    }],
                    "language": language or "zh"
                }
            
            # Speaker diarization is not supported with Breeze ASR 25
            # due to lack of word-level timestamps
            if diarize:
                import warnings
                warnings.warn(
                    "Speaker diarization is not supported with Breeze ASR 25 due to lack of word-level timestamps. "
                    "Please use a WhisperX model (tiny, base, small, medium, large-v2, large-v3) for diarization. "
                    "Continuing without diarization...",
                    UserWarning
                )
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to transcribe audio with Breeze ASR 25 {audio_path}: {str(e)}")

    def format_transcript(self, result: Dict, format_type: str = "text") -> str:
        """Format transcription result.

        Args:
            result: Transcription result from transcribe_audio
            format_type: Output format ("text", "srt", "vtt")

        Returns:
            Formatted transcript string
        """
        if self.is_breeze_model:
            return self._format_breeze_transcript(result, format_type)
        else:
            return self._format_whisperx_transcript(result, format_type)

    def _format_breeze_transcript(self, result: Dict, format_type: str) -> str:
        """Format Breeze ASR 25 transcript results."""
        if format_type == "text":
            return self._format_text(result)
        elif format_type == "srt":
            return self._format_breeze_srt(result)
        elif format_type == "vtt":
            return self._format_breeze_vtt(result)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _format_whisperx_transcript(self, result: Dict, format_type: str) -> str:
        """Format WhisperX transcript results."""
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
            # Check if words have speaker information
            words = segment.get("words", [])
            if words and any("speaker" in word for word in words):
                # Group words by speaker
                current_speaker = None
                current_text = []

                for word in words:
                    word_speaker = word.get("speaker")
                    if word_speaker != current_speaker:
                        # Speaker changed, output previous group
                        if current_speaker and current_text:
                            lines.append(
                                f"[{current_speaker}] {' '.join(current_text).strip()}"
                            )
                        current_speaker = word_speaker
                        current_text = []

                    if "word" in word:
                        current_text.append(word["word"])

                # Output final group
                if current_speaker and current_text:
                    lines.append(
                        f"[{current_speaker}] {' '.join(current_text).strip()}"
                    )
            else:
                # Fallback to segment-level speaker or no speaker
                text = segment["text"].strip()
                if "speaker" in segment:
                    lines.append(f"[{segment['speaker']}] {text}")
                else:
                    lines.append(text)

        return "\n".join(lines)

    def _format_srt(self, result: Dict) -> str:
        """Format as SRT subtitle file."""
        segments = result.get("segments", [])
        srt_content = []
        subtitle_index = 1

        for segment in segments:
            # Check if words have speaker information
            words = segment.get("words", [])
            if words and any("speaker" in word for word in words):
                # Group words by speaker within the segment
                current_speaker = None
                current_text = []
                current_start = segment["start"]

                for i, word in enumerate(words):
                    word_speaker = word.get("speaker")
                    if word_speaker != current_speaker:
                        # Speaker changed, create subtitle for previous group
                        if current_speaker and current_text:
                            word_end = words[i - 1].get("end", current_start + 1)
                            start_time = self._seconds_to_srt_time(current_start)
                            end_time = self._seconds_to_srt_time(word_end)
                            text = (
                                f"[{current_speaker}] {' '.join(current_text).strip()}"
                            )

                            srt_content.append(f"{subtitle_index}")
                            srt_content.append(f"{start_time} --> {end_time}")
                            srt_content.append(text)
                            srt_content.append("")
                            subtitle_index += 1

                        current_speaker = word_speaker
                        current_text = []
                        current_start = word.get("start", current_start)

                    if "word" in word:
                        current_text.append(word["word"])

                # Output final group in this segment
                if current_speaker and current_text:
                    end_time = self._seconds_to_srt_time(segment["end"])
                    start_time = self._seconds_to_srt_time(current_start)
                    text = f"[{current_speaker}] {' '.join(current_text).strip()}"

                    srt_content.append(f"{subtitle_index}")
                    srt_content.append(f"{start_time} --> {end_time}")
                    srt_content.append(text)
                    srt_content.append("")
                    subtitle_index += 1
            else:
                # Fallback to segment-level
                start_time = self._seconds_to_srt_time(segment["start"])
                end_time = self._seconds_to_srt_time(segment["end"])
                text = segment["text"].strip()

                if "speaker" in segment:
                    text = f"[{segment['speaker']}] {text}"

                srt_content.append(f"{subtitle_index}")
                srt_content.append(f"{start_time} --> {end_time}")
                srt_content.append(text)
                srt_content.append("")
                subtitle_index += 1

        return "\n".join(srt_content)

    def _format_vtt(self, result: Dict) -> str:
        """Format as WebVTT file."""
        segments = result.get("segments", [])
        vtt_content = ["WEBVTT", ""]

        for segment in segments:
            # Check if words have speaker information
            words = segment.get("words", [])
            if words and any("speaker" in word for word in words):
                # Group words by speaker within the segment
                current_speaker = None
                current_text = []
                current_start = segment["start"]

                for i, word in enumerate(words):
                    word_speaker = word.get("speaker")
                    if word_speaker != current_speaker:
                        # Speaker changed, create subtitle for previous group
                        if current_speaker and current_text:
                            word_end = words[i - 1].get("end", current_start + 1)
                            start_time = self._seconds_to_vtt_time(current_start)
                            end_time = self._seconds_to_vtt_time(word_end)
                            text = (
                                f"[{current_speaker}] {' '.join(current_text).strip()}"
                            )

                            vtt_content.append(f"{start_time} --> {end_time}")
                            vtt_content.append(text)
                            vtt_content.append("")

                        current_speaker = word_speaker
                        current_text = []
                        current_start = word.get("start", current_start)

                    if "word" in word:
                        current_text.append(word["word"])

                # Output final group in this segment
                if current_speaker and current_text:
                    end_time = self._seconds_to_vtt_time(segment["end"])
                    start_time = self._seconds_to_vtt_time(current_start)
                    text = f"[{current_speaker}] {' '.join(current_text).strip()}"

                    vtt_content.append(f"{start_time} --> {end_time}")
                    vtt_content.append(text)
                    vtt_content.append("")
            else:
                # Fallback to segment-level
                start_time = self._seconds_to_vtt_time(segment["start"])
                end_time = self._seconds_to_vtt_time(segment["end"])
                text = segment["text"].strip()

                if "speaker" in segment:
                    text = f"[{segment['speaker']}] {text}"

                vtt_content.append(f"{start_time} --> {end_time}")
                vtt_content.append(text)
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

    def _format_breeze_srt(self, result: Dict) -> str:
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

    def _format_breeze_vtt(self, result: Dict) -> str:
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
