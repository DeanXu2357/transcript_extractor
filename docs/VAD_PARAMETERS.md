# WhisperX VAD Parameters Research Notes

## Background

This document records the actual VAD (Voice Activity Detection) parameter support in WhisperX based on source code analysis. The purpose is to provide factual information for future development and avoid confusion with other libraries.

## WhisperX VAD Parameter Support

Based on source code analysis (`/whisperx/asr.py`, `/whisperx/vads/silero.py`, `/whisperx/vads/pyannote.py`), WhisperX **supports only 3 VAD parameters**:

### 1. `vad_onset` (float, default: 0.5)
- **Function**: Speech onset detection threshold (0-1)
- **Silero VAD**: Passed as `threshold` parameter to `get_speech_timestamps()`
- **Pyannote VAD**: Passed as `onset` parameter to `Binarize` class

### 2. `vad_offset` (float, default: 0.363)
- **Function**: Speech offset detection threshold
- **Silero VAD**: **IGNORED** (this parameter is not used)
- **Pyannote VAD**: Passed as `offset` parameter to `Binarize` class

### 3. `chunk_size` (int, default: 30)
- **Function**: Maximum speech segment duration in seconds
- **Silero VAD**: Passed as `max_speech_duration_s` parameter
- **Pyannote VAD**: Used as `max_duration` parameter in `merge_chunks()`

## VAD Method Differences

### Silero VAD (`vad_method="silero"`)
- Uses only `vad_onset` and `chunk_size`
- `vad_offset` is ignored
- Probability-based speech detection

### Pyannote VAD (`vad_method="pyannote"`)
- Uses all 3 parameters
- Supports separate onset/offset detection
- Deep learning-based speech detection

## Unsupported Parameters

The following parameters are common in other VAD libraries but **NOT supported** in WhisperX:

### Silero VAD native parameters not exposed by WhisperX:
- `min_speech_duration_ms` - Minimum speech segment duration
- `min_silence_duration_ms` - Minimum silence duration
- `window_size_samples` - Window size for processing
- `speech_pad_ms` - Padding around speech segments
- `return_seconds` - Return format control
- `per_channel` - Multi-channel processing

### Pyannote VAD parameters that are hardcoded:
```python
# Hardcoded in load_vad_model()
hyperparameters = {
    "onset": vad_onset,
    "offset": vad_offset,
    "min_duration_on": 0.1,   # HARDCODED
    "min_duration_off": 0.1   # HARDCODED
}
```

## Code Usage Example

```python
vad_options = {
    "vad_onset": 0.5,     # Speech detection threshold
    "vad_offset": 0.363,  # Only effective with pyannote VAD
    "chunk_size": 30      # Maximum segment duration in seconds
}

model = whisperx.load_model(
    model_size, device, 
    vad_options=vad_options
)
```

## Alternative Approaches

For more fine-grained VAD control, consider:

1. **Use faster-whisper directly**: Supports more VAD parameters
2. **Audio preprocessing**: Use Silero VAD or pyannote-audio for pre-segmentation
3. **Post-processing**: Merge or split segments after transcription

## Conclusion

WhisperX has limited VAD parameter support with only 3 configurable parameters, and some parameters are only effective with specific VAD methods. For more precise voice activity detection control, alternative tools or pre/post-processing approaches should be considered.