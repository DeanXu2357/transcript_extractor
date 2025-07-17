#!/usr/bin/env python3
"""Integration tests for core functionality."""

import sys
import tempfile
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, '/app/src')

from transcript_extractor.core.downloader import YouTubeDownloader
from transcript_extractor.core.transcriber import WhisperTranscriber
from transcript_extractor.core.service import TranscriptionConfig, transcribe_youtube_video

# Test URLs - using short, safe videos
TEST_URLS = {
    "short_english": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll (short)
    "with_captions": "https://www.youtube.com/watch?v=jNQXAC9IVRw",   # Short video with captions
    "whisper_test": "https://www.youtube.com/watch?v=9bZkp7q19f0",    # Different short video for Whisper test
    "service_test": "https://www.youtube.com/watch?v=hFZFjoX2cGg"      # Different video for service test
}

# Shared cache directory for all tests
SHARED_CACHE_DIR = None

def get_shared_cache_dir():
    """Get or create shared cache directory for all tests."""
    global SHARED_CACHE_DIR
    if SHARED_CACHE_DIR is None:
        import os
        # Use system configured download directory for consistent environment
        SHARED_CACHE_DIR = os.getenv("DOWNLOAD_DIR", "/app/downloads")
    return SHARED_CACHE_DIR

def retry_download(downloader, url, format="wav", max_retries=3, delay=2):
    """Retry downloading with exponential backoff."""
    for attempt in range(max_retries):
        try:
            print(f"    Download attempt {attempt + 1}/{max_retries}")
            audio_path = downloader.download_audio(url, format=format)
            if audio_path and Path(audio_path).exists():
                return audio_path
            else:
                print(f"    ⚠ Attempt {attempt + 1} failed: No file created")
        except Exception as e:
            print(f"    ⚠ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                print(f"    Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    return None

def test_youtube_transcript_download():
    """Test downloading YouTube's built-in transcripts."""
    print("Testing YouTube transcript download...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = YouTubeDownloader(output_dir=temp_dir)
        
        # Test with a video that likely has captions
        for name, url in TEST_URLS.items():
            print(f"  Testing {name}: {url}")
            
            try:
                transcripts = downloader.get_youtube_transcripts(url)
                
                if transcripts:
                    print(f"    ✓ Found transcripts in languages: {list(transcripts.keys())}")
                    
                    # Check transcript content
                    for lang, content in transcripts.items():
                        if content and len(content.strip()) > 0:
                            print(f"    ✓ {lang} transcript has content ({len(content)} chars)")
                            # Show first 100 chars as sample
                            sample = content[:100].replace('\n', ' ')
                            print(f"      Sample: {sample}...")
                        else:
                            print(f"    ⚠ {lang} transcript is empty")
                else:
                    print(f"    ⚠ No YouTube transcripts found for {name}")
                    
            except Exception as e:
                print(f"    ❌ Error getting transcripts for {name}: {e}")
    
    print("✓ YouTube transcript download test completed\n")

def test_audio_download():
    """Test downloading audio from YouTube."""
    print("Testing audio download...")
    
    cache_dir = get_shared_cache_dir()
    downloader = YouTubeDownloader(output_dir=cache_dir)
    
    # Test with shortest URL
    test_url = TEST_URLS["short_english"]
    print(f"  Testing audio download: {test_url}")
    
    try:
        # Download audio with retry
        audio_path = retry_download(downloader, test_url, format="wav")
        
        if audio_path and Path(audio_path).exists():
            file_size = Path(audio_path).stat().st_size
            print(f"    ✓ Audio downloaded to: {audio_path}")
            print(f"    ✓ File size: {file_size} bytes")
            
            # Basic validation
            if file_size > 1000:  # At least 1KB
                print("    ✓ Audio file has reasonable size")
            else:
                print("    ⚠ Audio file seems too small")
                
            # Keep file for subsequent tests (shared cache)
            print("    ✓ Audio file preserved for subsequent tests")
        else:
            print("    ❌ Audio download failed - no file created")
            
    except Exception as e:
        print(f"    ❌ Error downloading audio: {e}")
    
    print("✓ Audio download test completed\n")

def test_whisper_transcription():
    """Test Whisper transcription with actual audio."""
    print("Testing Whisper transcription...")
    
    cache_dir = get_shared_cache_dir()
    downloader = YouTubeDownloader(output_dir=cache_dir)
    
    # Use different test video to avoid rate limiting
    test_url = TEST_URLS["whisper_test"]
    print(f"  Testing transcription: {test_url}")
    
    try:
        # Download audio first with retry
        print("    Downloading audio...")
        audio_path = retry_download(downloader, test_url, format="wav")
        
        if not audio_path or not Path(audio_path).exists():
            print("    ❌ Could not download audio for transcription test")
            return
        
        print("    Initializing Whisper transcriber...")
        # Use smallest model for speed, with system defaults
        transcriber = WhisperTranscriber(
            model_size="tiny"
        )
        
        print("    Running transcription...")
        # Transcribe with basic settings
        result = transcriber.transcribe_audio(
            audio_path,
            language=None,  # Auto-detect
        )
        
        if result and 'segments' in result:
            print(f"    ✓ Transcription successful")
            print(f"    ✓ Detected language: {result.get('language', 'unknown')}")
            print(f"    ✓ Number of segments: {len(result['segments'])}")
            
            # Test formatting
            text_format = transcriber.format_transcript(result, "text")
            srt_format = transcriber.format_transcript(result, "srt")
            vtt_format = transcriber.format_transcript(result, "vtt")
            
            if text_format and len(text_format.strip()) > 0:
                print("    ✓ Text format generated")
                print(f"      Sample: {text_format[:100]}...")
            else:
                print("    ⚠ Text format is empty")
            
            if srt_format and "1\n" in srt_format:
                print("    ✓ SRT format generated")
            else:
                print("    ⚠ SRT format invalid")
            
            if vtt_format and "WEBVTT" in vtt_format:
                print("    ✓ VTT format generated")
            else:
                print("    ⚠ VTT format invalid")
        else:
            print("    ❌ Transcription failed - no segments returned")
        
        # Keep file for subsequent tests (shared cache)
        print("    ✓ Audio file preserved for subsequent tests")
        
    except Exception as e:
        print(f"    ❌ Error in transcription: {e}")
        import traceback
        traceback.print_exc()
    
    print("✓ Whisper transcription test completed\n")

def test_full_service_integration():
    """Test the full service end-to-end."""
    print("Testing full service integration...")
    
    with tempfile.TemporaryDirectory():
        # Test configuration with system defaults
        config = TranscriptionConfig(
            url=TEST_URLS["service_test"],
            model_size="tiny",  # Fastest model
            language=None,      # Auto-detect
        )
        
        progress_messages = []
        def progress_callback(msg):
            progress_messages.append(msg)
            print(f"    Progress: {msg}")
        
        try:
            print("  Running full transcription service...")
            result = transcribe_youtube_video(config, progress_callback)
            
            if result.success:
                print("    ✓ Service completed successfully")
                if result.detected_language and result.detected_language != "unknown":
                    print(f"    ✓ Detected language: {result.detected_language}")
                else:
                    print("    ⚠ Language detection failed or unknown")
                print(f"    ✓ Progress messages: {len(progress_messages)}")
                
                # Check outputs
                if result.transcript_text and len(result.transcript_text.strip()) > 0:
                    print("    ✓ Text transcript generated")
                    print(f"      Length: {len(result.transcript_text)} chars")
                else:
                    print("    ⚠ Text transcript is empty")
                
                if result.transcript_srt and "1\n" in result.transcript_srt:
                    print("    ✓ SRT transcript generated")
                else:
                    print("    ⚠ SRT transcript invalid")
                
                if result.transcript_vtt and "WEBVTT" in result.transcript_vtt:
                    print("    ✓ VTT transcript generated")
                else:
                    print("    ⚠ VTT transcript invalid")
                
                # Check YouTube transcripts
                if result.youtube_transcripts:
                    print(f"    ✓ YouTube transcripts: {list(result.youtube_transcripts.keys())}")
                else:
                    print("    ⚠ No YouTube transcripts found")
                
                # Check raw result structure
                if result.raw_result and isinstance(result.raw_result, dict):
                    print(f"    ✓ Raw result available with {len(result.raw_result)} keys")
                    # Check for expected WhisperX result structure
                    if "segments" in result.raw_result:
                        segments = result.raw_result["segments"]
                        if isinstance(segments, list) and len(segments) > 0:
                            print(f"    ✓ Raw result contains {len(segments)} segments")
                        else:
                            print("    ⚠ Raw result segments empty or invalid")
                    else:
                        print("    ⚠ Raw result missing segments")
                else:
                    print("    ⚠ Raw result missing or invalid")
                
            else:
                print(f"    ❌ Service failed: {result.error_message}")
                # Validate error result structure
                if result.error_message and len(result.error_message.strip()) > 0:
                    print("    ✓ Error message provided")
                else:
                    print("    ⚠ Error message missing or empty")
                    
                # Check that error result has expected empty values
                if (result.transcript_text == "" and 
                    result.transcript_srt == "" and 
                    result.transcript_vtt == ""):
                    print("    ✓ Error result has empty transcript fields")
                else:
                    print("    ⚠ Error result should have empty transcript fields")
                    
                if result.detected_language == "unknown":
                    print("    ✓ Error result has unknown language")
                else:
                    print("    ⚠ Error result should have unknown language")
                
        except Exception as e:
            print(f"    ❌ Error in full service test: {e}")
            import traceback
            traceback.print_exc()
    
    print("✓ Full service integration test completed\n")

def main():
    """Run all integration tests."""
    print("Starting integration tests...\n")
    print("Note: These tests use real YouTube videos and may take several minutes.\n")
    
    try:
        test_youtube_transcript_download()
        test_audio_download()
        test_whisper_transcription()
        test_full_service_integration()
        
        print("🎉 All integration tests completed!")
        print("Check the output above for any warnings or errors.")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Note: Using system download directory, no cleanup needed
        global SHARED_CACHE_DIR
        if SHARED_CACHE_DIR:
            print(f"Test files remain in system download directory: {SHARED_CACHE_DIR}")

if __name__ == "__main__":
    sys.exit(main())