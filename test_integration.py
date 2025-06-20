#!/usr/bin/env python3
"""Integration tests for core functionality."""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, '/app/src')

from transcript_extractor.core.downloader import YouTubeDownloader
from transcript_extractor.core.transcriber import WhisperTranscriber
from transcript_extractor.core.service import TranscriptionConfig, transcribe_youtube_video

# Test URLs - using short, safe videos
TEST_URLS = {
    "short_english": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll (short)
    "with_captions": "https://www.youtube.com/watch?v=jNQXAC9IVRw"   # Short video with captions
}

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
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = YouTubeDownloader(output_dir=temp_dir)
        
        # Test with shortest URL
        test_url = TEST_URLS["short_english"]
        print(f"  Testing audio download: {test_url}")
        
        try:
            # Download audio
            audio_path = downloader.download_audio(test_url, format="wav")
            
            if audio_path and Path(audio_path).exists():
                file_size = Path(audio_path).stat().st_size
                print(f"    ✓ Audio downloaded to: {audio_path}")
                print(f"    ✓ File size: {file_size} bytes")
                
                # Basic validation
                if file_size > 1000:  # At least 1KB
                    print("    ✓ Audio file has reasonable size")
                else:
                    print("    ⚠ Audio file seems too small")
                    
                # Cleanup
                downloader.cleanup(audio_path)
                if not Path(audio_path).exists():
                    print("    ✓ Cleanup successful")
                else:
                    print("    ⚠ Cleanup failed")
            else:
                print("    ❌ Audio download failed - no file created")
                
        except Exception as e:
            print(f"    ❌ Error downloading audio: {e}")
    
    print("✓ Audio download test completed\n")

def test_whisper_transcription():
    """Test Whisper transcription with actual audio."""
    print("Testing Whisper transcription...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = YouTubeDownloader(output_dir=temp_dir)
        
        # Use shortest test video
        test_url = TEST_URLS["short_english"]
        print(f"  Testing transcription: {test_url}")
        
        try:
            # Download audio first
            print("    Downloading audio...")
            audio_path = downloader.download_audio(test_url, format="wav")
            
            if not audio_path or not Path(audio_path).exists():
                print("    ❌ Could not download audio for transcription test")
                return
            
            print("    Initializing Whisper transcriber...")
            # Use smallest model for speed
            transcriber = WhisperTranscriber(
                model_size="tiny",
                device="cpu",  # Force CPU for compatibility
                compute_type="float32"
            )
            
            print("    Running transcription...")
            # Transcribe with basic settings
            result = transcriber.transcribe_audio(
                audio_path,
                language=None,  # Auto-detect
                align=False     # Skip alignment for speed
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
            
            # Cleanup
            downloader.cleanup(audio_path)
            
        except Exception as e:
            print(f"    ❌ Error in transcription: {e}")
            import traceback
            traceback.print_exc()
    
    print("✓ Whisper transcription test completed\n")

def test_full_service_integration():
    """Test the full service end-to-end."""
    print("Testing full service integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test configuration
        config = TranscriptionConfig(
            url=TEST_URLS["short_english"],
            model_size="tiny",  # Fastest model
            language=None,      # Auto-detect
            audio_format="wav",
            device="cpu",       # Force CPU
            compute_type="float32",
            align=False,        # Skip alignment for speed
            temp_dir=Path(temp_dir),
            keep_audio=True     # Keep for inspection
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
                print(f"    ✓ Detected language: {result.detected_language}")
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
                
                # Check audio file
                if result.audio_path and Path(result.audio_path).exists():
                    print(f"    ✓ Audio file preserved: {result.audio_path}")
                else:
                    print("    ⚠ Audio file not preserved")
                
            else:
                print(f"    ❌ Service failed: {result.error_message}")
                
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

if __name__ == "__main__":
    sys.exit(main())