import sys
import tempfile
from pathlib import Path
from typing import Optional

import click

from .core.downloader import YouTubeDownloader
from .core.transcriber import WhisperTranscriber


@click.command()
@click.argument('url')
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    help='Output file path. If not specified, prints to stdout.'
)
@click.option(
    '--format', '-f',
    type=click.Choice(['text', 'srt', 'vtt']),
    default='text',
    help='Output format (default: text)'
)
@click.option(
    '--model', '-m',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']),
    default='base',
    help='What model to use (default: base)'
)
@click.option(
    '--language', '-l',
    help='Language code (e.g., zh, en). Auto-detect if not specified.'
)
@click.option(
    '--audio-format',
    type=click.Choice(['wav', 'mp3', 'flac'], case_sensitive=False),
    default='wav',
    help='Audio format for download (default: wav)'
)
@click.option(
    '--keep-audio',
    is_flag=True,
    help='Keep downloaded audio file after transcription'
)
@click.option(
    '--temp-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help='Temporary directory for audio files'
)
@click.option(
    '--device',
    type=click.Choice(['cpu', 'cuda'], case_sensitive=False),
    help='Device to run transcription on (auto-detect if not specified)'
)
@click.option(
    '--compute-type',
    type=click.Choice(['float16', 'float32', 'int8'], case_sensitive=False),
    default='float16',
    help='Compute precision (default: float16)'
)
@click.option(
    '--no-align',
    is_flag=True,
    help='Skip word-level alignment for faster processing'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
def main(
    url: str,
    output: Optional[Path],
    format: str,
    model: str,
    language: Optional[str],
    audio_format: str,
    keep_audio: bool,
    temp_dir: Optional[Path],
    device: Optional[str],
    compute_type: str,
    no_align: bool,
    verbose: bool
) -> None:
    """Extract transcript from YouTube video.
    
    URL: YouTube video URL to transcribe
    """
    try:
        # Setup temporary directory
        temp_directory = temp_dir or Path(tempfile.gettempdir())
        
        if verbose:
            click.echo(f"Using temporary directory: {temp_directory}")
            click.echo(f"Model: {model}")
            click.echo(f"Language: {language or 'auto-detect'}")
            click.echo(f"Output format: {format}")
        
        # Initialize downloader
        downloader = YouTubeDownloader(output_dir=str(temp_directory))
        
        # Download audio
        if verbose:
            click.echo("Downloading audio...")
        
        audio_path = downloader.download_audio(url, format=audio_format)
        
        if verbose:
            click.echo(f"Audio downloaded to: {audio_path}")
        
        # Initialize transcriber
        transcriber = WhisperTranscriber(
            model_size=model,
            device=device,
            compute_type=compute_type
        )
        
        # Transcribe audio
        if verbose:
            click.echo("Loading model and transcribing...")
        
        result = transcriber.transcribe_audio(
            audio_path,
            language=language,
            align=not no_align
        )
        
        if verbose:
            detected_language = result.get('language', 'unknown')
            click.echo(f"Detected language: {detected_language}")
        
        # Format output
        transcript = transcriber.format_transcript(result, format_type=format)
        
        # Output result
        if output:
            output.write_text(transcript, encoding='utf-8')
            if verbose:
                click.echo(f"Transcript saved to: {output}")
        else:
            click.echo(transcript)
        
        # Cleanup audio file if requested
        if not keep_audio:
            downloader.cleanup(audio_path)
            if verbose:
                click.echo("Audio file cleaned up")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()