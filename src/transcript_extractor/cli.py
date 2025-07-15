import sys
from pathlib import Path
from typing import Optional

import click

from .core.service import TranscriptionConfig, transcribe_youtube_video


@click.command()
@click.argument('url')
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
        # Create progress callback for verbose output
        def progress_callback(message: str) -> None:
            if verbose:
                click.echo(message)
        
        # Create configuration
        config = TranscriptionConfig(
            url=url,
            model_size=model,
            language=language,
            audio_format=audio_format,
            device=device,
            compute_type=compute_type,
            align=not no_align,
            temp_dir=temp_dir,
            keep_audio=keep_audio
        )
        
        if verbose:
            click.echo(f"Output format: {format}")
        
        # Run transcription
        result = transcribe_youtube_video(config, progress_callback)
        
        # Check for errors
        if not result.success:
            click.echo(f"Error: {result.error_message}", err=True)
            sys.exit(1)
        
        # Get transcript in requested format
        if format == "text":
            transcript = result.transcript_text
        elif format == "srt":
            transcript = result.transcript_srt
        elif format == "vtt":
            transcript = result.transcript_vtt
        else:
            transcript = result.transcript_text
        
        # Output result
        click.echo(transcript)
        
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()