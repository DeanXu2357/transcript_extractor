import sys
from typing import Optional

import click

from .core.service import TranscriptionConfig, TranscriptionService
from .core.constants import ALL_MODELS


@click.command()
@click.argument("url")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "srt", "vtt"]),
    default="text",
    help="Output format (default: text)",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(ALL_MODELS),
    default="base",
    help="What model to use (default: base)",
)
@click.option(
    "--language",
    "-l",
    help="Language code (e.g., zh, en). Auto-detect if not specified.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    help="Device to run transcription on (auto-detect if not specified)",
)
@click.option(
    "--compute-type",
    type=click.Choice(["float16", "float32", "int8"], case_sensitive=False),
    default="float16",
    help="Compute precision (default: float16)",
)
@click.option(
    "--diarize",
    is_flag=True,
    help="Enable speaker diarization (requires HF_TOKEN environment variable)",
)
@click.option(
    "--num-speakers",
    type=int,
    help="Number of speakers (if known, improves diarization accuracy)",
)
@click.option("--min-speakers", type=int, help="Minimum number of speakers")
@click.option("--max-speakers", type=int, help="Maximum number of speakers")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def main(
    url: str,
    format: str,
    model: str,
    language: Optional[str],
    device: Optional[str],
    compute_type: str,
    diarize: bool,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    verbose: bool,
) -> None:
    """Extract transcript from YouTube video.

    URL: YouTube video URL to transcribe
    """
    try:

        def progress_callback(message: str) -> None:
            if verbose:
                click.echo(message)

        config = TranscriptionConfig(
            url=url,
            model_name=model,
            language=language,
            device=device,
            compute_type=compute_type,
            diarize=diarize,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        if verbose:
            click.echo(f"Output format: {format}")

        service = TranscriptionService(device=device, compute_type=compute_type)
        result = service.transcribe_youtube_video(config, progress_callback)

        if not result.success:
            click.echo(f"Error: {result.error_message}", err=True)
            sys.exit(1)

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


if __name__ == "__main__":
    main()
