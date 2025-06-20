import os
import tempfile
from pathlib import Path
from typing import Optional
import yt_dlp


class YouTubeDownloader:
    """YouTube audio downloader using yt-dlp."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded audio files. 
                       If None, use system temp directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_audio(self, url: str, format: str = "wav") -> Path:
        """Download audio from YouTube URL.
        
        Args:
            url: YouTube video URL
            format: Audio format (wav, mp3, etc.)
            
        Returns:
            Path to downloaded audio file
            
        Raises:
            Exception: If download fails
        """
        output_path = self.output_dir / f"%(title)s.%(ext)s"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': format,
                'preferredquality': '192',
            }],
            'outtmpl': str(output_path),
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'unknown')
                
                # Clean title for filename
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                audio_path = self.output_dir / f"{safe_title}.{format}"
                
                # Update template with cleaned title
                ydl_opts['outtmpl'] = str(self.output_dir / f"{safe_title}.%(ext)s")
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                    ydl_download.download([url])
                
                return audio_path
                
        except Exception as e:
            raise Exception(f"Failed to download audio from {url}: {str(e)}")
    
    def cleanup(self, file_path: Path) -> None:
        """Remove downloaded file.
        
        Args:
            file_path: Path to file to remove
        """
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass  # Ignore cleanup errors