import tempfile
from pathlib import Path
from typing import Optional, Dict
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
                
                ydl_opts['outtmpl'] = str(self.output_dir / f"{safe_title}.%(ext)s")
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                    ydl_download.download([url])
                
                return audio_path
                
        except Exception as e:
            raise Exception(f"Failed to download audio from {url}: {str(e)}")
    
    def get_youtube_transcripts(self, url: str) -> Dict[str, str]:
        """Get all available YouTube transcripts/subtitles.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary mapping language codes to transcript content
        """
        try:
            ydl_opts = {
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})
                
                transcripts = {}
                
                for lang, sub_list in subtitles.items():
                    content = self._download_subtitle_content(sub_list)
                    if content:
                        transcripts[lang] = content
                
                for lang, sub_list in automatic_captions.items():
                    if lang not in transcripts:  # Don't override manual subtitles
                        content = self._download_subtitle_content(sub_list)
                        if content:
                            transcripts[f"{lang}-auto"] = content
                
                return transcripts
                
        except Exception:
            return {}
    
    def _download_subtitle_content(self, sub_list: list) -> str:
        """Download and parse subtitle content from subtitle info list.
        
        Args:
            sub_list: List of subtitle format info from yt-dlp
            
        Returns:
            Parsed transcript text, or empty string if failed
        """
        try:
            # Prefer VTT format, fallback to other formats
            for sub in sub_list:
                if sub.get('ext') == 'vtt':
                    return self._fetch_and_parse_subtitle(sub['url'])
            
            for sub in sub_list:
                if 'url' in sub:
                    return self._fetch_and_parse_subtitle(sub['url'])
            
            return ""
            
        except Exception:
            return ""
    
    def _fetch_and_parse_subtitle(self, subtitle_url: str) -> str:
        """Fetch subtitle from URL and parse to plain text.
        
        Args:
            subtitle_url: URL to subtitle file
            
        Returns:
            Plain text transcript
        """
        try:
            import requests
            response = requests.get(subtitle_url, timeout=10)
            if response.status_code == 200:
                return self._parse_subtitle_content(response.text)
            return ""
        except Exception:
            return ""
    
    def _parse_subtitle_content(self, content: str) -> str:
        """Parse subtitle content (VTT/SRT) to plain text.
        
        Args:
            content: Raw subtitle content
            
        Returns:
            Plain text transcript
        """
        import re
        
        lines = content.split('\n')
        transcript_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip headers, timestamps, and empty lines
            if (line and 
                not line.startswith('WEBVTT') and 
                not line.startswith('NOTE') and 
                '-->' not in line and 
                not line.isdigit() and
                not re.match(r'^\d+$', line)):  # Skip SRT sequence numbers
                
                clean_line = re.sub(r'<[^>]+>', '', line)
                clean_line = re.sub(r'&[a-zA-Z]+;', '', clean_line)
                clean_line = clean_line.strip()
                
                if clean_line:
                    transcript_lines.append(clean_line)
        
        return '\n'.join(transcript_lines)

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

