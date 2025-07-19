"""
Cache service for URL-to-file mapping using Valkey/Redis.
"""
import os
import hashlib
import redis
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from functools import wraps

class CacheService:
    """Service for caching URL-to-file mappings in Valkey/Redis."""
    
    def __init__(self):
        """Initialize the cache service with Valkey/Redis connection."""
        self.redis_client = redis.Redis(
            host=os.getenv('VALKEY_HOST', 'localhost'),
            port=int(os.getenv('VALKEY_PORT', '6379')),
            db=int(os.getenv('VALKEY_DB', '1')),
            password=os.getenv('VALKEY_PASSWORD') or None,
            username=os.getenv('VALKEY_USERNAME') or None,
            decode_responses=True
        )
        
        # Test connection
        try:
            self.redis_client.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Valkey: {e}")
    
    def _get_url_key(self, url: str) -> str:
        """Generate a unique key for the URL."""
        return f"url:{hashlib.md5(url.encode()).hexdigest()}"
    
    def _get_metadata_key(self, url: str) -> str:
        """Generate a metadata key for the URL."""
        return f"meta:{hashlib.md5(url.encode()).hexdigest()}"
    
    def get_cached_file(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get cached file information for a URL.
        
        Args:
            url: The video URL
            
        Returns:
            Dict with file info if cached, None otherwise
        """
        url_key = self._get_url_key(url)
        meta_key = self._get_metadata_key(url)
        
        file_path = self.redis_client.get(url_key)
        if not file_path:
            return None
        
        if not os.path.exists(file_path):
            # File was deleted, remove from cache
            self.remove_cached_file(url)
            return None
        
        metadata = self.redis_client.hgetall(meta_key)
        
        return {
            'file_path': file_path,
            'url': url,
            'metadata': metadata
        }
    
    def cache_file(self, url: str, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Cache a file for a URL.
        
        Args:
            url: The video URL
            file_path: Path to the downloaded file
            metadata: Optional metadata about the file
        """
        url_key = self._get_url_key(url)
        meta_key = self._get_metadata_key(url)
        
        # Cache the file path
        self.redis_client.set(url_key, file_path)
        
        # Cache metadata if provided
        if metadata:
            self.redis_client.hset(meta_key, mapping=metadata)
        
        self.redis_client.expire(url_key, 30 * 24 * 60 * 60)
        self.redis_client.expire(meta_key, 30 * 24 * 60 * 60)
    
    def remove_cached_file(self, url: str) -> None:
        """
        Remove a cached file entry.
        
        Args:
            url: The video URL
        """
        url_key = self._get_url_key(url)
        meta_key = self._get_metadata_key(url)
        
        self.redis_client.delete(url_key)
        self.redis_client.delete(meta_key)
    


def with_cache(cache_service: Optional[CacheService] = None, progress_callback: Optional[Callable[[str], None]] = None):
    """
    Decorator for caching download_audio method results based on URL parameter.
    
    Args:
        cache_service: Optional cache service instance
        progress_callback: Optional progress callback function
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(url: str, format: str = "wav"):
            print(f"DEBUG: wrapper called with url={url}, format={format}")
            
            if progress_callback:
                progress_callback("Checking cache for URL...")
            
            cached_result = cache_service.get_cached_file(url)
            if cached_result:
                if progress_callback:
                    progress_callback(f"Found cached file: {cached_result['file_path']}")
                
                return Path(cached_result['file_path'])
            
            if progress_callback:
                progress_callback("No cached file found, downloading...")
            
            result = func(url, format)
            
            # Cache the result
            if result and isinstance(result, Path):
                try:
                    metadata = {
                        'original_url': url,
                        'audio_format': format
                    }
                    cache_service.cache_file(url, str(result), metadata)
                    if progress_callback:
                        progress_callback(f"Cached file for URL: {url}")
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Warning: Failed to cache file: {e}")
            
            return result
        
        return wrapper
    return decorator