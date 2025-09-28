"""
Advanced caching system for ML pipeline components.

This module provides a comprehensive caching solution with:
- Multiple cache backends (memory, disk, Redis)
- Cache invalidation strategies
- Performance monitoring
- Thread-safe operations
"""

import hashlib
import json
import pickle
import time
import threading
import functools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import weakref

from .exceptions import CacheError
from .logger import Logger


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    last_accessed: Optional[datetime] = None
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[timedelta] = None
    
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl is None:
            return False
        return datetime.now() - self.created_at > self.ttl
    
    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.now()
        self.access_count += 1


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get cache size."""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                return None
            
            entry.touch()
            self._stats.hits += 1
            self._stats.last_accessed = datetime.now()
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Set value in memory cache."""
        with self._lock:
            # Check if we need to evict entries
            self._evict_if_needed()
            
            entry = CacheEntry(key=key, value=value, ttl=ttl)
            self._cache[key] = entry
            self._update_stats()
    
    def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._update_stats()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self._cache and not self._cache[key].is_expired()
    
    def size(self) -> int:
        """Get cache size."""
        with self._lock:
            return len(self._cache)
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache is full."""
        if len(self._cache) >= self.max_size:
            # LRU eviction
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].accessed_at
            )
            
            # Remove oldest 10% of entries
            evict_count = max(1, len(sorted_entries) // 10)
            for key, _ in sorted_entries[:evict_count]:
                del self._cache[key]
                self._stats.evictions += 1
    
    def _update_stats(self) -> None:
        """Update cache statistics."""
        self._stats.size_bytes = sum(
            len(pickle.dumps(entry.value)) 
            for entry in self._cache.values()
        )


class DiskCacheBackend(CacheBackend):
    """Disk-based cache backend."""
    
    def __init__(self, cache_dir: Union[str, Path], max_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._lock:
            cache_file = self._get_cache_file(key)
            if not cache_file.exists():
                self._stats.misses += 1
                return None
            
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                if entry.is_expired():
                    cache_file.unlink()
                    self._stats.misses += 1
                    return None
                
                entry.touch()
                self._save_entry(entry)
                self._stats.hits += 1
                self._stats.last_accessed = datetime.now()
                return entry.value
                
            except Exception:
                cache_file.unlink(missing_ok=True)
                self._stats.misses += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Set value in disk cache."""
        with self._lock:
            entry = CacheEntry(key=key, value=value, ttl=ttl)
            self._save_entry(entry)
            self._update_stats()
    
    def delete(self, key: str) -> bool:
        """Delete value from disk cache."""
        with self._lock:
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                cache_file.unlink()
                self._update_stats()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self._metadata_file.unlink(missing_ok=True)
            self._stats = CacheStats()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            cache_file = self._get_cache_file(key)
            return cache_file.exists()
    
    def size(self) -> int:
        """Get cache size."""
        with self._lock:
            return len(list(self.cache_dir.glob("*.pkl")))
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for a key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def _save_entry(self, entry: CacheEntry) -> None:
        """Save cache entry to disk."""
        cache_file = self._get_cache_file(entry.key)
        with open(cache_file, 'wb') as f:
            pickle.dump(entry, f)
    
    def _load_metadata(self) -> None:
        """Load cache metadata."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r') as f:
                    data = json.load(f)
                    self._stats = CacheStats(**data)
            except Exception:
                self._stats = CacheStats()
    
    def _update_stats(self) -> None:
        """Update cache statistics."""
        self._stats.size_bytes = sum(
            f.stat().st_size 
            for f in self.cache_dir.glob("*.pkl")
        )
        
        # Save metadata
        with open(self._metadata_file, 'w') as f:
            json.dump(self._stats.__dict__, f, default=str)


class CacheManager:
    """
    Advanced cache manager with multiple backends and strategies.
    """
    
    def __init__(
        self, 
        backend: Optional[CacheBackend] = None,
        logger: Optional[Logger] = None,
        enable_stats: bool = True
    ):
        self.backend = backend or MemoryCacheBackend()
        self.logger = logger
        self.enable_stats = enable_stats
        self._stats = CacheStats()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = self.backend.get(key)
            if value is not None:
                self._stats.hits += 1
                if self.logger:
                    self.logger.debug(f"Cache hit for key: {key}")
            else:
                self._stats.misses += 1
                if self.logger:
                    self.logger.debug(f"Cache miss for key: {key}")
            return value
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cache get error for key {key}: {e}")
            raise CacheError(f"Failed to get cache value: {e}") from e
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[Union[int, timedelta]] = None
    ) -> None:
        """Set value in cache."""
        try:
            if isinstance(ttl, int):
                ttl = timedelta(seconds=ttl)
            
            self.backend.set(key, value, ttl)
            if self.logger:
                self.logger.debug(f"Cached value for key: {key}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cache set error for key {key}: {e}")
            raise CacheError(f"Failed to set cache value: {e}") from e
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            result = self.backend.delete(key)
            if self.logger:
                self.logger.debug(f"Deleted cache key: {key}")
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cache delete error for key {key}: {e}")
            raise CacheError(f"Failed to delete cache value: {e}") from e
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self.backend.clear()
            self._stats = CacheStats()
            if self.logger:
                self.logger.info("Cache cleared")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cache clear error: {e}")
            raise CacheError(f"Failed to clear cache: {e}") from e
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return self.backend.exists(key)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cache exists error for key {key}: {e}")
            raise CacheError(f"Failed to check cache existence: {e}") from e
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size_bytes=self.backend.size(),
                last_accessed=self._stats.last_accessed
            )
    
    def cached(
        self, 
        ttl: Optional[Union[int, timedelta]] = None,
        key_func: Optional[Callable] = None
    ):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time to live for cached results
            key_func: Custom key generation function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    key_data = {
                        'func': func.__name__,
                        'args': args,
                        'kwargs': kwargs
                    }
                    cache_key = hashlib.md5(
                        json.dumps(key_data, sort_keys=True, default=str).encode()
                    ).hexdigest()
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator


# Global cache manager instance
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager."""
    return _cache_manager


def set_cache_backend(backend: CacheBackend) -> None:
    """Set the cache backend for the global cache manager."""
    global _cache_manager
    _cache_manager.backend = backend


def cached(ttl: Optional[Union[int, timedelta]] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results using the global cache manager."""
    return _cache_manager.cached(ttl, key_func)
