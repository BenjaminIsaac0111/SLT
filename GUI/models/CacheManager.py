import logging
import threading

from cachetools import LRUCache


class CacheManager:
    def __init__(self, cache_size=256):
        """
        Initialize a cache manager with an LRU cache.
        :param cache_size: Maximum number of items to store in the cache.
        """
        self._lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_size = cache_size
        self._cache = LRUCache(maxsize=cache_size)
        logging.info(f"CacheManager initialized with cache size {cache_size}")

    def get(self, key):
        """Retrieve item from cache if it exists, otherwise return None."""
        with self._lock:
            try:
                item = self._cache.get(key)
                if item is not None:
                    self._cache_hits += 1
                    logging.debug(f"Cache hit for key: {key}. Total hits: {self._cache_hits}")
                else:
                    self._cache_misses += 1
                    logging.debug(f"Cache miss for key: {key}. Total misses: {self._cache_misses}")
                return item
            except Exception as e:
                logging.error(f"Exception in 'get' method for key {key}: {e}")
                raise

    def set(self, key, value):
        """Store an item in the cache."""
        with self._lock:
            try:
                self._cache[key] = value  # LRUCache handles eviction automatically
                logging.debug(f"Item cached with key: {key}. Current cache size: {len(self._cache)}")
            except Exception as e:
                logging.error(f"Exception in 'set' method for key {key}: {e}")
                raise

    def clear(self):
        """Clear all items from the cache."""
        with self._lock:
            try:
                self._cache.clear()
                logging.info("Cache cleared")
            except Exception as e:
                logging.error(f"Exception in 'clear' method: {e}")
                raise

    def get_cache_statistics(self):
        """Return cache statistics."""
        with self._lock:
            try:
                stats = {
                    'hits': self._cache_hits,
                    'misses': self._cache_misses,
                    'current_size': len(self._cache),
                    'max_size': self._cache_size
                }
                logging.debug(f"Cache statistics: {stats}")
                return stats
            except Exception as e:
                logging.error(f"Exception in 'get_cache_statistics' method: {e}")
                raise
