# models/cache_manager.py
import logging
import threading

from cachetools import LRUCache


class CacheManager:
    def __init__(self, cache_size=256, lock_timeout=5):
        """
        Initialize a cache manager with an LRU cache and handle potential deadlocks.
        :param cache_size: Maximum number of items to store in the cache.
        :param lock_timeout: Maximum time to wait for acquiring the lock before logging a warning.
        """
        self.lock = threading.Lock()
        self.lock_timeout = lock_timeout
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size = cache_size
        self.cache = LRUCache(maxsize=cache_size)
        logging.info(f"CacheManager initialized with cache size {cache_size}")

    def get(self, key):
        """Retrieve item from cache if it exists, otherwise return None."""
        acquired = self.lock.acquire(timeout=self.lock_timeout)
        if not acquired:
            logging.warning(f"Potential deadlock detected while trying to acquire lock in 'get' for key: {key}")
            return None

        try:
            item = self.cache.get(key)
            if item is not None:
                self.cache_hits += 1
                logging.debug(f"Cache hit for key: {key}. Total hits: {self.cache_hits}")
            else:
                self.cache_misses += 1
                logging.debug(f"Cache miss for key: {key}. Total misses: {self.cache_misses}")
            return item
        finally:
            self.lock.release()

    def set(self, key, value):
        """Store an item in the cache, evicting the least recently used item if the cache is full."""
        acquired = self.lock.acquire(timeout=self.lock_timeout)
        if not acquired:
            logging.warning(f"Potential deadlock detected while trying to acquire lock in 'set' for key: {key}")
            return

        try:
            if len(self.cache) >= self.cache_size:
                # Manual eviction: Remove the least recently used item
                evicted_key, evicted_value = self.cache.popitem(last=False)
                logging.debug(f"Cache is full. Evicting key: {evicted_key}")

            # Now add the new item
            self.cache[key] = value
            logging.debug(f"Item cached with key: {key}. Current cache size: {len(self.cache)}")
        finally:
            self.lock.release()

    def clear(self):
        """Clear all items from the cache."""
        acquired = self.lock.acquire(timeout=self.lock_timeout)
        if not acquired:
            logging.warning("Potential deadlock detected while trying to acquire lock in 'clear'")
            return

        try:
            self.cache.clear()
            logging.info("Cache cleared")
        finally:
            self.lock.release()

    def get_cache_statistics(self):
        """Return cache statistics."""
        acquired = self.lock.acquire(timeout=self.lock_timeout)
        if not acquired:
            logging.warning("Potential deadlock detected while trying to acquire lock in 'get_cache_statistics'")
            return {}

        try:
            stats = {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'current_size': len(self.cache),
                'max_size': self.cache_size
            }
            logging.debug(f"Cache statistics: {stats}")
            return stats
        finally:
            self.lock.release()
