import logging
import threading
from typing import Any, Optional, Dict

from cachetools import LRUCache


class SingletonMeta(type):
    """
    Thread-safe Singleton metaclass.
    Ensures only one instance of a class can exist at a time.
    """
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """
        Controls the instantiation of the Singleton class.
        """
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:  # Double-check after acquiring lock
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class CacheManager(metaclass=SingletonMeta):
    """
    Thread-safe, Singleton-based cache manager using an LRU cache under the hood.
    """

    def __init__(self, cache_size: int = 2048):
        """
        Initialize the CacheManager with an LRU cache of the given size.

        :param cache_size: Maximum number of items to store in the cache.
        """
        self._lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_size = cache_size
        self._cache = LRUCache(maxsize=cache_size)
        logging.info(f"CacheManager initialized with cache size: {cache_size}")

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve an item from the cache by its key.

        :param key: Key of the item to retrieve.
        :return: The cached item if present, otherwise None.
        """
        with self._lock:
            item = self._cache.get(key)
            if item is not None:
                self._cache_hits += 1
                logging.debug(f"Cache hit for key={key}. Hits={self._cache_hits}")
            else:
                self._cache_misses += 1
                logging.debug(f"Cache miss for key={key}. Misses={self._cache_misses}")
            return item

    def set(self, key: Any, value: Any) -> None:
        """
        Store an item in the cache under the given key.

        :param key: Key to store the item under.
        :param value: The item to store in the cache.
        """
        with self._lock:
            self._cache[key] = value  # LRUCache handles eviction automatically
            logging.debug(f"Cached item under key={key}. Current size={len(self._cache)}")

    def clear(self) -> None:
        """
        Clear the entire cache, resetting hits and misses.
        """
        with self._lock:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logging.info("Cache cleared successfully.")

    def get_cache_statistics(self) -> Dict[str, int]:
        """
        Returns cache usage statistics.

        :return: Dictionary with keys 'hits', 'misses', 'current_size', and 'max_size'.
        """
        with self._lock:
            stats = {
                'hits': self._cache_hits,
                'misses': self._cache_misses,
                'current_size': len(self._cache),
                'max_size': self._cache_size,
            }
            logging.debug(f"Cache statistics: {stats}")
            return stats
