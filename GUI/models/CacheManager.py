# GUI/models/CacheManager.py

import logging
import threading
from typing import Any, Optional, Dict

from cachetools import LRUCache


class SingletonMeta(type):
    """
    A thread-safe implementation of Singleton.
    """
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """
        Controls the instantiation of the Singleton class.
        """
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class CacheManager(metaclass=SingletonMeta):
    """
    A thread-safe Singleton CacheManager using cachetools' LRUCache.
    """

    def __init__(self, cache_size: int = 1024):
        """
        Initialize the CacheManager with an LRU cache.

        :param cache_size: Maximum number of items to store in the cache.
        """
        self._lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_size = cache_size
        self._cache = LRUCache(maxsize=cache_size)
        logging.info(f"CacheManager initialized with cache size {cache_size}")

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve an item from the cache.

        :param key: The key of the item to retrieve.
        :return: The cached item if present, else None.
        """
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

    def set(self, key: Any, value: Any) -> None:
        """
        Store an item in the cache.

        :param key: The key under which to store the item.
        :param value: The item to store.
        """
        with self._lock:
            try:
                self._cache[key] = value  # LRUCache handles eviction automatically
                logging.debug(f"Item cached with key: {key}. Current cache size: {len(self._cache)}")
            except Exception as e:
                logging.error(f"Exception in 'set' method for key {key}: {e}")
                raise

    def clear(self) -> None:
        """
        Clear all items from the cache.
        """
        with self._lock:
            try:
                self._cache.clear()
                self._cache_hits = 0
                self._cache_misses = 0
                logging.info("Cache cleared")
            except Exception as e:
                logging.error(f"Exception in 'clear' method: {e}")
                raise

    def get_cache_statistics(self) -> Dict[str, int]:
        """
        Retrieve cache statistics.

        :return: A dictionary containing hits, misses, current size, and max size.
        """
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
