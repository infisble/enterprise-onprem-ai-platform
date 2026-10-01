"""Redis is used for what it is good at: caching, short-lived state and job coordination.

It is deliberately NOT the product search engine (see ADR-003). When REDIS_URL is empty an
in-process implementation with the same interface is used.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any, Protocol

from packages.core.config import get_settings


class KeyValueStore(Protocol):
    backend: str

    def get_json(self, key: str) -> Any | None: ...
    def set_json(self, key: str, value: Any, ttl_s: int = 300) -> None: ...
    def push(self, queue: str, value: dict) -> None: ...
    def pop(self, queue: str, timeout_s: int = 1) -> dict | None: ...
    def queue_size(self, queue: str) -> int: ...
    def set_if_absent(self, key: str, value: Any, ttl_s: int) -> bool: ...
    def ping(self) -> bool: ...


class InMemoryStore:
    backend = "in-memory"

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, str]] = {}
        self._queues: dict[str, deque[str]] = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def get_json(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            expires, raw = item
            if expires < time.time():
                self._data.pop(key, None)
                return None
            return json.loads(raw)

    def set_json(self, key: str, value: Any, ttl_s: int = 300) -> None:
        with self._lock:
            self._data[key] = (time.time() + ttl_s, json.dumps(value, default=str))

    def set_if_absent(self, key: str, value: Any, ttl_s: int) -> bool:
        with self._lock:
            item = self._data.get(key)
            if item and item[0] >= time.time():
                return False
            self._data[key] = (time.time() + ttl_s, json.dumps(value, default=str))
            return True

    def push(self, queue: str, value: dict) -> None:
        with self._cv:
            self._queues.setdefault(queue, deque()).append(json.dumps(value))
            self._cv.notify()

    def pop(self, queue: str, timeout_s: int = 1) -> dict | None:
        with self._cv:
            q = self._queues.setdefault(queue, deque())
            if not q:
                self._cv.wait(timeout_s)
            return json.loads(q.popleft()) if q else None

    def queue_size(self, queue: str) -> int:
        with self._lock:
            return len(self._queues.get(queue, ()))

    def ping(self) -> bool:
        return True


class RedisStore:
    backend = "redis"

    def __init__(self, url: str) -> None:
        import redis

        self._r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=5)

    def get_json(self, key: str) -> Any | None:
        raw = self._r.get(key)
        return json.loads(raw) if raw else None

    def set_json(self, key: str, value: Any, ttl_s: int = 300) -> None:
        self._r.set(key, json.dumps(value, default=str), ex=ttl_s)

    def set_if_absent(self, key: str, value: Any, ttl_s: int) -> bool:
        return bool(self._r.set(key, json.dumps(value, default=str), ex=ttl_s, nx=True))  # atomic SET NX

    def push(self, queue: str, value: dict) -> None:
        self._r.lpush(queue, json.dumps(value))

    def pop(self, queue: str, timeout_s: int = 1) -> dict | None:
        item = self._r.brpop([queue], timeout=timeout_s)
        return json.loads(item[1]) if item else None

    def queue_size(self, queue: str) -> int:
        return int(self._r.llen(queue))

    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except Exception:
            return False


_store: KeyValueStore | None = None


def get_store() -> KeyValueStore:
    global _store
    if _store is None:
        url = get_settings().redis_url
        _store = RedisStore(url) if url else InMemoryStore()
    return _store


def reset_store() -> None:
    global _store
    _store = None
