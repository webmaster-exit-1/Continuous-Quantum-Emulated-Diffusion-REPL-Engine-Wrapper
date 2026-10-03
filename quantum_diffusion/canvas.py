"""A bounded, thread-safe text canvas for continuous code context."""

from __future__ import annotations

from collections import deque
from threading import RLock


class ContinuousCanvas:
    """Keep the newest text within a fixed character budget."""

    def __init__(self, max_chars: int = 32_768) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars
        self._chunks: deque[str] = deque()
        self._length = 0
        self._version = 0
        self._lock = RLock()

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def text(self) -> str:
        with self._lock:
            return "".join(self._chunks)

    def append(self, text: str) -> str:
        """Append text, evicting its oldest prefix if the budget is exceeded."""
        if not isinstance(text, str):
            raise TypeError("canvas updates must be strings")
        with self._lock:
            if len(text) >= self.max_chars:
                self._chunks.clear()
                self._chunks.append(text[-self.max_chars :])
                self._length = self.max_chars
            elif text:
                self._chunks.append(text)
                self._length += len(text)
                while self._length > self.max_chars:
                    excess = self._length - self.max_chars
                    oldest = self._chunks.popleft()
                    if len(oldest) > excess:
                        self._chunks.appendleft(oldest[excess:])
                        self._length -= excess
                    else:
                        self._length -= len(oldest)
            if text:
                self._version += 1
            return "".join(self._chunks)

    def replace(self, text: str) -> str:
        """Replace the canvas while preserving the fixed-size suffix policy."""
        if not isinstance(text, str):
            raise TypeError("canvas updates must be strings")
        with self._lock:
            self._chunks.clear()
            suffix = text[-self.max_chars :]
            if suffix:
                self._chunks.append(suffix)
            self._length = len(suffix)
            self._version += 1
            return suffix
