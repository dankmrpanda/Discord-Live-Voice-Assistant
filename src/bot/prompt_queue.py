"""Queue utilities for /ask prompt management."""

from collections import deque
from typing import Deque, List, Optional, Tuple


PromptItem = Tuple[str, int]


class PromptQueue:
    """FIFO queue for text prompts with safe snapshot support."""

    DEFAULT_MAX_SIZE = 50

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE) -> None:
        self._items: Deque[PromptItem] = deque()
        self._max_size = max(1, max_size)

    def enqueue(self, prompt: str, user_id: int) -> int:
        """Add a prompt and return its 1-indexed queue position.

        Returns -1 if the queue is full.
        """
        if len(self._items) >= self._max_size:
            return -1
        self._items.append((prompt, user_id))
        return len(self._items)

    def is_full(self) -> bool:
        """Return True when the queue has reached its max size."""
        return len(self._items) >= self._max_size

    def dequeue(self) -> Optional[PromptItem]:
        """Pop the next prompt, or None if the queue is empty."""
        if not self._items:
            return None
        return self._items.popleft()

    def clear(self) -> int:
        """Clear all queued prompts and return how many were removed."""
        removed = len(self._items)
        self._items.clear()
        return removed

    def size(self) -> int:
        """Return queue size."""
        return len(self._items)

    def is_empty(self) -> bool:
        """Return True when queue has no items."""
        return not self._items

    def snapshot(self) -> List[PromptItem]:
        """Return a shallow copy of queued items in queue order."""
        return list(self._items)
