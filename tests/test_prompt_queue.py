"""Tests for prompt queue behavior."""

import importlib.util
import unittest
from pathlib import Path


def _load_prompt_queue_class():
    module_path = Path(__file__).resolve().parents[1] / "src" / "bot" / "prompt_queue.py"
    spec = importlib.util.spec_from_file_location("prompt_queue_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.PromptQueue


PromptQueue = _load_prompt_queue_class()


class PromptQueueTests(unittest.TestCase):
    def test_enqueue_dequeue_fifo_order(self) -> None:
        queue = PromptQueue()

        pos1 = queue.enqueue("first", 1)
        pos2 = queue.enqueue("second", 2)

        self.assertEqual(pos1, 1)
        self.assertEqual(pos2, 2)
        self.assertEqual(queue.size(), 2)

        self.assertEqual(queue.dequeue(), ("first", 1))
        self.assertEqual(queue.dequeue(), ("second", 2))
        self.assertIsNone(queue.dequeue())
        self.assertTrue(queue.is_empty())

    def test_snapshot_and_clear(self) -> None:
        queue = PromptQueue()
        queue.enqueue("one", 11)
        queue.enqueue("two", 22)

        snapshot = queue.snapshot()
        self.assertEqual(snapshot, [("one", 11), ("two", 22)])

        removed = queue.clear()
        self.assertEqual(removed, 2)
        self.assertEqual(queue.size(), 0)
        self.assertEqual(queue.snapshot(), [])

    # -- max_size tests --

    def test_max_size_default(self) -> None:
        queue = PromptQueue()
        self.assertEqual(queue._max_size, PromptQueue.DEFAULT_MAX_SIZE)

    def test_max_size_custom(self) -> None:
        queue = PromptQueue(max_size=3)
        self.assertEqual(queue._max_size, 3)

    def test_max_size_floor_at_one(self) -> None:
        queue = PromptQueue(max_size=0)
        self.assertEqual(queue._max_size, 1)

    def test_enqueue_returns_negative_when_full(self) -> None:
        queue = PromptQueue(max_size=2)
        self.assertEqual(queue.enqueue("a", 1), 1)
        self.assertEqual(queue.enqueue("b", 2), 2)
        self.assertEqual(queue.enqueue("c", 3), -1)
        # queue should still have exactly 2 items
        self.assertEqual(queue.size(), 2)

    def test_is_full(self) -> None:
        queue = PromptQueue(max_size=1)
        self.assertFalse(queue.is_full())
        queue.enqueue("x", 1)
        self.assertTrue(queue.is_full())
        queue.dequeue()
        self.assertFalse(queue.is_full())

    def test_dequeue_after_full_allows_enqueue(self) -> None:
        queue = PromptQueue(max_size=1)
        queue.enqueue("first", 1)
        self.assertEqual(queue.enqueue("second", 2), -1)
        queue.dequeue()  # free a slot
        self.assertEqual(queue.enqueue("second", 2), 1)


if __name__ == "__main__":
    unittest.main()
