import pytest


def test_basic():
    """Simple put/get/delete works."""
    with mock.patch("time.monotonic", return_value=0):
        c = TTLCache(2, 10)
        assert c.size() == 0
        c.put("a", 42)
        assert c.get("a") == 42
        assert c.size() == 1


def test_ttl_expiration():
    """Entry is removed when its TTL expires."""
    with mock.patch("time.monotonic", return_value=5):
        c = TTLCache(2, 3)          # ttl = 3 seconds
        c.put("x", "val")
        assert c.get("x") == "val"
        # advance time beyond TTL
        with mock.patch("time.monotonic", return_value=8):
            assert c.get("x") is None
            assert c.size() == 0


def test_eviction_lru():
    """When capacity is exceeded the LRU entry is evicted."""
    with mock.patch("time.monotonic", return_value=0):
        c = TTLCache(2, 10)
        c.put("a", 1)
        c.put("b", 2)               # cache full
        assert c.get("a") == 1
        assert c.get("b") == 2

        # delete a (LRU), then put c – b should be evicted
        c.delete("a")
        c.put("c", 3)
        assert c.get("b") is None
        assert c.get("c") == 3


def test_multiple_ttl():
    """Different TTLs cause different expirations."""
    with mock.patch("time.monotonic", return_value=0):
        c = TTLCache(2, 5)          # default ttl = 5
        c.put("a", "v1")
        c.put("b", "v2")            # both present
        assert c.size() == 2

    with mock.patch("time.monotonic", return_value=6):
        assert c.get("a") is None   # expired first
        assert c.get("b") is None   # also expired (same TTL)
        assert c.size() == 0


def test_delete_nonexistent():
    """Deleting a missing key returns False and does nothing."""
    with mock.patch("time.monotonic", return_value=0):
        c = TTLCache(2, 10)
        assert not c.delete("missing")
        assert c.size() == 0


def test_size_updates():
    """size() reflects the current number of entries."""
    with mock.patch("time.monotonic", return_value=0):
        c = TTLCache(3, 10)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)

        assert c.size() == 3

        # delete a
        c.delete("a")
        assert c.size() == 2


# -------------------------------------------------------------------------- #
# Run the tests when executed directly -------------------------------------- #
# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    unittest.main()
