import pytest
from unittest.mock import patch
from ttl_cache_impl import TTLCache  # Assuming the code above is saved as ttl_cache_impl.py

# Helper to create a cache instance
def create_cache(capacity=2, default_ttl=10.0):
    return TTLCache(capacity=capacity, default_ttl=default_ttl)

class TestTTLCache:

    def test_basic_get_put(self):
        """Test basic insertion and retrieval."""
        cache = create_cache()
        cache.put("key1", "value1")
        
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None

    def test_capacity_eviction_lru_order(self):
        """Test that LRU eviction happens correctly when capacity is reached."""
        cache = create_cache(capacity=2, default_ttl=100.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        
        # Access 'a' to make it recently used
        cache.get("a")
        
        # Add 'c', should evict 'b' (least recently used)
        cache.put("c", 3)
        
        assert cache.get("a") == 1
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == 3

    def test_ttl_expiry(self):
        """Test that items expire after default_ttl."""
        with patch('time.monotonic', side_effect=[0.0, 0.0, 11.0]) as mock_time:
            cache = create_cache(capacity=2, default_ttl=10.0)
            
            # t=0: Put item
            cache.put("key1", "value1")
            assert cache.get("key1") == "value1"
            
            # t=11: Item should be expired
            assert cache.get("key1") is None

    def test_custom_per_key_ttl(self):
        """Test that custom TTL overrides default_ttl."""
        with patch('time.monotonic', side_effect=[0.0, 0.0, 5.0, 5.0, 15.0]) as mock_time:
            cache = create_cache(capacity=2, default_ttl=10.0)
            
            # t=0: Put with custom TTL of 5s
            cache.put("short", "val", ttl=5.0)
            # t=0: Put with default TTL of 10s
            cache.put("long", "val", ttl=None)
            
            # t=5: 'short' expires, 'long' still valid
            assert cache.get("short") is None
            assert cache.get("long") == "val"
            
            # t=15: 'long' also expires
            assert cache.get("long") is None

    def test_delete(self):
        """Test explicit deletion of keys."""
        cache = create_cache()
        cache.put("key1", "value1")
        
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        
        assert cache.delete("nonexistent") is False

    def test_size_with_mixed_expired_valid(self):
        """Test size() returns count of non-expired items and cleans up expired ones."""
        with patch('time.monotonic', side_effect=[0.0, 0.0, 0.0, 15.0]) as mock_time:
            cache = create_cache(capacity=5, default_ttl=10.0)
            
            # t=0: Add 3 items
            cache.put("a", 1)
            cache.put("b", 2)
            cache.put("c", 3)
            
            # t=0: Size should be 3
            assert cache.size() == 3
            
            # t=15: All items expired
            # size() should clean them up and return 0
            assert cache.size() == 0
            
            # Verify they are actually gone
            assert cache.get("a") is None
            assert cache.get("b") is None
            assert cache.get("c") is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
