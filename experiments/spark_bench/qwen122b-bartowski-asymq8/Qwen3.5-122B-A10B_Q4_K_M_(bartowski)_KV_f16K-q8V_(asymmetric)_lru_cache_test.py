import pytest
from unittest.mock import patch
from ttl_cache_impl import TTLCache  # Assuming the class above is saved as ttl_cache_impl.py

# Helper to create a cache instance
def create_cache(capacity=3, default_ttl=10.0):
    return TTLCache(capacity=capacity, default_ttl=default_ttl)

class TestTTLCache:

    def test_basic_get_put(self):
        """Test basic insertion and retrieval."""
        cache = create_cache()
        cache.put("key1", "value1")
        
        assert cache.get("key1") == "value1"
        assert cache.get("non_existent") is None
        
        # Update existing key
        cache.put("key1", "updated_value")
        assert cache.get("key1") == "updated_value"

    def test_capacity_eviction_lru_order(self):
        """Test that LRU eviction works correctly when capacity is reached."""
        cache = create_cache(capacity=2)
        
        cache.put("a", 1)
        cache.put("b", 2)
        
        # Access 'a' to make it MRU, 'b' becomes LRU
        cache.get("a")
        
        # Insert 'c', should evict 'b' (LRU)
        cache.put("c", 3)
        
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("b") is None  # Evicted

    def test_ttl_expiry(self):
        """Test that items expire after default TTL."""
        with patch('time.monotonic', return_value=0.0):
            cache = create_cache(capacity=2, default_ttl=5.0)
            cache.put("key1", "value1")
            
            # Time passes, but within TTL
            with patch('time.monotonic', return_value=4.9):
                assert cache.get("key1") == "value1"
            
            # Time passes, exceeds TTL
            with patch('time.monotonic', return_value=5.1):
                assert cache.get("key1") is None
                # Verify it's actually removed from internal state
                assert cache.size() == 0

    def test_custom_per_key_ttl(self):
        """Test that custom TTL overrides default TTL."""
        with patch('time.monotonic', return_value=0.0):
            cache = create_cache(capacity=2, default_ttl=10.0)
            
            # Set with custom short TTL
            cache.put("short", "val", ttl=2.0)
            # Set with default TTL
            cache.put("long", "val")
            
            # At t=3, 'short' should be expired, 'long' should be valid
            with patch('time.monotonic', return_value=3.0):
                assert cache.get("short") is None
                assert cache.get("long") == "val"

    def test_delete(self):
        """Test the delete method."""
        cache = create_cache()
        cache.put("key1", "value1")
        
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        
        # Deleting non-existent key
        assert cache.delete("non_existent") is False

    def test_size_with_mixed_expired_valid_items(self):
        """Test size() returns count of non-expired items and cleans up expired ones."""
        with patch('time.monotonic', return_value=0.0):
            cache = create_cache(capacity=5, default_ttl=5.0)
            
            cache.put("valid1", "v1")
            cache.put("valid2", "v2")
            cache.put("expiring", "v3", ttl=2.0)
            
            # At t=0, size should be 3
            assert cache.size() == 3
            
            # Advance time to expire one item
            with patch('time.monotonic', return_value=3.0):
                # size() should clean up 'expiring' and return 2
                assert cache.size() == 2
                
                # Verify the expired item is gone
                assert cache.get("expiring") is None
                assert cache.get("valid1") == "v1"
