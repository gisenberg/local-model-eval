import pytest
from unittest.mock import patch
from ttl_cache_impl import TTLCache  # Assuming the implementation is saved as ttl_cache_impl.py

# Helper to create a cache instance
def create_cache(capacity=3, default_ttl=10.0):
    return TTLCache(capacity=capacity, default_ttl=default_ttl)

class TestTTLCache:

    def test_basic_get_put(self):
        """Test basic insertion and retrieval."""
        cache = create_cache(capacity=2, default_ttl=10.0)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        cache.put("key2", "value2")
        assert cache.get("key2") == "value2"
        
        # Update existing key
        cache.put("key1", "updated_value1")
        assert cache.get("key1") == "updated_value1"
        
        # Non-existent key
        assert cache.get("non_existent") is None

    def test_capacity_eviction_lru_order(self):
        """Test that LRU items are evicted when capacity is reached."""
        cache = create_cache(capacity=2, default_ttl=100.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        
        # Access 'a' to make it MRU
        cache.get("a")
        
        # Add 'c'. 'b' is LRU and should be evicted.
        cache.put("c", 3)
        
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("b") is None  # Evicted

    def test_ttl_expiry(self):
        """Test that items expire after default TTL."""
        with patch('time.monotonic', return_value=0.0) as mock_time:
            cache = create_cache(capacity=2, default_ttl=5.0)
            
            cache.put("key1", "value1")
            assert cache.get("key1") == "value1"
            
            # Advance time to just before expiry
            mock_time.return_value = 4.9
            assert cache.get("key1") == "value1"
            
            # Advance time past expiry
            mock_time.return_value = 5.1
            assert cache.get("key1") is None

    def test_custom_per_key_ttl(self):
        """Test that custom TTL overrides default TTL."""
        with patch('time.monotonic', return_value=0.0) as mock_time:
            cache = create_cache(capacity=2, default_ttl=10.0)
            
            # Item with custom short TTL
            cache.put("short", "val", ttl=2.0)
            # Item with default TTL
            cache.put("long", "val", ttl=None)
            
            # Time 3.0: 'short' should be expired, 'long' valid
            mock_time.return_value = 3.0
            assert cache.get("short") is None
            assert cache.get("long") == "val"
            
            # Time 11.0: 'long' should also be expired
            mock_time.return_value = 11.0
            assert cache.get("long") is None

    def test_delete(self):
        """Test manual deletion of keys."""
        cache = create_cache(capacity=2, default_ttl=10.0)
        
        cache.put("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        
        # Delete non-existent key
        assert cache.delete("non_existent") is False

    def test_size_with_mixed_expired_valid_items(self):
        """Test size() returns count of valid items and cleans up expired ones."""
        with patch('time.monotonic', return_value=0.0) as mock_time:
            cache = create_cache(capacity=5, default_ttl=5.0)
            
            cache.put("valid1", 1)
            cache.put("valid2", 2)
            cache.put("exp1", 3, ttl=2.0)
            cache.put("exp2", 4, ttl=1.0)
            
            # Initially all 4 are present in map, but size should count valid?
            # Requirement: "return count of non-expired items (lazy cleanup: expired items removed on access)"
            # size() triggers cleanup.
            
            # At t=0, all are valid.
            assert cache.size() == 4
            
            # Advance time to 3.0. exp1 and exp2 are expired.
            mock_time.return_value = 3.0
            
            # size() should clean up exp1 and exp2 and return 2
            assert cache.size() == 2
            
            # Verify expired items are actually gone
            assert cache.get("exp1") is None
            assert cache.get("exp2") is None
            assert cache.get("valid1") == 1
            assert cache.get("valid2") == 2
