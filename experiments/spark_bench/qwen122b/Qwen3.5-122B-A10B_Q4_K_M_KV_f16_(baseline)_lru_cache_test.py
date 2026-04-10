import pytest
from unittest.mock import patch
from typing import Any
from lru_cache import TTLCache  # Replace 'your_module' with the actual filename

# Helper to create a cache instance
def create_cache(capacity: int = 3, default_ttl: float = 10.0):
    return TTLCache(capacity, default_ttl)

def test_basic_get_put():
    """Test basic insertion and retrieval."""
    cache = create_cache(capacity=2, default_ttl=10.0)
    
    cache.put("a", 1)
    cache.put("b", 2)
    
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") is None
    
    # Update existing
    cache.put("a", 10)
    assert cache.get("a") == 10

def test_capacity_eviction_lru_order():
    """Test that LRU items are evicted when capacity is reached."""
    cache = create_cache(capacity=2, default_ttl=100.0)
    
    cache.put("a", 1)
    cache.put("b", 2)
    
    # Access 'a' to make it MRU
    cache.get("a")
    
    # Add 'c', should evict 'b' (LRU)
    cache.put("c", 3)
    
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None

def test_ttl_expiry():
    """Test that items expire after default TTL."""
    cache = create_cache(capacity=2, default_ttl=5.0)
    
    with patch('time.monotonic', return_value=0.0):
        cache.put("a", 1)
        assert cache.get("a") == 1
        
        # Advance time past TTL
        with patch('time.monotonic', return_value=6.0):
            assert cache.get("a") is None
            assert cache.size() == 0

def test_custom_per_key_ttl():
    """Test that custom TTL overrides default TTL."""
    cache = create_cache(capacity=2, default_ttl=10.0)
    
    with patch('time.monotonic', return_value=0.0):
        cache.put("short", 1, ttl=2.0)
        cache.put("long", 2, ttl=20.0)
        
        # Advance time to 3.0: 'short' should expire, 'long' should remain
        with patch('time.monotonic', return_value=3.0):
            assert cache.get("short") is None
            assert cache.get("long") == 2
            
            # Advance time to 21.0: 'long' should expire
            with patch('time.monotonic', return_value=21.0):
                assert cache.get("long") is None

def test_delete():
    """Test manual deletion of keys."""
    cache = create_cache(capacity=2, default_ttl=10.0)
    
    cache.put("a", 1)
    cache.put("b", 2)
    
    assert cache.delete("a") is True
    assert cache.delete("a") is False  # Already deleted
    assert cache.delete("c") is False  # Never existed
    
    assert cache.get("a") is None
    assert cache.get("b") == 2

def test_size_with_mixed_expired_valid():
    """Test size calculation with mixed expired and valid items."""
    cache = create_cache(capacity=5, default_ttl=10.0)
    
    with patch('time.monotonic', return_value=0.0):
        cache.put("valid1", 1)
        cache.put("valid2", 2)
        cache.put("expiring", 3, ttl=5.0)
        cache.put("valid3", 4)
        
        # Advance time to 6.0: 'expiring' is now expired
        with patch('time.monotonic', return_value=6.0):
            # Size should trigger cleanup and return 3 (valid1, valid2, valid3)
            assert cache.size() == 3
            
            # Verify expired item is gone
            assert cache.get("expiring") is None
            
            # Size should now be 3
            assert cache.size() == 3
