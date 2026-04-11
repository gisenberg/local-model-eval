import pytest
from unittest.mock import patch
from typing import Any
from your_module import TTLCache  # Replace 'your_module' with the actual filename

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
    """Test that LRU eviction works correctly when capacity is reached."""
    cache = create_cache(capacity=2, default_ttl=100.0)
    
    cache.put("a", 1)
    cache.put("b", 2)
    
    # Access 'a' to make it MRU, 'b' becomes LRU
    cache.get("a")
    
    # Insert 'c', should evict 'b'
    cache.put("c", 3)
    
    assert cache.get("a") == 1
    assert cache.get("b") is None  # Evicted
    assert cache.get("c") == 3

def test_ttl_expiry():
    """Test that items expire after default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = create_cache(capacity=2, default_ttl=5.0)
        
        cache.put("x", 100)
        assert cache.get("x") == 100
        
        # Advance time past TTL
        mock_time.return_value = 6.0
        
        assert cache.get("x") is None
        assert cache.size() == 0

def test_custom_per_key_ttl():
    """Test that custom TTL overrides default TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = create_cache(capacity=2, default_ttl=10.0)
        
        # 'short' expires in 2s, 'long' expires in 20s
        cache.put("short", 1, ttl=2.0)
        cache.put("long", 2, ttl=20.0)
        
        # Advance to 3s
        mock_time.return_value = 3.0
        
        assert cache.get("short") is None  # Expired
        assert cache.get("long") == 2      # Still valid
        
        # Advance to 25s
        mock_time.return_value = 25.0
        assert cache.get("long") is None   # Now expired

def test_delete():
    """Test the delete method."""
    cache = create_cache(capacity=2, default_ttl=10.0)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    
    assert cache.delete("key1") is False  # Already deleted
    assert cache.delete("nonexistent") is False
    
    assert cache.get("key2") == "value2"

def test_size_with_mixed_expired_valid():
    """Test size() returns count of non-expired items and cleans up expired ones."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = create_cache(capacity=5, default_ttl=5.0)
        
        cache.put("a", 1, ttl=2.0)   # Expires at 2
        cache.put("b", 2, ttl=10.0)  # Expires at 10
        cache.put("c", 3, ttl=2.0)   # Expires at 2
        
        # Initial size should be 3
        assert cache.size() == 3
        
        # Advance time to 3s (a and c expired)
        mock_time.return_value = 3.0
        
        # size() should clean up expired items and return 1
        assert cache.size() == 1
        
        # Verify only 'b' remains
        assert cache.get("b") == 2
        assert cache.get("a") is None
        assert cache.get("c") is None
