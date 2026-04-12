import pytest
from unittest.mock import patch
import time

# Assuming the TTLCache class is in the same file or imported
# from your_module import TTLCache 

def test_basic_get_put():
    """Test basic get and put operations."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=2, default_ttl=10.0)
        
        cache.put("a", 1)
        assert cache.get("a") == 1
        
        cache.put("b", 2)
        assert cache.get("b") == 2
        assert cache.get("a") == 1

def test_capacity_eviction_lru_order():
    """Test that eviction removes the least recently used item."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=3, default_ttl=10.0)
        
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        # Access 'a' to make it most recent
        cache.get("a")
        
        # Add one more, should evict 'b' (least recent)
        cache.put("d", 4)
        
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3
        assert cache.get("d") == 4

def test_ttl_expiry():
    """Test that items expire after TTL."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=5.0)
        
        cache.put("key", "value")
        assert cache.get("key") == "value"
        
        # Advance time past expiry
        mock_time.return_value = 6.0
        assert cache.get("key") is None

def test_custom_per_key_ttl():
    """Test that custom TTL overrides default."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=2.0)
        
        cache.put("short", "val", ttl=1.0)
        cache.put("long", "val", ttl=10.0)
        
        # At t=1.5, short should expire, long should not
        mock_time.return_value = 1.5
        assert cache.get("short") is None
        assert cache.get("long") == "val"
        
        # At t=3.0, long should expire
        mock_time.return_value = 3.0
        assert cache.get("long") is None

def test_delete():
    """Test deletion of existing and non-existing keys."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=5, default_ttl=10.0)
        
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        
        assert cache.delete("a") is False # Already deleted
        assert cache.delete("missing") is False

def test_size_with_mixed_expired_valid():
    """Test size calculation with a mix of expired and valid items."""
    with patch('time.monotonic') as mock_time:
        mock_time.return_value = 0.0
        cache = TTLCache(capacity=10, default_ttl=5.0)
        
        cache.put("a", 1)      # Expires at 5
        cache.put("b", 2)      # Expires at 5
        cache.put("c", 3, ttl=100) # Expires at 100
        
        mock_time.return_value = 6.0
        
        # 'a' and 'b' expired, 'c' valid
        assert cache.size() == 1
        
        # Add more valid items
        cache.put("d", 4)
        assert cache.size() == 2
        
        # Verify we can still evict correctly if needed
        cache.put("e", 5)
        cache.put("f", 6)
        # Capacity is 10, we have 4 valid, adding more is fine
        assert cache.size() == 4
