import unittest
from unittest.mock import patch
from typing import Any, Optional

# Import the implementation here to keep it in one file for copy-paste convenience
# In a real project, these would be separate files.
import sys
sys.path.insert(0, '.') 
from lru_cache import TTLCache, Node

class TestTTLCache(unittest.TestCase):
    
    def setUp(self):
        self.cache = TTLCache(capacity=2, default_ttl=10.0)

    @patch('ttl_cache.time.monotonic')
    def test_basic_put_and_get(self, mock_time):
        """Test basic put and get functionality."""
        mock_time.return_value = 100.0
        
        self.cache.put("key1", "value1")
        result = self.cache.get("key1")
        
        self.assertEqual(result, "value1")
        self.assertEqual(self.cache.size(), 1)

    @patch('ttl_cache.time.monotonic')
    def test_lru_eviction(self, mock_time):
        """Test that LRU eviction works when capacity is reached."""
        mock_time.return_value = 100.0
        
        # Fill cache
        self.cache.put("a", "val_a")
        self.cache.put("b", "val_b")
        
        # Access 'a' to make it MRU
        self.cache.get("a")
        
        # Add new item, should evict 'b' (LRU)
        self.cache.put("c", "val_c")
        
        self.assertIsNone(self.cache.get("b"))
        self.assertEqual(self.cache.get("a"), "val_a")
        self.assertEqual(self.cache.get("c"), "val_c")

    @patch('ttl_cache.time.monotonic')
    def test_ttl_expiration_lazy_cleanup(self, mock_time):
        """Test that expired items are removed on access (lazy cleanup)."""
        mock_time.return_value = 100.0
        
        # Insert item with TTL=5 seconds
        self.cache.put("key1", "value1", ttl=5.0)
        
        # Wait until it expires
        mock_time.return_value = 106.0
        
        # Access should return None and remove from cache
        result = self.cache.get("key1")
        
        self.assertIsNone(result)
        self.assertEqual(self.cache.size(), 0)

    @patch('ttl_cache.time.monotonic')
    def test_update_existing_key_refreshes_ttl(self, mock_time):
        """Test that updating a key refreshes its TTL."""
        mock_time.return_value = 100.0
        
        # Insert with short TTL
        self.cache.put("key1", "value1", ttl=5.0)
        
        # Wait for expiration
        mock_time.return_value = 106.0
        
        # Access should fail (expired)
        self.assertIsNone(self.cache.get("key1"))
        
        # Update the key with new value and long TTL
        self.cache.put("key1", "value2", ttl=100.0)
        
        # Now it should be valid again
        mock_time.return_value = 107.0
        result = self.cache.get("key1")
        
        self.assertEqual(result, "value2")

    @patch('ttl_cache.time.monotonic')
    def test_delete_method(self, mock_time):
        """Test delete method returns correct boolean."""
        mock_time.return_value = 100.0
        
        self.cache.put("key1", "value1")
        
        # Delete existing
        result = self.cache.delete("key1")
        self.assertTrue(result)
        self.assertIsNone(self.cache.get("key1"))
        
        # Delete non-existing
        result = self.cache.delete("missing_key")
        self.assertFalse(result)

    @patch('ttl_cache.time.monotonic')
    def test_size_consistency_with_lazy_cleanup(self, mock_time):
        """Test that size reflects valid items after lazy cleanup."""
        mock_time.return_value = 100.0
        
        # Fill cache with capacity 2
        self.cache.put("a", "val_a")
        self.cache.put("b", "val_b")
        
        # Set 'a' to expire immediately (TTL=0)
        self.cache.put("a", "val_a", ttl=0.0)
        
        # Size should still be 2 because cleanup is lazy
        self.assertEqual(self.cache.size(), 2)
        
        # Access 'a', it expires and gets removed
        mock_time.return_value = 100.0
        self.assertIsNone(self.cache.get("a"))
        
        # Size should now be 1
        self.assertEqual(self.cache.size(), 1)

if __name__ == '__main__':
    unittest.main()
