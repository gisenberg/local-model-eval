# StringProcessor Class with Tests

## Implementation

```python
"""
StringProcessor - A utility class for common string operations.
"""

import re
from typing import Optional


class StringProcessor:
    """A class providing various string manipulation methods."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string.
        
        Multiple spaces between words are collapsed to a single space.
        Leading and trailing spaces are removed.
        
        Args:
            s: Input string
            
        Returns:
            String with words in reverse order
        """
        words = s.split()  # Split handles multiple spaces and stripping automatically
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive.
        
        Args:
            s: Input string
            
        Returns:
            Number of vowels in the string
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string
            
        Returns:
            True if the string is a palindrome, False otherwise
        """
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift to alphabetic characters.
        
        Only a-z and A-Z characters are shifted. Other characters remain unchanged.
        Supports negative shifts.
        
        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)
            
        Returns:
            String with shifted characters
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26 + base
                result.append(chr(shifted))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive).
        
        If tied, returns the word that appears first in the string.
        Returns None for empty strings.
        
        Args:
            s: Input string
            
        Returns:
            Most common word or None if string is empty
        """
        if not s.strip():
            return None
        
        words = s.lower().split()
        if not words:
            return None
        
        # Track counts while preserving order
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find max count
        max_count = max(word_counts.values())
        
        # Return first word with max count (preserves order)
        for word in words:
            if word_counts[word] == max_count:
                return word
```

## Pytest Tests

```python
"""
Tests for StringProcessor class.
"""

import pytest
from string_processor import StringProcessor


@pytest.fixture
def processor():
    """Create a StringProcessor instance for testing."""
    return StringProcessor()


def test_reverse_words(processor):
    """Test reverse_words method."""
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert processor.reverse_words("") == ""
    assert processor.reverse_words("single") == "single"
    assert processor.reverse_words("one two three four") == "four three two one"


def test_count_vowels(processor):
    """Test count_vowels method."""
    assert processor.count_vowels("hello") == 2
    assert processor.count_vowels("AEIOU") == 5
    assert processor.count_vowels("bcdfg") == 0
    assert processor.count_vowels("") == 0
    assert processor.count_vowels("aEiOu") == 5


def test_is_palindrome(processor):
    """Test is_palindrome method."""
    assert processor.is_palindrome("racecar") == True
    assert processor.is_palindrome("A man a plan a canal Panama") == True
    assert processor.is_palindrome("hello") == False
    assert processor.is_palindrome("Was it a car or a cat I saw?") == True
    assert processor.is_palindrome("12321") == True


def test_caesar_cipher(processor):
    """Test caesar_cipher method."""
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("xyz", 3) == "abc"
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("Hello!", 13) == "Uryyb!"
    assert processor.caesar_cipher("ABC", 26) == "ABC"  # Full rotation


def test_most_common_word(processor):
    """Test most_common_word method."""
    assert processor.most_common_word("the cat sat on the mat") == "the"
    assert processor.most_common_word("a b c a b") == "a"  # First in tie
    assert processor.most_common_word("") == None
    assert processor.most_common_word("hello") == "hello"
    assert processor.most_common_word("A a B b") == "a"  # Case-insensitive tie
```

## Verification

Let me verify the implementation against the requirements:

| Method | Requirement | Implementation |
|--------|-------------|----------------|
| `reverse_words` | Reverse word order, single spaces, strip | ✓ Uses `split()` which handles multiple spaces and stripping |
| `count_vowels` | Case-insensitive vowel count | ✓ Uses set with both cases |
| `is_palindrome` | Ignore case, spaces, punctuation | ✓ Uses regex to remove non-alphanumeric |
| `caesar_cipher` | Shift only letters, support negative | ✓ Uses modulo arithmetic with proper base |
| `most_common_word` | Case-insensitive, first on tie, None for empty | ✓ Tracks order and handles edge cases |

All methods include type hints and docstrings as required.