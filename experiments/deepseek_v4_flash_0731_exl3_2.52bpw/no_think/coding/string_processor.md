Here's the implementation of the `StringProcessor` class with all required methods and pytest tests:

```python
from typing import Optional
import re
from collections import Counter

class StringProcessor:
    """A class for processing and analyzing strings."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.

        Args:
            s: Input string

        Returns:
            String with words in reverse order, single spaces between words,
            no leading/trailing spaces
        """
        # Split on whitespace and filter empty strings
        words = [word for word in s.split() if word]
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count vowels (a, e, i, o, u) in the string, case-insensitive.

        Args:
            s: Input string

        Returns:
            Number of vowels in the string
        """
        vowels = 'aeiou'
        return sum(1 for char in s.lower() if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if the string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: Input string

        Returns:
            True if the string is a palindrome, False otherwise
        """
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift to the string.
        Only shifts a-z and A-Z, leaves other characters unchanged.

        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)

        Returns:
            Encrypted string
        """
        result = []
        for char in s:
            if char.isalpha():
                # Determine base (uppercase or lowercase)
                base = ord('A') if char.isupper() else ord('a')
                # Apply shift and wrap around
                shifted = (ord(char) - base + shift) % 26
                result.append(chr(base + shifted))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Find the most frequently occurring word (case-insensitive).
        If tied, return the one that appears first.

        Args:
            s: Input string

        Returns:
            Most common word or None for empty strings
        """
        if not s.strip():
            return None

        # Split into words, convert to lowercase
        words = [word.lower() for word in s.split()]

        # Count word frequencies
        word_counts = Counter(words)

        # Find the most common word, with tie-breaking by first occurrence
        max_count = max(word_counts.values())
        most_common = None

        for word in words:
            if word_counts[word] == max_count:
                most_common = word
                break

        return most_common
```

Now here are the pytest tests:

```python
import pytest
from string_processor import StringProcessor

@pytest.fixture
def processor():
    return StringProcessor()

def test_reverse_words(processor):
    """Test reversing word order with various spacing scenarios."""
    # Basic case
    assert processor.reverse_words("Hello world") == "world Hello"

    # Multiple spaces between words
    assert processor.reverse_words("Hello   world   again") == "again world Hello"

    # Leading and trailing spaces
    assert processor.reverse_words("  Hello world  ") == "world Hello"

    # Single word
    assert processor.reverse_words("Hello") == "Hello"

    # Empty string
    assert processor.reverse_words("") == ""

def test_count_vowels(processor):
    """Test vowel counting with various cases."""
    # Basic case
    assert processor.count_vowels("Hello") == 2  # e, o

    # Case-insensitive
    assert processor.count_vowels("AEIOU") == 5

    # Mixed case
    assert processor.count_vowels("Hello World") == 3  # e, o, o

    # No vowels
    assert processor.count_vowels("bcdfg") == 0

    # Empty string
    assert processor.count_vowels("") == 0

def test_is_palindrome(processor):
    """Test palindrome detection with various formatting."""
    # Basic palindrome
    assert processor.is_palindrome("racecar") == True

    # Case-insensitive
    assert processor.is_palindrome("Racecar") == True

    # With spaces and punctuation
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True

    # Not a palindrome
    assert processor.is_palindrome("hello") == False

    # Empty string (should be palindrome)
    assert processor.is_palindrome("") == True

def test_caesar_cipher(processor):
    """Test Caesar cipher with various shifts."""
    # Basic shift
    assert processor.caesar_cipher("Hello", 3) == "Khoor"

    # Negative shift
    assert processor.caesar_cipher("Khoor", -3) == "Hello"

    # Wrap around
    assert processor.caesar_cipher("xyz", 3) == "abc"

    # Preserve non-alphabetic characters
    assert processor.caesar_cipher("Hello, World!", 1) == "Ifmmp, Xpsme!"

    # Uppercase and lowercase preserved
    assert processor.caesar_cipher("AbC", 1) == "BcD"

def test_most_common_word(processor):
    """Test finding the most common word with tie-breaking."""
    # Basic case
    assert processor.most_common_word("the quick brown fox jumps over the lazy dog") == "the"

    # Case-insensitive
    assert processor.most_common_word("The quick brown fox jumps over the lazy dog") == "the"

    # Tie-breaking (first occurrence wins)
    assert processor.most_common_word("apple banana apple cherry banana") == "apple"

    # Empty string
    assert processor.most_common_word("") is None

    # Whitespace only
    assert processor.most_common_word("   ") is None
```

To run the tests, save the class in a file named `string_processor.py` and the tests in `test_string_processor.py`, then run:

```bash
pytest test_string_processor.py -v
```

The tests cover all five methods with various edge cases including empty strings, special characters, case sensitivity, and tie-breaking scenarios.