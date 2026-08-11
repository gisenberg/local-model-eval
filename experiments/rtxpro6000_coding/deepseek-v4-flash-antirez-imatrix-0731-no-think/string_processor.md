Here's the implementation of the `StringProcessor` class with all the required methods and pytest tests:

```python
from typing import Optional
import re
from collections import Counter

class StringProcessor:
    """
    A class that provides various string processing utilities.
    """

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.

        Args:
            s: Input string

        Returns:
            String with words in reverse order, single spaces between words,
            no leading/trailing spaces.
        """
        # Split by whitespace and filter out empty strings
        words = s.split()
        # Reverse the words and join with single space
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count vowels (a, e, i, o, u) in the string, case-insensitive.

        Args:
            s: Input string

        Returns:
            Number of vowels in the string
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

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
        # Check if the cleaned string equals its reverse
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift to the string.
        Only shifts a-z and A-Z, leaves other characters unchanged.
        Supports negative shifts.

        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)

        Returns:
            String with Caesar cipher applied
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                # Shift lowercase letters
                shifted = (ord(char) - ord('a') + shift) % 26 + ord('a')
                result.append(chr(shifted))
            elif 'A' <= char <= 'Z':
                # Shift uppercase letters
                shifted = (ord(char) - ord('A') + shift) % 26 + ord('A')
                result.append(chr(shifted))
            else:
                # Leave non-alphabetic characters unchanged
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive).
        If tied, return the one that appears first.
        Return None for empty strings.

        Args:
            s: Input string

        Returns:
            Most common word or None if string is empty
        """
        if not s or not s.strip():
            return None

        # Split into words and convert to lowercase
        words = s.lower().split()

        # Count occurrences
        word_counts = Counter(words)

        # Find the most common word (first occurrence wins on ties)
        max_count = max(word_counts.values())
        for word in words:
            if word_counts[word] == max_count:
                return word

        return None
```

Now, here are the pytest tests:

```python
import pytest
from typing import Optional
from string_processor import StringProcessor

@pytest.fixture
def processor():
    return StringProcessor()

def test_reverse_words(processor):
    """Test reversing words in a string."""
    # Basic case
    assert processor.reverse_words("Hello world") == "world Hello"

    # Multiple spaces between words
    assert processor.reverse_words("Hello   world   test") == "test world Hello"

    # Leading/trailing spaces
    assert processor.reverse_words("  Hello world  ") == "world Hello"

    # Single word
    assert processor.reverse_words("Hello") == "Hello"

    # Empty string
    assert processor.reverse_words("") == ""

def test_count_vowels(processor):
    """Test counting vowels in a string."""
    # Basic case
    assert processor.count_vowels("Hello") == 2  # e, o

    # Case-insensitive
    assert processor.count_vowels("AEIOU") == 5
    assert processor.count_vowels("aeiou") == 5

    # Mixed case
    assert processor.count_vowels("Hello World") == 3  # e, o, o

    # No vowels
    assert processor.count_vowels("bcdfg") == 0

    # Empty string
    assert processor.count_vowels("") == 0

def test_is_palindrome(processor):
    """Test palindrome checking."""
    # Basic palindrome
    assert processor.is_palindrome("racecar") == True

    # Palindrome with spaces and punctuation
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True

    # Case-insensitive
    assert processor.is_palindrome("Racecar") == True

    # Not a palindrome
    assert processor.is_palindrome("hello") == False

    # Palindrome with numbers
    assert processor.is_palindrome("12321") == True

    # Empty string (considered palindrome)
    assert processor.is_palindrome("") == True

def test_caesar_cipher(processor):
    """Test Caesar cipher encryption."""
    # Basic shift
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("ABC", 1) == "BCD"

    # Wrap around
    assert processor.caesar_cipher("xyz", 3) == "abc"
    assert processor.caesar_cipher("XYZ", 3) == "ABC"

    # Negative shift
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("ABC", -1) == "ZAB"

    # Non-alphabetic characters unchanged
    assert processor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"

    # Zero shift
    assert processor.caesar_cipher("Hello", 0) == "Hello"

    # Large shift
    assert processor.caesar_cipher("abc", 26) == "abc"

def test_most_common_word(processor):
    """Test finding the most common word."""
    # Basic case
    assert processor.most_common_word("the quick brown fox the lazy dog") == "the"

    # Case-insensitive
    assert processor.most_common_word("Hello hello HELLO world") == "hello"

    # Tie-breaking (first occurrence wins)
    assert processor.most_common_word("apple banana apple banana") == "apple"

    # Single word
    assert processor.most_common_word("hello") == "hello"

    # Empty string
    assert processor.most_common_word("") is None

    # String with only spaces
    assert processor.most_common_word("   ") is None

    # Multiple words with different frequencies
    assert processor.most_common_word("a b c a b a") == "a"
```

The tests cover:

1. **test_reverse_words**: Tests basic reversal, multiple spaces, leading/trailing spaces, single word, and empty string.
2. **test_count_vowels**: Tests basic counting, case-insensitivity, mixed case, no vowels, and empty string.
3. **test_is_palindrome**: Tests basic palindromes, palindromes with punctuation/spaces, case-insensitivity, non-palindromes, palindromes with numbers, and empty string.
4. **test_caesar_cipher**: Tests basic shifts, wrap-around, negative shifts, non-alphabetic characters, zero shift, and large shifts.
5. **test_most_common_word**: Tests basic frequency, case-insensitivity, tie-breaking, single word, empty string, whitespace-only string, and different frequencies.