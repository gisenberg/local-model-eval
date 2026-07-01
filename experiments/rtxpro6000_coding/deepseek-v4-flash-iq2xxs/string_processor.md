Below is the implementation of the `StringProcessor` class along with five pytest tests covering each method.

```python
import re
from typing import Optional

class StringProcessor:
    """A utility class for processing strings."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in a string.

        Multiple spaces between words become a single space.
        Leading/trailing spaces are removed.

        Args:
            s: Input string.

        Returns:
            String with words reversed.
        """
        words = s.split()
        return " ".join(reversed(words))

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string (case-insensitive).

        Args:
            s: Input string.

        Returns:
            Number of vowels.
        """
        vowels = set("aeiou")
        return sum(1 for ch in s.lower() if ch in vowels)

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation.

        Only alphanumeric characters are considered.

        Args:
            s: Input string.

        Returns:
            True if palindrome, False otherwise.
        """
        # Keep only alphanumeric characters (letters and digits)
        filtered = "".join(ch for ch in s if ch.isalnum())
        return filtered.lower() == filtered.lower()[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift.

        Only shift a-z and A-Z, leave other characters unchanged.
        Supports negative shifts.

        Args:
            s: Input string.
            shift: Shift value (can be negative).

        Returns:
            Shifted string.
        """
        result = []
        for ch in s:
            if "a" <= ch <= "z":
                new_ord = (ord(ch) - ord("a") + shift) % 26 + ord("a")
                result.append(chr(new_ord))
            elif "A" <= ch <= "Z":
                new_ord = (ord(ch) - ord("A") + shift) % 26 + ord("A")
                result.append(chr(new_ord))
            else:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive).

        If tied, return the one that appears first.
        Return None for empty strings.

        Args:
            s: Input string.

        Returns:
            Most common word or None.
        """
        if not s:
            return None

        words = s.split()
        if not words:
            return None

        # Count occurrences (case-insensitive)
        word_counts = {}
        for word in words:
            lower_word = word.lower()
            word_counts[lower_word] = word_counts.get(lower_word, 0) + 1

        # Find the most common word, using first occurrence as tie-breaker
        max_count = max(word_counts.values())
        # Iterate in order of first appearance (preserve insertion order)
        for word in words:
            if word_counts[word.lower()] == max_count:
                return word.lower()  # return the word in lower case as per typical convention
        return None  # Should not reach here
```

## Pytest Tests

```python
import pytest
from string_processor import StringProcessor  # assuming the class is in string_processor.py

class TestStringProcessor:

    def test_reverse_words(self):
        # Test basic reversal
        assert StringProcessor.reverse_words("Hello world") == "world Hello"
        # Test multiple spaces
        assert StringProcessor.reverse_words("a   b   c") == "c b a"
        # Test leading/trailing spaces
        assert StringProcessor.reverse_words("   leading trailing   ") == "trailing leading"
        # Test single word
        assert StringProcessor.reverse_words("hello") == "hello"
        # Test empty string
        assert StringProcessor.reverse_words("") == ""

    def test_count_vowels(self):
        # Test basic
        assert StringProcessor.count_vowels("Hello World") == 3  # e, o, o
        # Test case-insensitivity
        assert StringProcessor.count_vowels("AEIOUaeiou") == 10
        # Test no vowels
        assert StringProcessor.count_vowels("bcdfg") == 0
        # Test mixed
        assert StringProcessor.count_vowels("Python is fun") == 4  # o, i, u
        # Test empty string
        assert StringProcessor.count_vowels("") == 0

    def test_is_palindrome(self):
        # Classic palindrome
        assert StringProcessor.is_palindrome("A man, a plan, a canal, Panama!")
        # Case-insensitive
        assert StringProcessor.is_palindrome("Racecar")
        # With digits
        assert StringProcessor.is_palindrome("12321")
        # Not a palindrome
        assert not StringProcessor.is_palindrome("hello")
        # Empty string (considered palindrome)
        assert StringProcessor.is_palindrome("")
        # Only punctuation
        assert StringProcessor.is_palindrome(".,!")  # after filtering, empty string -> palindrome

    def test_caesar_cipher(self):
        # Positive shift
        assert StringProcessor.caesar_cipher("abc", 1) == "bcd"
        # Wrap around
        assert StringProcessor.caesar_cipher("xyz", 1) == "yza"
        # Negative shift
        assert StringProcessor.caesar_cipher("bcd", -1) == "abc"
        # Mixed case
        assert StringProcessor.caesar_cipher("Hello", 3) == "Khoor"
        # Non-alphabetic characters unchanged
        assert StringProcessor.caesar_cipher("abc123", 2) == "cde123"
        # Large shift
        assert StringProcessor.caesar_cipher("abc", 26) == "abc"
        # Negative large shift
        assert StringProcessor.caesar_cipher("abc", -26) == "abc"

    def test_most_common_word(self):
        # Basic
        assert StringProcessor.most_common_word("apple banana apple") == "apple"
        # Tie-breaking (first occurrence)
        assert StringProcessor.most_common_word("apple banana banana apple") == "apple"  # both appear twice, apple first
        # Case-insensitive
        assert StringProcessor.most_common_word("Apple apple banana") == "apple"
        # Single word
        assert StringProcessor.most_common_word("hello") == "hello"
        # Empty string
        assert StringProcessor.most_common_word("") is None
        # String with only spaces
        assert StringProcessor.most_common_word("   ") is None
        # Multiple words with punctuation (treated as part of word)
        assert StringProcessor.most_common_word("hello, hello world") == "hello,"  # because "hello," and "hello" are different
```

**Note:** The test for `most_common_word` with punctuation reflects the current implementation where punctuation is part of the word. If the intended behavior is to strip punctuation, the method would need to be adjusted accordingly. The provided tests match the implementation as described.