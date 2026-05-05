Here's the implementation of the `StringProcessor` class with all the requested methods, along with pytest tests:

```python
from typing import Optional
import re
from collections import defaultdict

class StringProcessor:
    """A class for processing strings with various operations."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in a string, handling multiple spaces.

        Args:
            s: Input string to process

        Returns:
            String with words in reverse order, single spaces between words,
            no leading/trailing spaces
        """
        words = s.split()
        return ' '.join(reversed(words))

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in a string (case-insensitive).

        Args:
            s: Input string to process

        Returns:
            Number of vowels (a, e, i, o, u) in the string
        """
        vowels = {'a', 'e', 'i', 'o', 'u'}
        return sum(1 for char in s.lower() if char in vowels)

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: Input string to check

        Returns:
            True if the string is a palindrome, False otherwise
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift to alphabetic characters.

        Args:
            s: Input string to process
            shift: Number of positions to shift (can be negative)

        Returns:
            String with alphabetic characters shifted, other characters unchanged
        """
        result = []
        shift = shift % 26  # Normalize shift to handle large values

        for char in s:
            if 'a' <= char <= 'z':
                # Shift lowercase letters
                new_ord = ord('a') + (ord(char) - ord('a') + shift) % 26
                result.append(chr(new_ord))
            elif 'A' <= char <= 'Z':
                # Shift uppercase letters
                new_ord = ord('A') + (ord(char) - ord('A') + shift) % 26
                result.append(chr(new_ord))
            else:
                # Leave other characters unchanged
                result.append(char)

        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Return the most frequently occurring word (case-insensitive).

        Args:
            s: Input string to process

        Returns:
            The most common word (first in case of tie), or None for empty string
        """
        if not s.strip():
            return None

        words = re.findall(r'\b\w+\b', s.lower())
        word_counts = defaultdict(int)

        for word in words:
            word_counts[word] += 1

        if not word_counts:
            return None

        max_count = max(word_counts.values())
        # Find the first word with max count
        for word in words:
            if word_counts[word] == max_count:
                return word

        return None

# Pytest tests
import pytest

def test_reverse_words():
    processor = StringProcessor()
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("  multiple   spaces   ") == "spaces multiple"
    assert processor.reverse_words("") == ""
    assert processor.reverse_words("single") == "single"
    assert processor.reverse_words("  leading and trailing  ") == "trailing and leading"

def test_count_vowels():
    processor = StringProcessor()
    assert processor.count_vowels("hello") == 2
    assert processor.count_vowels("AEIOUaeiou") == 10
    assert processor.count_vowels("xyz") == 0
    assert processor.count_vowels("") == 0
    assert processor.count_vowels("The quick brown fox") == 5

def test_is_palindrome():
    processor = StringProcessor()
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
    assert processor.is_palindrome("racecar") == True
    assert processor.is_palindrome("hello") == False
    assert processor.is_palindrome("") == True
    assert processor.is_palindrome("No 'x' in Nixon") == True

def test_caesar_cipher():
    processor = StringProcessor()
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("XYZ", 3) == "ABC"
    assert processor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("abc", 26) == "abc"

def test_most_common_word():
    processor = StringProcessor()
    assert processor.most_common_word("hello hello world") == "hello"
    assert processor.most_common_word("the quick brown fox jumps over the lazy dog") == "the"
    assert processor.most_common_word("") is None
    assert processor.most_common_word("   ") is None
    assert processor.most_common_word("tie tie first first") == "tie"
```

This implementation includes:

1. The `StringProcessor` class with all 5 requested methods, each with:
   - Type hints
   - Docstrings explaining the method's purpose and parameters
   - Proper handling of edge cases

2. For each method:
   - `reverse_words`: Handles multiple spaces, leading/trailing spaces
   - `count_vowels`: Case-insensitive counting of a, e, i, o, u
   - `is_palindrome`: Ignores case, spaces, and punctuation
   - `caesar_cipher`: Handles both positive and negative shifts, wraps around alphabet, leaves non-alphabetic characters unchanged
   - `most_common_word`: Case-insensitive, returns first in case of tie, returns None for empty strings

3. Pytest tests covering:
   - Normal cases
   - Edge cases (empty strings, etc.)
   - Special cases (punctuation, mixed case, etc.)
   - Each method has at least one test case

The tests can be run with pytest to verify the implementation works correctly.