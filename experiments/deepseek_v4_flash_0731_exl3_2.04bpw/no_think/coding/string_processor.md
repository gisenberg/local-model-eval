Here's the complete implementation with tests:

```python
from typing import Optional
import re
import string


class StringProcessor:
    """A class with various string processing methods."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.
        Multiple spaces become single space, leading/trailing spaces removed.

        Args:
            s: Input string

        Returns:
            String with words in reverse order
        """
        # Split on whitespace (handles multiple spaces), then join in reverse
        words = s.split()
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
        Check if string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: Input string

        Returns:
            True if palindrome, False otherwise
        """
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift to letters only.
        Non-letter characters remain unchanged. Supports negative shifts.

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
        Return the most frequently occurring word (case-insensitive).
        If tied, return the one that appears first.

        Args:
            s: Input string

        Returns:
            Most common word or None if empty string
        """
        if not s.strip():
            return None

        # Split into words and count occurrences
        words = s.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Find the word with highest count, first occurrence wins ties
        max_count = 0
        most_common = None
        for word in words:
            if word_counts[word] > max_count:
                max_count = word_counts[word]
                most_common = word

        return most_common


# Test file: test_string_processor.py
import pytest
from string_processor import StringProcessor


@pytest.fixture
def processor():
    return StringProcessor()


def test_reverse_words(processor):
    """Test reversing word order with various spacing scenarios."""
    assert processor.reverse_words("Hello world") == "world Hello"
    assert processor.reverse_words("a b c") == "c b a"
    assert processor.reverse_words("  multiple   spaces   here  ") == "here spaces multiple"
    assert processor.reverse_words("single") == "single"
    assert processor.reverse_words("") == ""


def test_count_vowels(processor):
    """Test vowel counting with case-insensitivity."""
    assert processor.count_vowels("hello") == 2
    assert processor.count_vowels("HELLO") == 2
    assert processor.count_vowels("aeiou") == 5
    assert processor.count_vowels("AEIOU") == 5
    assert processor.count_vowels("xyz") == 0
    assert processor.count_vowels("") == 0
    assert processor.count_vowels("Hello World!") == 3


def test_is_palindrome(processor):
    """Test palindrome detection ignoring case, spaces, and punctuation."""
    assert processor.is_palindrome("racecar") == True
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
    assert processor.is_palindrome("No 'x' in Nixon") == True
    assert processor.is_palindrome("hello") == False
    assert processor.is_palindrome("") == True
    assert processor.is_palindrome("12321") == True
    assert processor.is_palindrome("12345") == False


def test_caesar_cipher(processor):
    """Test Caesar cipher with positive, negative, and zero shifts."""
    assert processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert processor.caesar_cipher("Hello, World!", -3) == "Ebiil, Tloia!"
    assert processor.caesar_cipher("abc", 0) == "abc"
    assert processor.caesar_cipher("ABC", 1) == "BCD"
    assert processor.caesar_cipher("xyz", 1) == "yza"
    assert processor.caesar_cipher("XYZ", -1) == "WXY"
    assert processor.caesar_cipher("Hello123!", 5) == "Mjqqt123!"
    assert processor.caesar_cipher("", 5) == ""


def test_most_common_word(processor):
    """Test finding most common word with tie-breaking."""
    assert processor.most_common_word("apple banana apple") == "apple"
    assert processor.most_common_word("the quick brown fox jumps over the lazy dog") == "the"
    assert processor.most_common_word("a b c a b") == "a"  # tie, first occurrence wins
    assert processor.most_common_word("") == None
    assert processor.most_common_word("   ") == None
    assert processor.most_common_word("Hello hello HELLO") == "hello"
    assert processor.most_common_word("one two three") == "one"  # all unique, first wins
```

The implementation includes:
- Type hints on all methods
- Brief docstrings explaining each method's behavior
- Proper handling of edge cases (empty strings, multiple spaces, punctuation)
- Case-insensitive operations where specified
- Support for negative shifts in Caesar cipher
- Tie-breaking logic for most common word (first occurrence wins)

The tests cover:
1. **reverse_words**: Multiple spacing scenarios, single word, empty string
2. **count_vowels**: Case-insensitivity, various vowel combinations, empty string
3. **is_palindrome**: Classic palindromes with punctuation/spaces, non-palindromes, empty string, numeric palindromes
4. **caesar_cipher**: Positive/negative/zero shifts, wrapping around alphabet, non-letter characters preserved
5. **most_common_word**: Basic frequency, tie-breaking, empty strings, case-insensitivity