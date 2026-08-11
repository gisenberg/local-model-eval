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
        if not s.strip():
            return None

        # Split into words and convert to lowercase
        words = s.lower().split()

        # Count occurrences
        word_counts = Counter(words)

        # Find the most common word (Counter.most_common returns sorted by count,
        # then by insertion order for ties)
        most_common = word_counts.most_common(1)
        return most_common[0][0] if most_common else None
```

Now, here are the pytest tests:

```python
import pytest
from typing import Optional
from your_module import StringProcessor  # Replace with actual module name

class TestStringProcessor:
    """Test cases for StringProcessor class."""

    @pytest.fixture
    def processor(self):
        return StringProcessor()

    def test_reverse_words(self, processor):
        """Test reverse_words method."""
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

        # Multiple spaces only
        assert processor.reverse_words("   ") == ""

    def test_count_vowels(self, processor):
        """Test count_vowels method."""
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

        # With numbers and punctuation
        assert processor.count_vowels("Hello123!") == 2

    def test_is_palindrome(self, processor):
        """Test is_palindrome method."""
        # Basic palindrome
        assert processor.is_palindrome("racecar") == True

        # Case-insensitive
        assert processor.is_palindrome("Racecar") == True

        # With spaces
        assert processor.is_palindrome("A man a plan a canal Panama") == True

        # With punctuation
        assert processor.is_palindrome("Madam, I'm Adam") == True

        # Not a palindrome
        assert processor.is_palindrome("Hello") == False

        # Empty string (cleaned to empty, so palindrome)
        assert processor.is_palindrome("") == True

        # Only punctuation
        assert processor.is_palindrome("!!!") == True

    def test_caesar_cipher(self, processor):
        """Test caesar_cipher method."""
        # Basic shift
        assert processor.caesar_cipher("Hello", 1) == "Ifmmp"

        # Negative shift
        assert processor.caesar_cipher("Hello", -1) == "Gdkkn"

        # Wrap around
        assert processor.caesar_cipher("Zebra", 3) == "Cheud"

        # Mixed case
        assert processor.caesar_cipher("Hello World", 5) == "Mjqqt Btwqi"

        # Non-alphabetic characters unchanged
        assert processor.caesar_cipher("Hello123!", 2) == "Jgnnq123!"

        # Large shift
        assert processor.caesar_cipher("abc", 26) == "abc"

        # Shift with wrap for lowercase
        assert processor.caesar_cipher("xyz", 3) == "abc"

    def test_most_common_word(self, processor):
        """Test most_common_word method."""
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

        # Whitespace only
        assert processor.most_common_word("   ") is None

        # Multiple words with different frequencies
        assert processor.most_common_word("a b c a b a") == "a"
```

These tests cover:

1. **reverse_words**: Tests basic reversal, multiple spaces, leading/trailing spaces, single word, empty string, and whitespace-only strings.

2. **count_vowels**: Tests basic counting, case-insensitivity, mixed case, no vowels, empty string, and strings with numbers/punctuation.

3. **is_palindrome**: Tests basic palindromes, case-insensitivity, spaces, punctuation, non-palindromes, empty strings, and punctuation-only strings.

4. **caesar_cipher**: Tests basic shifts, negative shifts, wrap-around, mixed case, non-alphabetic characters, and large shifts.

5. **most_common_word**: Tests basic frequency, case-insensitivity, tie-breaking, single words, empty strings, and whitespace-only strings.

The implementation handles all edge cases mentioned in the requirements and includes proper type hints and docstrings for each method.