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

        # Find the word with maximum count, breaking ties by first occurrence
        max_count = max(word_counts.values())

        # Find the first word that has the max count
        for word in words:
            if word_counts[word] == max_count:
                return word

        return None
```

Now, here are the pytest tests:

```python
import pytest
from typing import Optional
from your_module import StringProcessor  # Replace with actual module name

class TestStringProcessor:
    """Test cases for StringProcessor class."""

    def setup_method(self):
        """Setup method to create a StringProcessor instance for each test."""
        self.processor = StringProcessor()

    def test_reverse_words(self):
        """Test reverse_words method."""
        # Basic test
        assert self.processor.reverse_words("Hello world") == "world Hello"

        # Multiple spaces between words
        assert self.processor.reverse_words("Hello   world   test") == "test world Hello"

        # Leading/trailing spaces
        assert self.processor.reverse_words("  Hello world  ") == "world Hello"

        # Single word
        assert self.processor.reverse_words("Hello") == "Hello"

        # Empty string
        assert self.processor.reverse_words("") == ""

        # String with only spaces
        assert self.processor.reverse_words("   ") == ""

    def test_count_vowels(self):
        """Test count_vowels method."""
        # Basic test
        assert self.processor.count_vowels("Hello") == 2  # e, o

        # Case-insensitive
        assert self.processor.count_vowels("AEIOU") == 5
        assert self.processor.count_vowels("aeiou") == 5

        # Mixed case
        assert self.processor.count_vowels("Hello World") == 3  # e, o, o

        # No vowels
        assert self.processor.count_vowels("bcdfg") == 0

        # Empty string
        assert self.processor.count_vowels("") == 0

        # String with numbers and symbols
        assert self.processor.count_vowels("Hello123!") == 2

    def test_is_palindrome(self):
        """Test is_palindrome method."""
        # Basic palindrome
        assert self.processor.is_palindrome("racecar") == True

        # Palindrome with spaces
        assert self.processor.is_palindrome("A man a plan a canal Panama") == True

        # Palindrome with punctuation
        assert self.processor.is_palindrome("Never odd or even") == True

        # Palindrome with mixed case
        assert self.processor.is_palindrome("Madam") == True

        # Not a palindrome
        assert self.processor.is_palindrome("Hello") == False

        # Empty string (considered palindrome)
        assert self.processor.is_palindrome("") == True

        # Palindrome with numbers
        assert self.processor.is_palindrome("12321") == True

    def test_caesar_cipher(self):
        """Test caesar_cipher method."""
        # Basic shift
        assert self.processor.caesar_cipher("Hello", 1) == "Ifmmp"

        # Negative shift
        assert self.processor.caesar_cipher("Hello", -1) == "Gdkkn"

        # Wrap around
        assert self.processor.caesar_cipher("Zebra", 3) == "Cheud"

        # Mixed case
        assert self.processor.caesar_cipher("Hello World", 5) == "Mjqqt Btwqi"

        # Non-alphabetic characters unchanged
        assert self.processor.caesar_cipher("Hello123!", 2) == "Jgnnq123!"

        # Shift of 0
        assert self.processor.caesar_cipher("Hello", 0) == "Hello"

        # Large shift
        assert self.processor.caesar_cipher("Hello", 26) == "Hello"

        # Negative wrap around
        assert self.processor.caesar_cipher("abc", -3) == "xyz"

    def test_most_common_word(self):
        """Test most_common_word method."""
        # Basic test
        assert self.processor.most_common_word("the quick brown fox the") == "the"

        # Case-insensitive
        assert self.processor.most_common_word("The quick brown fox the") == "the"

        # Tie-breaking (first occurrence)
        assert self.processor.most_common_word("apple banana apple banana") == "apple"

        # Single word
        assert self.processor.most_common_word("hello") == "hello"

        # Empty string
        assert self.processor.most_common_word("") is None

        # String with only spaces
        assert self.processor.most_common_word("   ") is None

        # Multiple words with different frequencies
        assert self.processor.most_common_word("a b c a b a") == "a"

        # Words with punctuation
        assert self.processor.most_common_word("hello, world! hello") == "hello"
```

These tests cover:

1. **reverse_words**: Tests basic reversal, multiple spaces, leading/trailing spaces, single words, and empty strings.
2. **count_vowels**: Tests basic counting, case-insensitivity, mixed cases, no vowels, empty strings, and strings with numbers/symbols.
3. **is_palindrome**: Tests basic palindromes, palindromes with spaces and punctuation, mixed case, non-palindromes, empty strings, and palindromes with numbers.
4. **caesar_cipher**: Tests basic shifts, negative shifts, wrap-around, mixed case, non-alphabetic characters, zero shift, large shifts, and negative wrap-around.
5. **most_common_word**: Tests basic frequency, case-insensitivity, tie-breaking, single words, empty strings, strings with only spaces, and words with punctuation.

Each test method has multiple assertions to thoroughly test the functionality of each method.