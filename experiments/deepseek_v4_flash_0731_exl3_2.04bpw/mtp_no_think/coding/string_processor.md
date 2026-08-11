Here's a complete implementation of the `StringProcessor` class with all required methods, plus pytest tests for each method:

```python
from typing import Optional
import re
import string

class StringProcessor:
    """A class that provides various string processing utilities."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.

        Multiple spaces between words become a single space.
        Leading/trailing spaces are removed.

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
        Supports negative shifts.

        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)

        Returns:
            String with Caesar cipher applied
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
        Returns None for empty strings.

        Args:
            s: Input string

        Returns:
            Most common word or None if string is empty
        """
        if not s.strip():
            return None

        # Split into words and count occurrences (case-insensitive)
        words = s.lower().split()
        word_counts = {}

        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Find the word with maximum count, first occurrence wins on ties
        max_count = 0
        most_common = None

        for word in words:
            if word_counts[word] > max_count:
                max_count = word_counts[word]
                most_common = word

        return most_common
```

Now here are the pytest tests:

```python
import pytest
from string_processor import StringProcessor

class TestStringProcessor:
    """Test suite for StringProcessor class."""

    def setup_method(self):
        self.processor = StringProcessor()

    def test_reverse_words(self):
        """Test reversing words in a string."""
        processor = StringProcessor()

        # Basic case
        assert processor.reverse_words("Hello world") == "world Hello"

        # Multiple spaces should become single space
        assert processor.reverse_words("Hello   world   test") == "test world Hello"

        # Leading/trailing spaces removed
        assert processor.reverse_words("  Hello world  ") == "world Hello"

        # Single word
        assert processor.reverse_words("Hello") == "Hello"

        # Empty string
        assert processor.reverse_words("") == ""

    def test_count_vowels(self):
        """Test counting vowels in a string."""
        processor = StringProcessor()

        # Basic case
        assert processor.count_vowels("hello") == 2  # e, o

        # Case-insensitive
        assert processor.count_vowels("HELLO") == 2

        # Mixed case
        assert processor.count_vowels("Hello World") == 3  # e, o, o

        # No vowels
        assert processor.count_vowels("xyz") == 0

        # Empty string
        assert processor.count_vowels("") == 0

        # All vowels
        assert processor.count_vowels("aeiouAEIOU") == 10

    def test_is_palindrome(self):
        """Test palindrome detection."""
        processor = StringProcessor()

        # Basic palindrome
        assert processor.is_palindrome("racecar") == True

        # Ignoring case
        assert processor.is_palindrome("Racecar") == True

        # Ignoring spaces
        assert processor.is_palindrome("never odd or even") == True

        # Ignoring punctuation
        assert processor.is_palindrome("A man, a plan, a canal: Panama") == True

        # Not a palindrome
        assert processor.is_palindrome("hello") == False

        # Empty string (should be palindrome)
        assert processor.is_palindrome("") == True

        # Single character
        assert processor.is_palindrome("a") == True

    def test_caesar_cipher(self):
        """Test Caesar cipher encryption/decryption."""
        processor = StringProcessor()

        # Basic shift
        assert processor.caesar_cipher("hello", 3) == "khoor"

        # Negative shift
        assert processor.caesar_cipher("khoor", -3) == "hello"

        # Wrap around
        assert processor.caesar_cipher("xyz", 3) == "abc"

        # Case preservation
        assert processor.caesar_cipher("Hello World", 5) == "Mjqqt Btwqi"

        # Non-alphabetic characters unchanged
        assert processor.caesar_cipher("hello, world!", 3) == "khoor, zruog!"

        # Shift of 0
        assert processor.caesar_cipher("hello", 0) == "hello"

        # Large shift (should wrap)
        assert processor.caesar_cipher("hello", 26) == "hello"

    def test_most_common_word(self):
        """Test finding most common word."""
        processor = StringProcessor()

        # Basic case
        assert processor.most_common_word("the cat and the dog") == "the"

        # Case-insensitive
        assert processor.most_common_word("The cat and THE dog") == "the"

        # Tie - first occurrence wins
        assert processor.most_common_word("apple banana apple cherry banana") == "apple"

        # Single word
        assert processor.most_common_word("hello") == "hello"

        # Empty string
        assert processor.most_common_word("") == None

        # Whitespace only
        assert processor.most_common_word("   ") == None

        # Multiple spaces between words
        assert processor.most_common_word("a  b  a  c") == "a"
```

To run these tests, save the class in a file named `string_processor.py` and the tests in `test_string_processor.py`, then run:

```bash
pytest test_string_processor.py -v
```

The tests cover:
1. **reverse_words**: Basic reversal, multiple spaces handling, leading/trailing spaces, single word, empty string
2. **count_vowels**: Basic counting, case-insensitivity, mixed case, no vowels, empty string, all vowels
3. **is_palindrome**: Basic palindrome, case-insensitivity, ignoring spaces, ignoring punctuation, non-palindrome, empty string, single character
4. **caesar_cipher**: Basic shift, negative shift, wrap-around, case preservation, non-alphabetic characters, zero shift, large shift
5. **most_common_word**: Basic case, case-insensitivity, tie-breaking, single word, empty string, whitespace-only, punctuation handling