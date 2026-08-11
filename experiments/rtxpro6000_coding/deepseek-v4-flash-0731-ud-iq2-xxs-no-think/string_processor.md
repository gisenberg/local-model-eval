Here's the complete implementation with tests:

```python
from typing import Optional
import re
from collections import Counter

class StringProcessor:
    """A class to process strings with various utility methods."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.

        Args:
            s: Input string

        Returns:
            String with words reversed, single spaces between words,
            no leading/trailing spaces.
        """
        # Split by whitespace and filter empty strings, then reverse
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
        vowels = set('aeiou')
        return sum(1 for char in s.lower() if char in vowels)

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

        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)

        Returns:
            String with letters shifted, other characters unchanged
        """
        result = []
        for char in s:
            if char.isalpha():
                # Determine base (uppercase or lowercase)
                base = ord('A') if char.isupper() else ord('a')
                # Shift and wrap around
                shifted = (ord(char) - base + shift) % 26 + base
                result.append(chr(shifted))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Find the most frequently occurring word (case-insensitive).

        Args:
            s: Input string

        Returns:
            Most common word, or None if empty string.
            If tied, returns the one that appears first.
        """
        if not s.strip():
            return None

        # Split into words, convert to lowercase
        words = s.lower().split()

        # Count occurrences while maintaining first appearance order
        word_counts = Counter(words)

        # Find max count
        max_count = max(word_counts.values())

        # Return first word with max count
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

@pytest.fixture
def processor():
    return StringProcessor()

class TestStringProcessor:

    def test_reverse_words(self, processor):
        """Test reversing word order with various spacing scenarios."""
        # Basic case
        assert processor.reverse_words("Hello world") == "world Hello"

        # Multiple spaces between words
        assert processor.reverse_words("Hello   world   again") == "again world Hello"

        # Leading/trailing spaces
        assert processor.reverse_words("  Hello world  ") == "world Hello"

        # Single word
        assert processor.reverse_words("Hello") == "Hello"

        # Empty string
        assert processor.reverse_words("") == ""

    def test_count_vowels(self, processor):
        """Test vowel counting with various cases."""
        # Basic case
        assert processor.count_vowels("Hello") == 2  # e, o

        # Case-insensitive
        assert processor.count_vowels("AEIOU") == 5

        # Mixed case
        assert processor.count_vowels("aEiOu") == 5

        # No vowels
        assert processor.count_vowels("bcdfg") == 0

        # Empty string
        assert processor.count_vowels("") == 0

    def test_is_palindrome(self, processor):
        """Test palindrome checking with various inputs."""
        # Simple palindrome
        assert processor.is_palindrome("racecar") == True

        # Palindrome with spaces and punctuation
        assert processor.is_palindrome("A man, a plan, a canal: Panama") == True

        # Case-insensitive
        assert processor.is_palindrome("Racecar") == True

        # Not a palindrome
        assert processor.is_palindrome("hello") == False

        # Empty string (should be True as it's vacuously true)
        assert processor.is_palindrome("") == True

    def test_caesar_cipher(self, processor):
        """Test Caesar cipher with various shifts."""
        # Basic shift
        assert processor.caesar_cipher("Hello", 3) == "Khoor"

        # Negative shift
        assert processor.caesar_cipher("Khoor", -3) == "Hello"

        # Wrap around
        assert processor.caesar_cipher("xyz", 3) == "abc"

        # Mixed case and non-letters
        assert processor.caesar_cipher("Hello, World!", 1) == "Ifmmp, Xpsme!"

        # Large shift
        assert processor.caesar_cipher("abc", 26) == "abc"

        # Zero shift
        assert processor.caesar_cipher("Hello", 0) == "Hello"

    def test_most_common_word(self, processor):
        """Test finding most common word with various scenarios."""
        # Basic case
        assert processor.most_common_word("the cat and the dog") == "the"

        # Case-insensitive
        assert processor.most_common_word("The cat THE dog the") == "the"

        # Tie-breaking (first occurrence)
        assert processor.most_common_word("apple banana apple orange banana") == "apple"

        # Empty string
        assert processor.most_common_word("") is None

        # Single word
        assert processor.most_common_word("hello") == "hello"

        # Whitespace only
        assert processor.most_common_word("   ") is None
```

**Key features of the implementation:**

1. **`reverse_words`**: Uses `split()` which automatically handles multiple spaces and leading/trailing whitespace, then reverses the list.

2. **`count_vowels`**: Uses a set for O(1) vowel lookup and iterates through lowercase characters.

3. **`is_palindrome`**: Uses regex to remove non-alphanumeric characters and compares the cleaned string with its reverse.

4. **`caesar_cipher`**: Handles both uppercase and lowercase letters, wraps around correctly, and preserves non-alphabetic characters.

5. **`most_common_word`**: Uses `Counter` for efficient counting and iterates through the original word order to break ties by first appearance.

The tests cover edge cases like empty strings, multiple spaces, case sensitivity, punctuation, negative shifts, and tie-breaking scenarios.