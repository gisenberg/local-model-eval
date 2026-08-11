from typing import Optional
import re
from collections import Counter
import string


class StringProcessor:
    """A class for processing and manipulating strings."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.

        Args:
            s: Input string

        Returns:
            String with words in reversed order
        """
        words = s.split()
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count the number of vowels in a string (case-insensitive).

        Args:
            s: Input string

        Returns:
            Number of vowels in the string
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if a string is a palindrome, ignoring case, spaces, and punctuation.

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
        Apply Caesar cipher to a string.

        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)

        Returns:
            Encrypted/decrypted string
        """
        result = []
        for char in s:
            if char.isalpha():
                # Determine if uppercase or lowercase
                base = ord('A') if char.isupper() else ord('a')
                # Apply shift and wrap around
                shifted = (ord(char) - base + shift) % 26 + base
                result.append(chr(shifted))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Find the most common word in a string (case-insensitive).

        Args:
            s: Input string

        Returns:
            Most common word in lowercase, or None if string is empty
        """
        if not s.strip():
            return None

        # Extract words (alphanumeric sequences)
        words = re.findall(r'[a-zA-Z0-9]+', s.lower())
        if not words:
            return None

        word_counts = Counter(words)
        max_count = word_counts.most_common(1)[0][1]

        # Return the first word that has the maximum count
        for word in words:
            if word_counts[word] == max_count:
                return word

        return None


# Tests
import pytest


class TestStringProcessor:
    """Test cases for StringProcessor class."""

    def setup_method(self):
        """Set up test fixture."""
        self.processor = StringProcessor()

    def test_reverse_words(self):
        """Test reversing words in a string."""
        assert self.processor.reverse_words("hello world") == "world hello"
        assert self.processor.reverse_words("a b c") == "c b a"
        assert self.processor.reverse_words("single") == "single"
        assert self.processor.reverse_words("") == ""

    def test_count_vowels(self):
        """Test counting vowels in a string."""
        assert self.processor.count_vowels("hello world") == 3
        assert self.processor.count_vowels("HELLO") == 2
        assert self.processor.count_vowels("xyz") == 0
        assert self.processor.count_vowels("aeiou") == 5

    def test_is_palindrome(self):
        """Test palindrome detection."""
        assert self.processor.is_palindrome("A man, a plan, a canal: Panama") is True
        assert self.processor.is_palindrome("race a car") is False
        assert self.processor.is_palindrome("Was it a car or a cat I saw?") is True
        assert self.processor.is_palindrome("") is True

    def test_caesar_cipher(self):
        """Test Caesar cipher encryption/decryption."""
        assert self.processor.caesar_cipher("abc", 1) == "bcd"
        assert self.processor.caesar_cipher("ABC", 1) == "BCD"
        assert self.processor.caesar_cipher("xyz", 2) == "zab"
        assert self.processor.caesar_cipher("ABC", -1) == "ZAB"
        assert self.processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"

    def test_most_common_word(self):
        """Test finding most common word."""
        assert self.processor.most_common_word("hello world hello") == "hello"
        assert self.processor.most_common_word("Hello hello HELLO") == "hello"
        assert self.processor.most_common_word("") is None
        assert self.processor.most_common_word("   ") is None
        # Test tie-breaking (should return first occurrence)
        assert self.processor.most_common_word("cat dog cat dog") == "cat"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])