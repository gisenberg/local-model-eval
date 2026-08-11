import re
from collections import Counter
from typing import Optional

class StringProcessor:
    """A class for processing and manipulating strings."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse each word in the string while maintaining word order.

        Args:
            s: Input string

        Returns:
            String with each word reversed
        """
        words = s.split()
        return ' '.join(word[::-1] for word in words)

    def count_vowels(self, s: str) -> int:
        """
        Count the number of vowels in a string (case-insensitive).

        Args:
            s: Input string

        Returns:
            Number of vowels (a, e, i, o, u)
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """
        Check if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: Input string

        Returns:
            True if the string is a palindrome, False otherwise
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s.lower())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher to a string with given shift.

        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)

        Returns:
            Encrypted/decrypted string
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26
                result.append(chr(base + shifted))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Find the most common word in a string (case-insensitive).

        Args:
            s: Input string

        Returns:
            Most common word, or None if string is empty.
            Returns first word alphabetically if there's a tie.
        """
        if not s.strip():
            return None

        words = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        if not words:
            return None

        word_counts = Counter(words)
        max_count = max(word_counts.values())
        most_common = [word for word, count in word_counts.items() if count == max_count]
        return min(most_common)  # First alphabetically if tied

# Tests
import pytest

class TestStringProcessor:
    def setup_method(self):
        self.processor = StringProcessor()

    def test_reverse_words(self):
        assert self.processor.reverse_words("hello world") == "olleh dlrow"
        assert self.processor.reverse_words("abc def ghi") == "cba fed ihg"
        assert self.processor.reverse_words("") == ""

    def test_count_vowels(self):
        assert self.processor.count_vowels("hello") == 2
        assert self.processor.count_vowels("HELLO") == 2
        assert self.processor.count_vowels("xyz") == 0

    def test_is_palindrome(self):
        assert self.processor.is_palindrome("A man, a plan, a canal: Panama") is True
        assert self.processor.is_palindrome("race a car") is False
        assert self.processor.is_palindrome("") is True

    def test_caesar_cipher(self):
        assert self.processor.caesar_cipher("abc", 1) == "bcd"
        assert self.processor.caesar_cipher("XYZ", 1) == "YZA"
        assert self.processor.caesar_cipher("abc", -1) == "zab"
        assert self.processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"

    def test_most_common_word(self):
        assert self.processor.most_common_word("the quick brown fox jumps over the lazy dog the") == "the"
        assert self.processor.most_common_word("apple apple banana banana") == "apple"
        assert self.processor.most_common_word("") is None
        assert self.processor.most_common_word("123 456") is None