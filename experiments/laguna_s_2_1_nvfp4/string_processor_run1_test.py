import re
from collections import Counter
from typing import Optional


class StringProcessor:
    """A class for various string processing operations."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string.

        Args:
            s: Input string

        Returns:
            String with words reversed in order
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count vowels in a string (case-insensitive).

        Args:
            s: Input string

        Returns:
            Number of vowels
        """
        return sum(1 for c in s.lower() if c in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if string is a palindrome (ignoring case, spaces, punctuation).

        Args:
            s: Input string

        Returns:
            True if palindrome, False otherwise
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to a string.

        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)

        Returns:
            Encrypted/decrypted string
        """
        result = []
        for c in s:
            if c.islower():
                result.append(chr((ord(c) - ord('a') + shift) % 26 + ord('a')))
            elif c.isupper():
                result.append(chr((ord(c) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(c)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Find the most common word in a string.

        Args:
            s: Input string

        Returns:
            Most common word (lowercase), or None if no words
        """
        words = re.findall(r'[a-zA-Z]+', s.lower())
        if not words:
            return None
        return Counter(words).most_common(1)[0][0]


# Tests
import pytest

processor = StringProcessor()

def test_reverse_words():
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert processor.reverse_words("") == ""

def test_count_vowels():
    assert processor.count_vowels("hello") == 2
    assert processor.count_vowels("HELLO") == 2
    assert processor.count_vowels("xyz") == 0

def test_is_palindrome():
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("race a car") is False
    assert processor.is_palindrome("Was it a car or a cat I saw?") is True

def test_caesar_cipher():
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("ABC", 1) == "BCD"
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("xyz", 3) == "abc"

def test_most_common_word():
    assert processor.most_common_word("Hello hello world") == "hello"
    assert processor.most_common_word("a b c a b a") == "a"
    assert processor.most_common_word("123 456") is None