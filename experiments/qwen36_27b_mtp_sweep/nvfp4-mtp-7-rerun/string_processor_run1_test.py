import re
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the string.
        Multiple consecutive whitespace characters are treated as a single delimiter."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = [char.lower() for char in s if char.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher to alphabetic characters, preserving case.
        Supports positive and negative shifts. Non-alphabetic characters remain unchanged."""
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= char <= 'Z':
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequent word in the string (case-insensitive).
        If multiple words share the highest frequency, return the first one encountered."""
        words = re.findall(r'[a-zA-Z0-9]+', s.lower())
        if not words:
            return None

        counts: dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1

        max_count = max(counts.values())
        # Iterate through original order to preserve tie-breaking rule
        for word in words:
            if counts[word] == max_count:
                return word
        return None

import pytest

class TestStringProcessor:
    @pytest.fixture
    def sp(self):
        return StringProcessor()

    def test_reverse_words(self, sp):
        assert sp.reverse_words("hello world") == "world hello"
        assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
        assert sp.reverse_words("") == ""

    def test_count_vowels(self, sp):
        assert sp.count_vowels("Hello World") == 3
        assert sp.count_vowels("rhythm") == 0
        assert sp.count_vowels("AEIOU aeiou") == 10

    def test_is_palindrome(self, sp):
        assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
        assert sp.is_palindrome("racecar") is True
        assert sp.is_palindrome("hello") is False
        assert sp.is_palindrome("") is True

    def test_caesar_cipher(self, sp):
        assert sp.caesar_cipher("abc XYZ", 1) == "bcd YZA"
        assert sp.caesar_cipher("abc XYZ", -1) == "zab WXY"
        assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
        assert sp.caesar_cipher("a", 26) == "a"  # Full rotation

    def test_most_common_word(self, sp):
        # Tie-breaking: 'apple' appears first
        assert sp.most_common_word("apple banana apple orange banana") == "apple"
        assert sp.most_common_word("the quick brown fox jumps over the lazy dog") == "the"
        assert sp.most_common_word("") is None
        assert sp.most_common_word("123 456 123") == "123"