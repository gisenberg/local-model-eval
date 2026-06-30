import re
from typing import Optional
from collections import Counter
import pytest


class StringProcessor:
    """Utility class for common string manipulation tasks."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the string.
        
        Multiple spaces are normalized to single spaces.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome.
        
        Ignores case, spaces, and punctuation.
        """
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher to the string with the given shift.
        
        Only affects a-z/A-Z characters. Non-alphabetic characters are unchanged.
        Supports negative shifts.
        """
        result = []
        normalized_shift = shift % 26
        for char in s:
            if 'a' <= char <= 'z':
                result.append(chr((ord(char) - ord('a') + normalized_shift) % 26 + ord('a')))
            elif 'A' <= char <= 'Z':
                result.append(chr((ord(char) - ord('A') + normalized_shift) % 26 + ord('A')))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Find the most common word in the string (case-insensitive).
        
        Returns the first word encountered in case of a tie.
        Returns None if the string contains no valid words.
        """
        if not s.strip():
            return None
            
        # Extract alphabetic words only
        words = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        if not words:
            return None
            
        counts = Counter(words)
        max_count = max(counts.values())
        
        # Return first word in original order that matches the max count
        for word in words:
            if counts[word] == max_count:
                return word
        return None


# ==================== PYTEST TESTS ====================

class TestStringProcessor:
    @pytest.fixture
    def processor(self):
        return StringProcessor()

    def test_reverse_words(self, processor):
        assert processor.reverse_words("hello world") == "world hello"
        assert processor.reverse_words("  multiple   spaces  ") == "spaces multiple"
        assert processor.reverse_words("") == ""

    def test_count_vowels(self, processor):
        assert processor.count_vowels("Hello World") == 3
        assert processor.count_vowels("rhythm") == 0
        assert processor.count_vowels("AEIOU") == 5

    def test_is_palindrome(self, processor):
        assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
        assert processor.is_palindrome("racecar") is True
        assert processor.is_palindrome("hello") is False
        assert processor.is_palindrome("") is True

    def test_caesar_cipher(self, processor):
        assert processor.caesar_cipher("abc XYZ", 3) == "def ABC"
        assert processor.caesar_cipher("def", -3) == "abc"
        assert processor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
        assert processor.caesar_cipher("abc", 26) == "abc"

    def test_most_common_word(self, processor):
        assert processor.most_common_word("apple banana apple orange banana") == "apple"
        assert processor.most_common_word("the quick brown fox jumps over the lazy dog") == "the"
        assert processor.most_common_word("") is None
        assert processor.most_common_word("123 !@#") is None