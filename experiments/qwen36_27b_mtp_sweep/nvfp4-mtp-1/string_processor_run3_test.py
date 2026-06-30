import re
from typing import Optional


class StringProcessor:
    """Utility class for common string processing operations."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the given string."""
        return ' '.join(s.split())

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters, preserving case.
        Non-alphabetic characters remain unchanged. Supports negative shifts."""
        result = []
        for char in s:
            if char.isupper():
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            elif char.islower():
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most common word in the string (case-insensitive).
        Returns the first word encountered if there's a tie. Returns None if no words."""
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        
        counts: dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
            
        max_count = max(counts.values())
        # Iterate through original order to guarantee "first if tied" behavior
        for word in words:
            if counts[word] == max_count:
                return word


# ==================== PYTEST TESTS ====================

def test_reverse_words() -> None:
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "multiple spaces"
    assert sp.reverse_words("single") == "single"

def test_count_vowels() -> None:
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOU aeiou") == 10
    assert sp.count_vowels("bcdfg") == 0
    assert sp.count_vowels("") == 0

def test_is_palindrome() -> None:
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False
    assert sp.is_palindrome("") is True  # Empty string is technically a palindrome

def test_caesar_cipher() -> None:
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("xyz", -1) == "wxy"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert sp.caesar_cipher("ABC", -3) == "XYZ"
    assert sp.caesar_cipher("a", 26) == "a"  # Full rotation

def test_most_common_word() -> None:
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat bird dog") == "cat"  # Tie, returns first encountered
    assert sp.most_common_word("Hello WORLD hello") == "hello"
    assert sp.most_common_word("") is None
    assert sp.most_common_word("123 !@#") is None