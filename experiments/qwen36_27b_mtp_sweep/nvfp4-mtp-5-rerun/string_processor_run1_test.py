import re
from typing import Optional


class StringProcessor:
    """Utility class for common string processing operations."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the string."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher to the string with the given shift.
        
        Only alphabetic characters (a-z/A-Z) are shifted. All other characters 
        remain unchanged. Supports negative shifts.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted_char = chr((ord(char) - base + shift) % 26 + base)
                result.append(shifted_char)
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most frequent word in the string (case-insensitive).
        
        Returns the first word encountered in case of a tie.
        Returns None if the string contains no words.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None

        counts: dict[str, int] = {}
        max_count = 0
        most_common: Optional[str] = None

        for word in words:
            counts[word] = counts.get(word, 0) + 1
            if counts[word] > max_count:
                max_count = counts[word]
                most_common = word
        return most_common


# ==================== PYTEST TESTS ====================

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("") == ""

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("Rhythm") == 0
    assert sp.count_vowels("AEIOU") == 5

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("Hello") is False

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert sp.caesar_cipher("abc", -1) == "zab"
    assert sp.caesar_cipher("Test 123!", 26) == "Test 123!"

def test_most_common_word():
    sp = StringProcessor()
    # Standard case
    assert sp.most_common_word("apple banana apple orange") == "apple"
    # Tie-breaking: returns first encountered
    assert sp.most_common_word("apple banana apple banana") == "apple"
    # Edge cases
    assert sp.most_common_word("") is None
    assert sp.most_common_word("   !@#   ") is None