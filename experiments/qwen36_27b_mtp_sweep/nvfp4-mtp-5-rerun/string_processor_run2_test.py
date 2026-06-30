import re
from typing import Optional

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in the string.
        Multiple consecutive spaces are collapsed to a single space."""
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for c in s.lower() if c in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters only.
        Supports positive and negative shifts. Non-alphabetic characters remain unchanged."""
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26
                result.append(chr(base + shifted))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Return the most common word in the string (case-insensitive).
        If multiple words are tied, returns the one that appears first in the original string."""
        if not s.strip():
            return None
            
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None

        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1

        max_count = max(counts.values())
        # Return first word in original order that matches the max count
        for w in words:
            if counts[w] == max_count:
                return w
        return None


# ========================
# Pytest Test Suite
# ========================

def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  one   two  three  ") == "three two one"
    assert StringProcessor.reverse_words("single") == "single"

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3  # e, o, o
    assert StringProcessor.count_vowels("rhythm") == 0
    assert StringProcessor.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("racecar") is True
    assert StringProcessor.is_palindrome("hello") is False
    assert StringProcessor.is_palindrome("Was it a car or a cat I saw?") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 3) == "def ABC"
    assert StringProcessor.caesar_cipher("abc", -1) == "zab"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert StringProcessor.caesar_cipher("Test 123!", 26) == "Test 123!"  # full rotation

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat dog") == "cat"  # tie -> first occurrence
    assert StringProcessor.most_common_word("dog cat dog cat") == "dog"  # tie -> first occurrence
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("123 !@#") is None