import re
from collections import Counter
from typing import Optional

class StringProcessor:
    """A utility class for advanced string manipulation and analysis."""

    def reverse_words(self, s: str) -> str:
        """Reverses the order of words in a string while preserving internal spacing."""
        return " ".join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Returns the total count of vowels (a, e, i, o, u) regardless of case."""
        vowels = "aeiou"
        return sum(1 for char in s.lower() if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """Checks if a string is a palindrome, ignoring case, spaces, and punctuation."""
        clean_s = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return clean_s == clean_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Shifts letters by the specified amount. 
        Preserves case and non-alphabetic characters. Supports negative shifts.
        """
        result = []
        for char in s:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                # Normalize shift to 0-25 range
                shifted = (ord(char) - start + shift) % 26
                result.append(chr(start + shifted))
            else:
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Returns the most frequent word (case-insensitive). 
        Returns the first occurring word in case of a tie.
        """
        if not s.strip():
            return None
        
        # Remove punctuation and split into lowercase words
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
            
        counts = Counter(words)
        # max() in Python is stable, returning the first occurrence in case of ties
        return max(words, key=lambda w: counts[w])

# --- Pytest Tests ---
import pytest

def test_string_processor():
    sp = StringProcessor()

    # 1. Test reverse_words
    assert sp.reverse_words("Hello World") == "World Hello"
    assert sp.reverse_words("  Python is fun  ") == "fun is Python"

    # 2. Test count_vowels
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOU aeiou") == 10
    assert sp.count_vowels("xyz") == 0

    # 3. Test is_palindrome
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("Racecar") is True
    assert sp.is_palindrome("Hello") is False

    # 4. Test caesar_cipher
    assert sp.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    assert sp.caesar_cipher("abc", -1) == "zab"
    assert sp.caesar_cipher("XYZ", 3) == "ABC"

    # 5. Test most_common_word
    assert sp.most_common_word("Apple banana apple orange") == "apple"
    assert sp.most_common_word("The cat sat on the mat") == "the"
    # Tie test: "apple" and "banana" both appear once; "apple" comes first.
    assert sp.most_common_word("apple banana") == "apple"
    assert sp.most_common_word("") is None

if __name__ == "__main__":
    # This allows running the tests via `python filename.py` if pytest isn't called from CLI
    pytest.main([__file__])