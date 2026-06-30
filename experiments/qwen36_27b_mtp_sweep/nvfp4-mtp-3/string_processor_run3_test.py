import re
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the string. Extra whitespace is normalized."""
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = [char.lower() for char in s if char.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher with the given shift. Only a-z/A-Z characters are shifted.
        Supports positive and negative shifts."""
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
        """Return the most frequent word (case-insensitive). 
        Returns the first occurring word in case of a tie. Returns None if no words exist."""
        words = re.findall(r'[a-z]+', s.lower())
        if not words:
            return None

        counts: dict = {}
        first_occurrence: dict = {}

        for idx, word in enumerate(words):
            counts[word] = counts.get(word, 0) + 1
            if word not in first_occurrence:
                first_occurrence[word] = idx

        max_count = max(counts.values())
        candidates = [w for w, c in counts.items() if c == max_count]
        return min(candidates, key=lambda w: first_occurrence[w])


# ========================
# Pytest Test Suite
# ========================
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("") == ""

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("rhythm") == 0
    assert sp.count_vowels("AEIOU") == 5

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race a car") is False
    assert sp.is_palindrome("Was it a car or a cat I saw?") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert sp.caesar_cipher("abc", -1) == "zab"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    # Tie-breaking: 'cat' and 'dog' both appear twice, but 'cat' appears first
    assert sp.most_common_word("cat dog cat bird dog") == "cat"
    assert sp.most_common_word("") is None
    assert sp.most_common_word("123 !@#") is None