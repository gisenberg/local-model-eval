from collections import Counter
import re
from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse each word in the string while maintaining word order."""
        return ' '.join(word[::-1] for word in s.split())

    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) case-insensitively."""
        return sum(1 for c in s.lower() if c in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if string is palindrome ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^a-z0-9]', '', s.lower())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters, preserving case."""
        result = []
        for c in s:
            if c.isalpha():
                base = ord('a') if c.islower() else ord('A')
                result.append(chr((ord(c) - base + shift) % 26 + base))
            else:
                result.append(c)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Find most common word (case-insensitive), return first if tied."""
        words = re.findall(r'[a-z]+', s.lower())
        if not words:
            return None
        counter = Counter(words)
        max_count = max(counter.values())
        for word in words:
            if counter[word] == max_count:
                return word
        return None

# Tests
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "olleh dlrow"
    assert sp.reverse_words("abc") == "cba"
    assert sp.reverse_words("") == ""

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("hello") == 2
    assert sp.count_vowels("HELLO") == 2
    assert sp.count_vowels("") == 0

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race a car") is False
    assert sp.is_palindrome("") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("ABC", -1) == "ZAB"
    assert sp.caesar_cipher("abc123", 1) == "bcd123"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple") == "apple"
    assert sp.most_common_word("a a b b") == "a"
    assert sp.most_common_word("") is None