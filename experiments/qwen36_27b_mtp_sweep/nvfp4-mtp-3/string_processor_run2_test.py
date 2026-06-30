import re
from typing import Optional
from collections import Counter


class StringProcessor:
    """Utility class for common string manipulation and analysis tasks."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the given string.
        
        Args:
            s: Input string containing words separated by whitespace.
            
        Returns:
            String with word order reversed. Multiple spaces are normalized.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            Integer count of vowels (a, e, i, o, u).
        """
        return sum(1 for c in s.lower() if c in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = [c.lower() for c in s if c.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher to the string. Only a-z/A-Z are shifted; supports negative shifts.
        
        Args:
            s: Input string.
            shift: Number of positions to shift. Negative values shift backwards.
            
        Returns:
            Ciphered string with non-alphabetic characters unchanged.
        """
        result = []
        for c in s:
            if c.isalpha():
                base = ord('a') if c.islower() else ord('A')
                result.append(chr((ord(c) - base + shift) % 26 + base))
            else:
                result.append(c)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Find the most common word in the string (case-insensitive).
        
        If multiple words share the highest frequency, returns the one that 
        appears first in the original string.
        
        Args:
            s: Input string.
            
        Returns:
            The most frequent word, or None if no words are found.
        """
        words = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        if not words:
            return None
            
        counts = Counter(words)
        max_count = max(counts.values())
        
        # Return the first word that reaches the max count
        for w in words:
            if counts[w] == max_count:
                return w


# ==================== PYTEST TESTS ====================

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("") == ""


def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("xyz") == 0
    assert sp.count_vowels("AEIOU") == 5


def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race a car") is False
    assert sp.is_palindrome("") is True


def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("xyz", -1) == "wxy"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"


def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("a b c") == "a"  # First if tied
    assert sp.most_common_word("") is None
    assert sp.most_common_word("123 !@#") is None