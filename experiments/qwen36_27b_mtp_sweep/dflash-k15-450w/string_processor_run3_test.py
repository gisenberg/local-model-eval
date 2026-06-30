import re
from typing import Optional
from collections import Counter

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in the given string.
        
        Args:
            s: Input string containing words.
            
        Returns:
            String with words in reversed order.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in the string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            Total count of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = re.sub(r'[^a-z0-9]', '', s.lower())
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher to alphabetic characters, preserving case and non-alphabetic characters.
        
        Args:
            s: Input string.
            shift: Number of positions to shift (supports negative values).
            
        Returns:
            Ciphered string.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('a') if char.islower() else ord('A')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Return the most common word in the string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            The most frequent word. Returns the first occurring word in case of a tie.
            Returns None if no words are found.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
            
        counts = Counter(words)
        max_count = max(counts.values())
        
        # Iterate in original order to guarantee "first if tied" behavior
        for word in words:
            if counts[word] == max_count:
                return word
        return None


# ========================
# Pytest Test Suite
# ========================
import pytest

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("single") == "single"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("rhythm") == 0
    assert sp.count_vowels("AEIOU aeiou") == 10

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False
    assert sp.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc XYZ", 3) == "def ABC"
    assert sp.caesar_cipher("def ABC", -3) == "abc XYZ"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert sp.caesar_cipher("z", -1) == "y"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat dog") == "cat"  # Tie: returns first occurring
    assert sp.most_common_word("The QUICK brown fox jumps over the lazy dog") == "the"
    assert sp.most_common_word("") is None
    assert sp.most_common_word("123 !@#") is None