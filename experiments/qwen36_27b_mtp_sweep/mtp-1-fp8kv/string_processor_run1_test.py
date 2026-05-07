import re
from typing import Optional, Dict

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in a string.
        
        Args:
            s: Input string containing words separated by whitespace.
            
        Returns:
            String with words in reversed order, separated by single spaces.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in a string (case-insensitive).
        
        Args:
            s: Input string to analyze.
            
        Returns:
            Integer count of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if a string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string to check.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = [c.lower() for c in s if c.isalnum()]
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher to a string, shifting only a-z/A-Z characters.
        
        Args:
            s: Input string to encrypt/decrypt.
            shift: Number of positions to shift. Supports positive and negative values.
            
        Returns:
            Ciphered string with non-alphabetic characters unchanged.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                shifted = (ord(char) - ord('a') + shift) % 26 + ord('a')
                result.append(chr(shifted))
            elif 'A' <= char <= 'Z':
                shifted = (ord(char) - ord('A') + shift) % 26 + ord('A')
                result.append(chr(shifted))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Find the most common word in a string (case-insensitive).
        
        Args:
            s: Input string to analyze.
            
        Returns:
            The most frequent word. Returns the first encountered word in case of a tie.
            Returns None if the string contains no words.
        """
        words = re.findall(r'\b[a-z]+\b', s.lower())
        if not words:
            return None

        counts: Dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1

        max_count = max(counts.values())
        # Python 3.7+ dicts preserve insertion order, ensuring first-occurrence tie-breaking
        for word in counts:
            if counts[word] == max_count:
                return word
        return None


# ========================
# Pytest Tests
# ========================
import pytest

def test_reverse_words():
    assert StringProcessor.reverse_words("Hello World") == "World Hello"
    assert StringProcessor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert StringProcessor.reverse_words("") == ""

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("AEIOU aeiou") == 10
    assert StringProcessor.count_vowels("rhythm") == 0

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("race a car") is False
    assert StringProcessor.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("bcd YZA", -1) == "abc XYZ"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert StringProcessor.caesar_cipher("z", -1) == "y"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat dog") == "cat"  # Tie-break: first encountered
    assert StringProcessor.most_common_word("123 !@#") is None
    assert StringProcessor.most_common_word("") is None