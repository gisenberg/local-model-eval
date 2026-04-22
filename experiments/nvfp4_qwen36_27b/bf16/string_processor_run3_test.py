import re
from typing import Optional

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in a string.
        
        Args:
            s: Input string.
            
        Returns:
            String with words in reversed order. Multiple spaces are normalized.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in a string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            Total count of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if a string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher to a string, shifting only a-z/A-Z characters.
        
        Args:
            s: Input string.
            shift: Number of positions to shift. Supports negative values.
            
        Returns:
            Ciphered string with non-alphabetic characters unchanged.
        """
        result = []
        shift = shift % 26  # Normalize shift to [0, 25]
        for char in s:
            if 'a' <= char <= 'z':
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= char <= 'Z':
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Find the most common word in a string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            The most frequent word. Returns the first encountered word in case of a tie.
            Returns None if the string contains no words.
        """
        if not s.strip():
            return None
            
        # Extract alphanumeric words, convert to lowercase
        words = re.findall(r'\b[a-zA-Z0-9]+\b', s.lower())
        if not words:
            return None

        counts = {}
        max_count = 0
        most_common = None
        
        for word in words:
            counts[word] = counts.get(word, 0) + 1
            # Strictly greater ensures we keep the first word that reached the max count
            if counts[word] > max_count:
                max_count = counts[word]
                most_common = word
                
        return most_common


# ========================
# Pytest Tests
# ========================
import pytest

def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert StringProcessor.reverse_words("single") == "single"

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("AEIOU aeiou") == 10
    assert StringProcessor.count_vowels("bcdfg") == 0

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") == True
    assert StringProcessor.is_palindrome("race a car") == False
    assert StringProcessor.is_palindrome("No 'x' in Nixon") == True

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("bcd YZA", -1) == "abc XYZ"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert StringProcessor.caesar_cipher("Test 123!", 0) == "Test 123!"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat bird dog") == "cat"  # Tie: cat appears first
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("123 !@#") is None