import re
from typing import Optional


class StringProcessor:
    """Utility class for common string manipulation and analysis tasks."""

    def reverse_words(self, s: str) -> str:
        """Reverses the order of words in the given string.
        
        Args:
            s: Input string containing words separated by whitespace.
            
        Returns:
            String with words in reversed order, separated by single spaces.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Counts the number of vowels in the string (case-insensitive).
        
        Args:
            s: Input string to analyze.
            
        Returns:
            Total count of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Checks if the string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string to check.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = [char.lower() for char in s if char.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Applies a Caesar cipher shift to alphabetic characters only.
        
        Args:
            s: Input string to encrypt/decrypt.
            shift: Number of positions to shift. Supports negative values.
            
        Returns:
            Transformed string with only a-z/A-Z characters shifted.
        """
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
        """Returns the most common word in the string (case-insensitive).
        
        Args:
            s: Input string to analyze.
            
        Returns:
            The most frequent word. Returns the first encountered word if there's a tie.
            Returns None if the string contains no words.
        """
        # Extract words (alphanumeric sequences), convert to lowercase
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None

        counts = {}
        max_count = 0
        most_common = None

        for word in words:
            counts[word] = counts.get(word, 0) + 1
            # Strictly greater ensures we keep the first word in case of a tie
            if counts[word] > max_count:
                max_count = counts[word]
                most_common = word

        return most_common

import pytest

@pytest.fixture
def sp():
    return StringProcessor()

class TestStringProcessor:
    def test_reverse_words(self, sp):
        assert sp.reverse_words("hello world") == "world hello"
        assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
        assert sp.reverse_words("single") == "single"

    def test_count_vowels(self, sp):
        assert sp.count_vowels("Hello World") == 3
        assert sp.count_vowels("rhythm") == 0
        assert sp.count_vowels("AEIOU aeiou") == 10

    def test_is_palindrome(self, sp):
        assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
        assert sp.is_palindrome("racecar") is True
        assert sp.is_palindrome("hello") is False
        assert sp.is_palindrome("") is True

    def test_caesar_cipher(self, sp):
        assert sp.caesar_cipher("abc", 1) == "bcd"
        assert sp.caesar_cipher("XYZ", 3) == "ABC"
        assert sp.caesar_cipher("Hello, World!", -1) == "Gdkkn, Vnqkc!"
        assert sp.caesar_cipher("a", -1) == "z"

    def test_most_common_word(self, sp):
        assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
        # Tie scenario: 'cat' appears first, so it should be returned
        assert sp.most_common_word("cat dog cat bird dog") == "cat"
        assert sp.most_common_word("") is None
        assert sp.most_common_word("123 !@#") is None