import re
from typing import Optional

class StringProcessor:
    """A utility class for common string processing operations."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverses the order of words in a string.

        Args:
            s: The input string.

        Returns:
            A new string with words in reverse order.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Counts the number of vowels in a string (case-insensitive).

        Args:
            s: The input string.

        Returns:
            The count of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s if char.lower() in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Checks if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: The input string.

        Returns:
            True if the string is a palindrome, False otherwise.
        """
        cleaned = [c.lower() for c in s if c.isalnum()]
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Applies a Caesar cipher to a string.

        Only applies to a-z and A-Z characters. Supports negative shifts.
        Non-alphabetic characters remain unchanged.

        Args:
            s: The input string.
            shift: The number of positions to shift (can be negative).

        Returns:
            The encrypted string.
        """
        result = []
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
        """Finds the most common word in a string (case-insensitive).

        If multiple words share the highest frequency, the first one encountered
        in the string is returned.

        Args:
            s: The input string.

        Returns:
            The most common word, or None if the string contains no words.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None

        counts = {}
        max_count = 0
        most_common = None

        for word in words:
            current_count = counts.get(word, 0) + 1
            counts[word] = current_count
            if current_count > max_count:
                max_count = current_count
                most_common = word

        return most_common

import pytest

def test_reverse_words():
    """Test that reverse_words correctly reverses the order of words."""
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("a  b   c") == "c b a"
    assert StringProcessor.reverse_words("single") == "single"

def test_count_vowels():
    """Test that count_vowels correctly counts vowels case-insensitively."""
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("BCDF") == 0
    assert StringProcessor.count_vowels("aEiOu") == 5
    assert StringProcessor.count_vowels("") == 0

def test_is_palindrome():
    """Test that is_palindrome ignores case, spaces, and punctuation."""
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") == True
    assert StringProcessor.is_palindrome("racecar") == True
    assert StringProcessor.is_palindrome("hello") == False
    assert StringProcessor.is_palindrome("No 'x' in Nixon") == True

def test_caesar_cipher():
    """Test that caesar_cipher shifts alphabetic characters correctly."""
    # Positive shift
    assert StringProcessor.caesar_cipher("abc XYZ", 1) == "bcd YZA"
    assert StringProcessor.caesar_cipher("Hello, World! 123", 5) == "Mjqqt, Btwqi! 123"
    # Negative shift
    assert StringProcessor.caesar_cipher("bcd YZA", -1) == "abc XYZ"
    assert StringProcessor.caesar_cipher("a", -1) == "z"

def test_most_common_word():
    """Test that most_common_word returns the most frequent word (first if tied)."""
    # Normal case
    assert StringProcessor.most_common_word("the cat in the hat the") == "the"
    # Tie-breaker (first encountered should win)
    assert StringProcessor.most_common_word("apple pear apple pear") == "apple"
    # Empty / non-word string
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("!@#$%") is None