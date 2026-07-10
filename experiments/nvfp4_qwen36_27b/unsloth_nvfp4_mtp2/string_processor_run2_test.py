import re
from collections import Counter
from typing import Optional

class StringProcessor:
    """
    A utility class for various string processing operations.
    """

    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in the string.

        Args:
            s: The input string.

        Returns:
            A string with words in reverse order.
        """
        # split() handles multiple spaces automatically by collapsing them
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """
        Counts the number of vowels in the string (case-insensitive).

        Args:
            s: The input string.

        Returns:
            The count of vowels (a, e, i, o, u).
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Checks if the string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: The input string.

        Returns:
            True if the cleaned string is a palindrome, False otherwise.
        """
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies a Caesar cipher shift to alphabetic characters.
        Supports negative shifts. Non-alphabetic characters remain unchanged.

        Args:
            s: The input string.
            shift: The number of positions to shift.

        Returns:
            The encrypted string.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                # Shift lowercase letters
                shifted = (ord(char) - ord('a') + shift) % 26
                result.append(chr(shifted + ord('a')))
            elif 'A' <= char <= 'Z':
                # Shift uppercase letters
                shifted = (ord(char) - ord('A') + shift) % 26
                result.append(chr(shifted + ord('A')))
            else:
                # Keep non-alphabetic characters as is
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Finds the most common word in the string (case-insensitive).
        If there is a tie, returns the word that appeared first.

        Args:
            s: The input string.

        Returns:
            The most common word, or None if the string is empty/has no words.
        """
        if not s or not s.strip():
            return None

        # Extract words (alphanumeric sequences), ignoring punctuation
        words = re.findall(r'\b\w+\b', s.lower())

        if not words:
            return None

        # Counter preserves insertion order for ties in Python 3.7+
        counter = Counter(words)
        # most_common(1) returns a list of tuples [(word, count)]
        return counter.most_common(1)[0][0]


# --- Pytest Tests ---

import pytest

def test_reverse_words():
    processor = StringProcessor()
    assert processor.reverse_words("Hello World") == "World Hello"
    assert processor.reverse_words("  Multiple   Spaces  ") == "Spaces Multiple"
    assert processor.reverse_words("Single") == "Single"

def test_count_vowels():
    processor = StringProcessor()
    assert processor.count_vowels("Hello") == 2
    assert processor.count_vowels("AEIOU") == 5
    assert processor.count_vowels("bcdfg") == 0
    assert processor.count_vowels("aEiOu") == 5

def test_is_palindrome():
    processor = StringProcessor()
    assert processor.is_palindrome("Racecar") == True
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
    assert processor.is_palindrome("Hello") == False
    assert processor.is_palindrome("Was it a car or a cat I saw?") == True

def test_caesar_cipher():
    processor = StringProcessor()
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("ABC", -1) == "ZAB"
    assert processor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert processor.caesar_cipher("z", -1) == "y"

def test_most_common_word():
    processor = StringProcessor()
    assert processor.most_common_word("apple banana apple") == "apple"
    assert processor.most_common_word("The the THE") == "the"
    assert processor.most_common_word("") is None
    # Tie-breaking: 'apple' appears first
    assert processor.most_common_word("apple banana apple banana") == "apple"