import re
import string
from collections import Counter
from typing import Optional

class StringProcessor:
    """
    A utility class for processing strings.
    """

    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in a string.

        Args:
            s: The input string.

        Returns:
            A string with the words in reversed order.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """
        Counts the number of vowels in a string (case-insensitive).

        Args:
            s: The input string.

        Returns:
            The number of vowels in the string.
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """
        Checks if a string is a palindrome (ignoring case, spaces, and punctuation).

        Args:
            s: The input string.

        Returns:
            True if the string is a palindrome, False otherwise.
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies a Caesar cipher to a string (a-z/A-Z only, supports negative shifts).

        Args:
            s: The input string.
            shift: The number of positions to shift each character.

        Returns:
            The encrypted string.
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
        """
        Finds the most common word in a string (case-insensitive, first if tied).

        Args:
            s: The input string.

        Returns:
            The most common word, or None if the string is empty.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        return Counter(words).most_common(1)[0][0]


# Pytest tests
def test_reverse_words():
    assert StringProcessor().reverse_words("hello world") == "world hello"
    assert StringProcessor().reverse_words("a b c") == "c b a"

def test_count_vowels():
    assert StringProcessor().count_vowels("Hello World") == 3
    assert StringProcessor().count_vowels("AEIOUaeiou") == 10

def test_is_palindrome():
    assert StringProcessor().is_palindrome("A man a plan a canal Panama") == True
    assert StringProcessor().is_palindrome("hello") == False

def test_caesar_cipher():
    assert StringProcessor().caesar_cipher("hello", 1) == "ifmmp"
    assert StringProcessor().caesar_cipher("hello", -1) == "gdkkn"

def test_most_common_word():
    assert StringProcessor().most_common_word("hello world hello") == "hello"
    assert StringProcessor().most_common_word("a a b b") == "a"
    assert StringProcessor().most_common_word("") is None