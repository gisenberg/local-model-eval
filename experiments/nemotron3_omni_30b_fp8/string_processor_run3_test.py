import re
import string
from collections import Counter
from typing import Optional

class StringProcessor:
    """
    A utility class for processing strings with various operations.
    """

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Reverses the order of words in a string.

        Args:
            s (str): The input string.

        Returns:
            str: The string with words in reversed order.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Counts the number of vowels in a string (case-insensitive).

        Args:
            s (str): The input string.

        Returns:
            int: The number of vowels in the string.
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Checks if a string is a palindrome (ignoring case, spaces, and punctuation).

        Args:
            s (str): The input string.

        Returns:
            bool: True if the string is a palindrome, False otherwise.
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Applies a Caesar cipher to a string, shifting letters by a specified amount.

        Args:
            s (str): The input string.
            shift (int): The number of positions to shift letters (can be negative).

        Returns:
            str: The string with letters shifted.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Finds the most common word in a string (case-insensitive, first if tied).

        Args:
            s (str): The input string.

        Returns:
            Optional[str]: The most common word, or None if the string is empty.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        return Counter(words).most_common(1)[0][0]

# Pytest tests
def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("a b c") == "c b a"

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("AEIOUaeiou") == 10

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") == True
    assert StringProcessor.is_palindrome("race a car") == False

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc", 1) == "bcd"
    assert StringProcessor.caesar_cipher("xyz", -1) == "wxy"

def test_most_common_word():
    assert StringProcessor.most_common_word("Hello hello world") == "hello"
    assert StringProcessor.most_common_word("") is None