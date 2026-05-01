import re
import string
from collections import Counter
from typing import Optional

class StringProcessor:
    """
    A class to process strings in various ways.
    """

    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in a string.

        :param s: The input string.
        :return: The string with reversed word order.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """
        Counts the number of vowels in a string (case-insensitive).

        :param s: The input string.
        :return: The number of vowels in the string.
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """
        Checks if a string is a palindrome (ignoring case, spaces, and punctuation).

        :param s: The input string.
        :return: True if the string is a palindrome, False otherwise.
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies a Caesar cipher to a string (only affects a-z/A-Z characters).

        :param s: The input string.
        :param shift: The number of positions to shift each character.
        :return: The Caesar-ciphered string.
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
        Finds the most common word in a string (case-insensitive).

        :param s: The input string.
        :return: The most common word, or None if the string is empty.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        return Counter(words).most_common(1)[0][0]


# Pytest tests
def test_reverse_words():
    assert StringProcessor().reverse_words("hello world") == "world hello"

def test_count_vowels():
    assert StringProcessor().count_vowels("Hello World") == 3

def test_is_palindrome():
    assert StringProcessor().is_palindrome("A man, a plan, a canal: Panama") == True

def test_caesar_cipher():
    assert StringProcessor().caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"

def test_most_common_word():
    assert StringProcessor().most_common_word("Hello hello world") == "hello"