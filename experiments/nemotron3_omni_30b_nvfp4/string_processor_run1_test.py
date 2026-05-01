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
            A string with the words in reverse order.
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
        Applies a Caesar cipher to a string (only affects a-z/A-Z characters).

        Args:
            s: The input string.
            shift: The number of positions to shift each character.

        Returns:
            The Caesar-ciphered string.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Finds the most common word in a string (case-insensitive).

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
    processor = StringProcessor()
    assert processor.reverse_words("hello world") == "world hello"

def test_count_vowels():
    processor = StringProcessor()
    assert processor.count_vowels("Hello World") == 3

def test_is_palindrome():
    processor = StringProcessor()
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True

def test_caesar_cipher():
    processor = StringProcessor()
    assert processor.caesar_cipher("Hello, World!", 1) == "Ifmmp, Xpsme!"

def test_most_common_word():
    processor = StringProcessor()
    assert processor.most_common_word("Hello world hello") == "hello"