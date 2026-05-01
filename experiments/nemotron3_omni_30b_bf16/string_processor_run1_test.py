import string
from collections import Counter
from typing import Optional

class StringProcessor:
    """
    A class providing various string processing utilities.
    """

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Reverses the order of words in a string.

        Args:
            s: The input string.

        Returns:
            A string with the words in reverse order.
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Counts the number of vowels in a string (case-insensitive).

        Args:
            s: The input string.

        Returns:
            The number of vowels in the string.
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Checks if a string is a palindrome (ignoring case, spaces, and punctuation).

        Args:
            s: The input string.

        Returns:
            True if the string is a palindrome, False otherwise.
        """
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Applies a Caesar cipher to a string (only affects a-z/A-Z characters).

        Args:
            s: The input string.
            shift: The number of positions to shift each character.

        Returns:
            The encrypted string.
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
        Finds the most common word in a string (case-insensitive).

        Args:
            s: The input string.

        Returns:
            The most common word, or None if the string is empty.
        """
        words = [word.lower() for word in s.split() if word]
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
    assert StringProcessor.caesar_cipher("Hello, World!", 1) == "Ifmmp, Xpsme!"
    assert StringProcessor.caesar_cipher("Ifmmp, Xpsme!", -1) == "Hello, World!"

def test_most_common_word():
    assert StringProcessor.most_common_word("Hello hello world") == "hello"
    assert StringProcessor.most_common_word("") is None