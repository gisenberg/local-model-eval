from typing import Optional
import re
from collections import Counter


class StringProcessor:
    """A class for processing and analyzing strings."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Reverse the order of words in a string.

        Args:
            s: Input string.

        Returns:
            String with words in reversed order.

        Example:
            >>> StringProcessor.reverse_words("hello world")
            'world hello'
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Count the number of vowels in a string (case-insensitive).

        Args:
            s: Input string.

        Returns:
            Number of vowels (a, e, i, o, u).

        Example:
            >>> StringProcessor.count_vowels("Hello")
            2
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Check if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: Input string.

        Returns:
            True if the cleaned string reads the same forwards and backwards.

        Example:
            >>> StringProcessor.is_palindrome("A man, a plan, a canal: Panama")
            True
        """
        cleaned = re.sub(r'[^a-z0-9]', '', s.lower())
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to a string, shifting letters by `shift` positions.
        Only a-z and A-Z are affected; case is preserved. Supports negative shifts.

        Args:
            s: Input string.
            shift: Number of positions to shift (can be negative).

        Returns:
            Encrypted/decrypted string.

        Example:
            >>> StringProcessor.caesar_cipher("Hello", 3)
            'Khoor'
            >>> StringProcessor.caesar_cipher("Khoor", -3)
            'Hello'
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                base = ord('a')
                shifted = (ord(char) - base + shift) % 26
                result.append(chr(base + shifted))
            elif 'A' <= char <= 'Z':
                base = ord('A')
                shifted = (ord(char) - base + shift) % 26
                result.append(chr(base + shifted))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Find the most common word in a string (case-insensitive).
        In case of a tie, returns the word that appears first.

        Args:
            s: Input string.

        Returns:
            The most common word, or None if no words are found.

        Example:
            >>> StringProcessor.most_common_word("apple banana apple cherry banana")
            'apple'
        """
        words = re.findall(r'[a-zA-Z]+', s.lower())
        if not words:
            return None
        counts = Counter(words)
        # Find the word with the highest count; Counter.most_common() preserves
        # insertion order for ties in Python 3.7+, but we ensure first occurrence
        # by iterating through the original word list.
        max_count = max(counts.values())
        for word in words:
            if counts[word] == max_count:
                return word
        return None


# pytest tests
def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  one   two  three  ") == "three two one"
    assert StringProcessor.reverse_words("") == ""
    assert StringProcessor.reverse_words("single") == "single"
    assert StringProcessor.reverse_words("a b c d") == "d c b a"


def test_count_vowels():
    assert StringProcessor.count_vowels("Hello") == 2
    assert StringProcessor.count_vowels("AEIOUaeiou") == 10
    assert StringProcessor.count_vowels("xyz") == 0
    assert StringProcessor.count_vowels("") == 0
    assert StringProcessor.count_vowels("Python is fun!") == 4


def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("racecar") is True
    assert StringProcessor.is_palindrome("hello") is False
    assert StringProcessor.is_palindrome("") is True
    assert StringProcessor.is_palindrome("No 'x' in Nixon") is True


def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("Hello", 3) == "Khoor"
    assert StringProcessor.caesar_cipher("Khoor", -3) == "Hello"
    assert StringProcessor.caesar_cipher("abc", 1) == "bcd"
    assert StringProcessor.caesar_cipher("XYZ", 3) == "ABC"
    assert StringProcessor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"


def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple cherry banana") == "apple"
    assert StringProcessor.most_common_word("dog cat dog cat bird") == "dog"
    assert StringProcessor.most_common_word("Hello hello HELLO") == "hello"
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("123 !@#") is None