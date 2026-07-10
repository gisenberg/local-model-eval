from typing import Optional
import re
import string


class StringProcessor:
    """A utility class for common string processing operations."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.

        Args:
            s: Input string.

        Returns:
            String with words in reversed order (words are space-separated).
        """
        words = s.split()
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count the number of vowels (a, e, i, o, u) in a string, case-insensitive.

        Args:
            s: Input string.

        Returns:
            Number of vowels in the string.
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: Input string.

        Returns:
            True if the string is a palindrome (ignoring case, spaces, punctuation), False otherwise.
        """
        # Keep only alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply a Caesar cipher to the input string, shifting only alphabetic characters.

        Args:
            s: Input string.
            shift: Number of positions to shift (supports negative values).

        Returns:
            Encoded string where only a-z and A-Z characters are shifted; non-letters unchanged.
        """
        result = []
        shift = shift % 26  # Normalize shift to 0-25

        for char in s:
            if 'a' <= char <= 'z':
                # Lowercase letters
                offset = ord('a')
                shifted = chr((ord(char) - offset + shift) % 26 + offset)
                result.append(shifted)
            elif 'A' <= char <= 'Z':
                # Uppercase letters
                offset = ord('A')
                shifted = chr((ord(char) - offset + shift) % 26 + offset)
                result.append(shifted)
            else:
                # Non-alphabetic characters unchanged
                result.append(char)

        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Find the most common word in a string, case-insensitive.
        In case of a tie, return the first word (by original order) with the highest frequency.

        Args:
            s: Input string.

        Returns:
            Most common word (lowercase), or None if string has no words.
        """
        # Extract words (alphanumeric sequences), ignore punctuation
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None

        freq = {}
        # Count frequencies, preserving first occurrence order
        for word in words:
            freq[word] = freq.get(word, 0) + 1

        # Find max frequency and first word achieving that frequency
        max_freq = max(freq.values())
        for word in words:
            if freq[word] == max_freq:
                return word


# ----------------- pytest tests -----------------
# Save the following in a file named test_string_processor.py

import pytest


@pytest.fixture
def processor():
    return StringProcessor()


def test_reverse_words_basic(processor):
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("   a  b  c   ") == "c b a"


def test_count_vowels_case_insensitive(processor):
    assert processor.count_vowels("Hello World") == 3  # e, o, o
    assert processor.count_vowels("AEIOUaeiou") == 10


def test_is_palindrome_ignores_case_spaces_punctuation(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("racecar") is True
    assert processor.is_palindrome("Hello, world!") is False


def test_caesar_cipher_shifts_letters_only(processor):
    assert processor.caesar_cipher("abc xyz", 1) == "bcd yza"
    assert processor.caesar_cipher("ABC XYZ", -1) == "ZAB WXY"
    assert processor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"  # letters only
    assert processor.caesar_cipher("test", 26) == "test"  # wrap-around
    assert processor.caesar_cipher("test", -26) == "test"


def test_most_common_word_ties_and_case(processor):
    assert processor.most_common_word("Hello hello world") == "hello"
    assert processor.most_common_word("apple banana apple banana") == "apple"  # tie: first wins
    assert processor.most_common_word("") is None
    assert processor.most_common_word("!!!") is None
    assert processor.most_common_word("The quick brown fox") == "the"