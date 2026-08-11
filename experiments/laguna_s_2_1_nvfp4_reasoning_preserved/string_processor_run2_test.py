import re
from collections import Counter
from typing import Optional

class StringProcessor:
    """A class to perform various string operations."""

    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in a given string.

        Args:
            s: The input string.

        Returns:
            A new string with the words in reverse order.
        """
        words = s.split()
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Counts the number of vowels in a string (case-insensitive).

        Args:
            s: The input string.

        Returns:
            The number of vowels in the string.
        """
        vowels = "aeiou"
        return sum(1 for char in s.lower() if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Checks if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: The input string.

        Returns:
            True if the string is a palindrome, False otherwise.
        """
        # Remove all non-alphanumeric characters and convert to lowercase
        cleaned_s = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned_s == cleaned_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies a Caesar cipher to a string.

        Only letters (a-z, A-Z) are shifted. The shift can be negative.
        Non-letter characters are left unchanged.

        Args:
            s: The input string.
            shift: The number of positions to shift each letter.

        Returns:
            The encrypted/decrypted string.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                # Shift within lowercase letters
                shifted = ord(char) + shift
                if shifted > ord('z'):
                    shifted -= 26
                elif shifted < ord('a'):
                    shifted += 26
                result.append(chr(shifted))
            elif 'A' <= char <= 'Z':
                # Shift within uppercase letters
                shifted = ord(char) + shift
                if shifted > ord('Z'):
                    shifted -= 26
                elif shifted < ord('A'):
                    shifted += 26
                result.append(chr(shifted))
            else:
                # Leave non-letters unchanged
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Finds the most common word in a string (case-insensitive).

        If there is a tie, the word that appears first in the string is returned.
        Returns None for an empty string.

        Args:
            s: The input string.

        Returns:
            The most common word, or None if the string is empty.
        """
        # Extract words using regex to handle punctuation
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None

        # Count occurrences while preserving order for tie-breaking
        word_counts = Counter(words)
        max_count = max(word_counts.values())

        # Find the first word with the maximum count
        for word in words:
            if word_counts[word] == max_count:
                return word
        return None

import pytest

@pytest.fixture
def processor():
    """Fixture to provide a StringProcessor instance."""
    return StringProcessor()

def test_reverse_words(processor):
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("a b c") == "c b a"
    assert processor.reverse_words("") == ""
    assert processor.reverse_words("single") == "single"
    assert processor.reverse_words("  leading and trailing  ") == "trailing and leading"

def test_count_vowels(processor):
    assert processor.count_vowels("hello") == 2
    assert processor.count_vowels("HELLO") == 2  # Case-insensitive
    assert processor.count_vowels("why") == 0
    assert processor.count_vowels("") == 0
    assert processor.count_vowels("AeIoU") == 5

def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("race a car") is False
    assert processor.is_palindrome("") is True
    assert processor.is_palindrome("a.") is True
    assert processor.is_palindrome("0P") is False

def test_caesar_cipher(processor):
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("XYZ", 1) == "YZA"  # Wrap around
    assert processor.caesar_cipher("abc", -1) == "zab"  # Negative shift
    assert processor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"
    assert processor.caesar_cipher("Mjqqt, Btwqi!", -5) == "Hello, World!"

def test_most_common_word(processor):
    assert processor.most_common_word("apple orange apple banana") == "apple"
    assert processor.most_common_word("Hello hello HELLO") == "hello"  # Case-insensitive
    assert processor.most_common_word("a a b b c") == "a"  # Tie, first one wins
    assert processor.most_common_word("") is None
    assert processor.most_common_word("one") == "one"