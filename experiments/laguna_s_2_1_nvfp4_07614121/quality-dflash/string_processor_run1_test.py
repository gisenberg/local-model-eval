import re
from collections import Counter
from typing import Optional

class StringProcessor:
    """A class for performing various string operations."""

    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in a given string.

        Args:
            s: The input string.

        Returns:
            A new string with the order of words reversed.
        """
        words = s.split()
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Counts the number of vowels in a given string.

        Args:
            s: The input string.

        Returns:
            The number of vowels (a, e, i, o, u) in the string, case-insensitive.
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Checks if a given string is a palindrome.

        The check is case-insensitive and ignores spaces and punctuation.

        Args:
            s: The input string.

        Returns:
            True if the string is a palindrome, False otherwise.
        """
        # Remove all non-alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies a Caesar cipher to a given string.

        Only letters (a-z, A-Z) are shifted. The shift can be negative.

        Args:
            s: The input string.
            shift: The number of positions to shift each letter.

        Returns:
            The encrypted/decrypted string.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                base = ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            elif 'A' <= char <= 'Z':
                base = ord('A')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Finds the most common word in a given string.

        The search is case-insensitive. If there is a tie, the first word
        encountered is returned.

        Args:
            s: The input string.

        Returns:
            The most common word, or None if the string is empty or contains no words.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        # Use Counter to find the most common word
        # most_common(1) returns a list of tuples like [('word', count)]
        return Counter(words).most_common(1)[0][0]

import pytest

@pytest.fixture
def processor():
    """Fixture to provide a StringProcessor instance."""
    return StringProcessor()

def test_reverse_words(processor):
    """Test that words in a sentence are reversed."""
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("a b c") == "c b a"
    assert processor.reverse_words("") == ""
    # Handles multiple spaces correctly
    assert processor.reverse_words("  hello   world  ") == "world hello"

def test_count_vowels(processor):
    """Test that vowels are counted correctly, case-insensitively."""
    assert processor.count_vowels("hello") == 2
    assert processor.count_vowels("HELLO") == 2
    assert processor.count_vowels("xyz") == 0
    assert processor.count_vowels("The quick brown fox") == 5

def test_is_palindrome(processor):
    """Test that palindromes are detected, ignoring case, spaces, and punctuation."""
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("racecar") is True
    assert processor.is_palindrome("No lemon, no melon") is True
    assert processor.is_palindrome("hello") is False

def test_caesar_cipher(processor):
    """Test that the Caesar cipher works for positive and negative shifts."""
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("ABC", 1) == "BCD"
    assert processor.caesar_cipher("xyz", 2) == "zab"
    assert processor.caesar_cipher("XYZ", 2) == "ZAB"
    assert processor.caesar_cipher("hello, world!", 5) == "mjqqt, btwqi!"
    # Negative shift
    assert processor.caesar_cipher("bcd", -1) == "abc"
    assert processor.caesar_cipher("ZAB", -2) == "XYZ"
    # Non-letter characters remain unchanged
    assert processor.caesar_cipher("123!@#", 3) == "123!@#"

def test_most_common_word(processor):
    """Test that the most common word is found, case-insensitively."""
    assert processor.most_common_word("apple banana apple") == "apple"
    assert processor.most_common_word("The the THE tHe") == "the"
    assert processor.most_common_word("hello world") == "hello" # Tie, first one wins
    assert processor.most_common_word("test1 test2 test1") == "test1"
    assert processor.most_common_word("") is None
    assert processor.most_common_word("123 456 123") == "123"