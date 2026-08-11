from collections import Counter
import re
from typing import Optional

class StringProcessor:
    """A class for processing and analyzing strings with various operations."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.

        Args:
            s: Input string

        Returns:
            String with words in reversed order
        """
        words = s.split()
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count the number of vowels in a string (case-insensitive).

        Args:
            s: Input string

        Returns:
            Number of vowels (a, e, i, o, u) in the string
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: Input string

        Returns:
            True if the string is a palindrome, False otherwise
        """
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher to a string with the given shift.

        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)

        Returns:
            Encrypted/decrypted string with only a-z/A-Z characters shifted
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= char <= 'Z':
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Find the most common word in a string (case-insensitive).

        Args:
            s: Input string

        Returns:
            The most common word, or None if string is empty.
            If there's a tie, returns the first one encountered.
        """
        if not s.strip():
            return None

        # Extract words (alphanumeric sequences)
        words = re.findall(r'[a-zA-Z0-9]+', s.lower())
        if not words:
            return None

        word_counts = Counter(words)
        max_count = max(word_counts.values())

        # Return first word with max count (in order of appearance)
        for word in words:
            if word_counts[word] == max_count:
                return word
        return None


# Test file: test_string_processor.py
import pytest

@pytest.fixture
def processor():
    return StringProcessor()

def test_reverse_words(processor):
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("a b c") == "c b a"
    assert processor.reverse_words("") == ""

def test_count_vowels(processor):
    assert processor.count_vowels("hello") == 2
    assert processor.count_vowels("HELLO") == 2
    assert processor.count_vowels("xyz") == 0

def test_is_palindrome(processor):
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("race a car") is False
    assert processor.is_palindrome("") is True

def test_caesar_cipher(processor):
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("ABC", 1) == "BCD"
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("xyz", 3) == "abc"

def test_most_common_word(processor):
    assert processor.most_common_word("hello world hello") == "hello"
    assert processor.most_common_word("a a b b") == "a"  # First encountered
    assert processor.most_common_word("") is None