import re
from collections import Counter
from typing import Optional

class StringProcessor:
    """A class for various string processing operations."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.

        Args:
            s: Input string

        Returns:
            String with words reversed
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """
        Count vowels in a string (case-insensitive).

        Args:
            s: Input string

        Returns:
            Number of vowels
        """
        return sum(1 for c in s.lower() if c in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """
        Check if string is palindrome (ignoring case, spaces, punctuation).

        Args:
            s: Input string

        Returns:
            True if palindrome, False otherwise
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher to string.

        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)

        Returns:
            Encrypted/decrypted string
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
        Find most common word in string (case-insensitive).

        Args:
            s: Input string

        Returns:
            Most common word, or None if string is empty
        """
        words = s.lower().split()
        if not words:
            return None
        counter = Counter(words)
        return counter.most_common(1)[0][0]

# Tests
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
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
    assert processor.is_palindrome("race a car") == False
    assert processor.is_palindrome("Was it a car or a cat I saw?") == True

def test_caesar_cipher(processor):
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("xyz", 3) == "abc"
    assert processor.caesar_cipher("ABC", -1) == "ZAB"
    assert processor.caesar_cipher("hello!", 2) == "jgnnq!"

def test_most_common_word(processor):
    assert processor.most_common_word("hello world hello") == "hello"
    assert processor.most_common_word("Python python PYTHON") == "python"
    assert processor.most_common_word("") is None