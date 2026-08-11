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
            String with words in reversed order

        Example:
            >>> sp = StringProcessor()
            >>> sp.reverse_words("hello world")
            'world hello'
        """
        return ' '.join(reversed(s.split()))

    def count_vowels(self, s: str) -> int:
        """
        Count the number of vowels in a string (case-insensitive).

        Args:
            s: Input string

        Returns:
            Number of vowels in the string

        Example:
            >>> sp = StringProcessor()
            >>> sp.count_vowels("hello")
            2
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if a string is a palindrome (ignore case, spaces, punctuation).

        Args:
            s: Input string

        Returns:
            True if string is palindrome, False otherwise

        Example:
            >>> sp = StringProcessor()
            >>> sp.is_palindrome("A man, a plan, a canal: Panama")
            True
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher to a string (only affects a-z/A-Z characters).

        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)

        Returns:
            Encrypted/decrypted string

        Example:
            >>> sp = StringProcessor()
            >>> sp.caesar_cipher("abc", 1)
            'bcd'
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
        Returns the first word if there's a tie.

        Args:
            s: Input string

        Returns:
            Most common word, or None if string is empty

        Example:
            >>> sp = StringProcessor()
            >>> sp.most_common_word("hello world hello")
            'hello'
        """
        words = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        if not words:
            return None

        word_counts = Counter(words)
        max_count = word_counts.most_common(1)[0][1]

        # Return first word with max count (maintains insertion order for ties)
        for word in words:
            if word_counts[word] == max_count:
                return word

import pytest

@pytest.fixture
def sp():
    return StringProcessor()

def test_reverse_words(sp):
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("a b c") == "c b a"
    assert sp.reverse_words("single") == "single"
    assert sp.reverse_words("") == ""

def test_count_vowels(sp):
    assert sp.count_vowels("hello") == 2
    assert sp.count_vowels("HELLO") == 2
    assert sp.count_vowels("xyz") == 0
    assert sp.count_vowels("aeiou") == 5

def test_is_palindrome(sp):
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race a car") is False
    assert sp.is_palindrome("Was it a car or a cat I saw?") is True
    assert sp.is_palindrome("") is True

def test_caesar_cipher(sp):
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("ABC", 1) == "BCD"
    assert sp.caesar_cipher("abc", -1) == "zab"
    assert sp.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"

def test_most_common_word(sp):
    assert sp.most_common_word("hello world hello") == "hello"
    assert sp.most_common_word("a b c a b a") == "a"
    assert sp.most_common_word("The quick brown fox") == "the"
    assert sp.most_common_word("") is None