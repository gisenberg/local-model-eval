import re
from typing import Optional
from collections import Counter


class StringProcessor:
    """A utility class for processing and analyzing strings."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Reverses the order of words in a given string.
        
        Args:
            s: The input string.
            
        Returns:
            A string with words in reversed order.
            Multiple consecutive whitespaces are normalized to single spaces.
        """
        return " ".join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Counts the number of vowels in the string (case-insensitive).
        
        Args:
            s: The input string.
            
        Returns:
            The integer count of vowels (a, e, i, o, u).
        """
        vowels = set("aeiou")
        return sum(1 for char in s.lower() if char in vowels)

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Checks if the string is a palindrome, ignoring case, spaces, and punctuation.
        
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
        Applies a Caesar cipher to the string.
        Only shifts alphabetical characters (a-z, A-Z); other characters remain unchanged.
        Supports negative shifts for decryption.
        
        Args:
            s: The input string.
            shift: The integer number of positions to shift.
            
        Returns:
            The shifted string.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted_char = chr((ord(char) - base + shift) % 26 + base)
                result.append(shifted_char)
            else:
                result.append(char)
        return "".join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Returns the most frequently occurring word in the string.
        Comparison is case-insensitive.
        If there is a tie, returns the word that appeared first in the text.
        Returns None if no valid words are found.
        
        Args:
            s: The input string.
            
        Returns:
            The most common word or None if empty.
        """
        words = re.findall(r'\w+', s.lower())
        if not words:
            return None
        
        counter = Counter(words)
        max_freq = max(counter.values())
        
        for word in words:
            if counter[word] == max_freq:
                return word
                
        return None


# --- Pytest Tests ---
import pytest


def test_reverse_words():
    processor = StringProcessor()
    assert processor.reverse_words("Hello World") == "World Hello"
    assert processor.reverse_words("") == ""
    assert processor.reverse_words("  multiple   spaces  ") == "spaces multiple"


def test_count_vowels():
    processor = StringProcessor()
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("AEIOU") == 5
    assert processor.count_vowels("xyz") == 0


def test_is_palindrome():
    processor = StringProcessor()
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("racecar") is True
    assert processor.is_palindrome("Not a palindrome") is False
    assert processor.is_palindrome("") is True


def test_caesar_cipher():
    processor = StringProcessor()
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("xyz", 1) == "yza"
    assert processor.caesar_cipher("ABC", -1) == "ZAB"
    assert processor.caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"


def test_most_common_word():
    processor = StringProcessor()
    assert processor.most_common_word("one two three two one") == "one"
    assert processor.most_common_word("APPLE APPLE Banana banana") == "apple"
    assert processor.most_common_word("") is None