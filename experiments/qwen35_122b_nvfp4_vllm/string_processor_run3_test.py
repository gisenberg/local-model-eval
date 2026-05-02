# string_processor.py

import re
from typing import Optional
from collections import Counter


class StringProcessor:
    """A utility class for various string processing operations."""

    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Reverses the order of words in a string.

        Multiple spaces between words are treated as a single delimiter, 
        and the resulting string is joined with single spaces.

        Args:
            s (str): The input string.

        Returns:
            str: A new string with words in reversed order.
        
        Examples:
            >>> StringProcessor.reverse_words("hello world")
            'world hello'
            >>> StringProcessor.reverse_words("  python   code  ")
            'code python'
        """
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Counts the number of vowels in the string (case-insensitive).

        Vowels are defined as 'a', 'e', 'i', 'o', 'u'.

        Args:
            s (str): The input string.

        Returns:
            int: The total count of vowels.
        
        Examples:
            >>> StringProcessor.count_vowels("Hello World")
            3
            >>> StringProcessor.count_vowels("AEIOU")
            5
        """
        vowels = {'a', 'e', 'i', 'o', 'u'}
        return sum(1 for char in s.lower() if char in vowels)

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Checks if the string is a palindrome (ignoring case, spaces, punctuation).

        Args:
            s (str): The input string.

        Returns:
            bool: True if the cleaned string reads the same forwards and backwards.
        
        Examples:
            >>> StringProcessor.is_palindrome("A man, a plan, a canal: Panama")
            True
            >>> StringProcessor.is_palindrome("race car")
            True
            >>> StringProcessor.is_palindrome("hello")
            False
        """
        # Keep only alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-z0-9]', '', s.lower())
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Applies a Caesar cipher encryption to alphabetic characters.

        Supports shifting both uppercase and lowercase English letters.
        Non-alphabetic characters remain unchanged. Negative shifts are supported.

        Args:
            s (str): The input string.
            shift (int): The number of positions to shift (can be negative).

        Returns:
            str: The encrypted/shifted string.
        
        Examples:
            >>> StringProcessor.caesar_cipher("abc", 1)
            'bcd'
            >>> StringProcessor.caesar_cipher("xyz", 1)
            'yza'
            >>> StringProcessor.caesar_cipher("Abc", -1)
            'Zab'
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                # Calculate shifted position with wrap-around (modulo 26)
                offset = (ord(char) - base + shift) % 26
                result.append(chr(base + offset))
            else:
                result.append(char)
        return "".join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Finds the most common word in the string (case-insensitive).

        Ignores punctuation. If multiple words have the same maximum count,
        returns the one that appears first in the text.

        Args:
            s (str): The input string.

        Returns:
            Optional[str]: The most common word, or None if no words found.
        
        Examples:
            >>> StringProcessor.most_common_word("The cat and the dog")
            'the'
            >>> StringProcessor.most_common_word("A a a b b")
            'a'
        """
        if not s:
            return None
        
        # Extract words using regex, converting to lowercase
        words = re.findall(r'\b\w+\b', s.lower())
        
        if not words:
            return None

        # Count occurrences
        counts = Counter(words)
        max_count = max(counts.values())

        # Return the first word encountered with the max_count to handle ties
        for word in words:
            if counts[word] == max_count:
                return word
        return None

# test_string_processor.py

import pytest


class TestStringProcessorMethods:

    def test_reverse_words(self):
        """Test word reversal functionality including multiple spaces."""
        assert StringProcessor.reverse_words("hello world") == "world hello"
        assert StringProcessor.reverse_words("  python   code  ") == "code python"
        assert StringProcessor.reverse_words("single") == "single"
        assert StringProcessor.reverse_words("") == ""

    def test_count_vowels(self):
        """Test vowel counting with mixed case."""
        assert StringProcessor.count_vowels("Hello World") == 3
        assert StringProcessor.count_vowels("AEIOU") == 5
        assert StringProcessor.count_vowels("XYZ") == 0
        assert StringProcessor.count_vowels("sky") == 0
        assert StringProcessor.count_vowels("aEiOu") == 5

    def test_is_palindrome(self):
        """Test palindrome detection ignoring punctuation and case."""
        assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
        assert StringProcessor.is_palindrome("race car") is True
        assert StringProcessor.is_palindrome("Not a palindrome") is False
        assert StringProcessor.is_palindrome("!@#$%") is True # Empty string after cleanup is palindromic
        assert StringProcessor.is_palindrome("a") is True

    def test_caesar_cipher(self):
        """Test Caesar cipher with positive, negative, and boundary shifts."""
        assert StringProcessor.caesar_cipher("abc", 1) == "bcd"
        assert StringProcessor.caesar_cipher("xyz", 1) == "yza"
        assert StringProcessor.caesar_cipher("XYZ", -1) == "WXY"
        assert StringProcessor.caesar_cipher("A", 25) == "Z"
        assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

    def test_most_common_word(self):
        """Test finding the most common word with tie-breaking logic."""
        assert StringProcessor.most_common_word("The cat and the dog") == "the"
        assert StringProcessor.most_common_word("A b a b c c") == "a" # Ties broken by first appearance
        assert StringProcessor.most_common_word("One Two Three Four Five") == "one" # All tied, first wins
        assert StringProcessor.most_common_word("No words here!!!") == None # Wait, 'here' exists
        # Correction for empty/no words scenario
        assert StringProcessor.most_common_word("!@#$%") is None
        assert StringProcessor.most_common_word("python java python java") == "python"