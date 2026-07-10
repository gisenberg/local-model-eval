import re
from typing import Optional

class StringProcessor:
    """
    A utility class for various string processing operations.
    """

    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in the string.

        Args:
            s: The input string.

        Returns:
            A string with the words in reverse order.
        """
        # split() handles multiple spaces and stripping automatically
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """
        Counts the number of vowels in the string (case-insensitive).

        Args:
            s: The input string.

        Returns:
            The count of vowels (a, e, i, o, u).
        """
        vowels = set("aeiouAEIOU")
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Checks if the string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: The input string.

        Returns:
            True if the string is a palindrome, False otherwise.
        """
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies a Caesar cipher shift to alphabetic characters.
        Supports negative shifts and wraps around the alphabet.
        Non-alphabetic characters remain unchanged.

        Args:
            s: The input string.
            shift: The number of positions to shift (positive or negative).

        Returns:
            The encrypted string.
        """
        result = []
        for char in s:
            if char.isalpha():
                # Determine base ASCII value ('a' or 'A')
                base = ord('a') if char.islower() else ord('A')
                # Calculate new character with wrap-around using modulo
                shifted_char = chr((ord(char) - base + shift) % 26 + base)
                result.append(shifted_char)
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Finds the most common word in the string (case-insensitive).
        If there is a tie, returns the word that appeared first in the text.

        Args:
            s: The input string.

        Returns:
            The most common word, or None if the string is empty/has no words.
        """
        # Extract words using regex to handle punctuation boundaries
        # \b ensures word boundaries, \w+ matches alphanumeric sequences
        words = re.findall(r'\b\w+\b', s.lower())

        if not words:
            return None

        # Count frequencies
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1

        # Find the maximum frequency
        max_count = max(counts.values())

        # Return the first word in the original sequence that matches the max frequency
        for word in words:
            if counts[word] == max_count:
                return word

        return None

# ==========================================
# Pytest Tests
# ==========================================

def test_reverse_words():
    processor = StringProcessor()
    assert processor.reverse_words("Hello World") == "World Hello"
    assert processor.reverse_words("Python is fun") == "fun is Python"
    assert processor.reverse_words("  Multiple   spaces  ") == "spaces Multiple"

def test_count_vowels():
    processor = StringProcessor()
    assert processor.count_vowels("Hello World") == 3  # e, o, o
    assert processor.count_vowels("AEIOU") == 5
    assert processor.count_vowels("Rhythm") == 0

def test_is_palindrome():
    processor = StringProcessor()
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
    assert processor.is_palindrome("racecar") == True
    assert processor.is_palindrome("hello") == False
    assert processor.is_palindrome("Was it a car or a cat I saw?") == True

def test_caesar_cipher():
    processor = StringProcessor()
    # Shift +1
    assert processor.caesar_cipher("abc", 1) == "bcd"
    # Shift -1 (wrap around)
    assert processor.caesar_cipher("abc", -1) == "zab"
    # Mixed case and symbols
    assert processor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word():
    processor = StringProcessor()
    # Standard case
    assert processor.most_common_word("apple banana apple") == "apple"
    # Tie-breaker: 'apple' appears before 'banana'
    assert processor.most_common_word("apple banana apple banana") == "apple"
    # Case insensitivity
    assert processor.most_common_word("Apple apple APPLE") == "apple"
    # Empty string
    assert processor.most_common_word("") is None