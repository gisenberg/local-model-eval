import re
from typing import Optional

class StringProcessor:
    """Utility class for common string manipulation and analysis tasks."""

    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string.
        
        Args:
            s: Input string containing words separated by whitespace.
            
        Returns:
            String with words in reversed order.
        """
        return ' '.join(s.split()[::-1])

    def count_vowels(self, s: str) -> int:
        """Count the number of vowels in a string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            Total count of vowels (a, e, i, o, u).
        """
        return sum(1 for char in s.lower() if char in 'aeiou')

    def is_palindrome(self, s: str) -> bool:
        """Check if a string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string to check.
            
        Returns:
            True if the cleaned string reads the same forwards and backwards, False otherwise.
        """
        cleaned = [char.lower() for char in s if char.isalnum()]
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply a Caesar cipher to a string, shifting only alphabetic characters.
        
        Args:
            s: Input string.
            shift: Number of positions to shift. Supports negative values.
            
        Returns:
            Ciphered string with non-alphabetic characters unchanged.
        """
        result = []
        for char in s:
            if char.isupper():
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            elif char.islower():
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """Find the most common word in a string (case-insensitive).
        
        Args:
            s: Input string.
            
        Returns:
            The most frequent word. Returns the first occurrence if multiple words are tied.
            Returns None if the string contains no words.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None

        counts: dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1

        max_count = max(counts.values())
        # Preserve original order to return the first word in case of a tie
        for word in words:
            if counts[word] == max_count:
                return word


# ========================
# Pytest Test Suite
# ========================

def test_reverse_words() -> None:
    sp = StringProcessor()
    assert sp.reverse_words("Hello World") == "World Hello"
    assert sp.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert sp.reverse_words("single") == "single"

def test_count_vowels() -> None:
    sp = StringProcessor()
    assert sp.count_vowels("Hello World") == 3
    assert sp.count_vowels("AEIOU aeiou") == 10
    assert sp.count_vowels("bcdfg") == 0
    assert sp.count_vowels("") == 0

def test_is_palindrome() -> None:
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("racecar") is True
    assert sp.is_palindrome("hello") is False
    assert sp.is_palindrome("No 'x' in Nixon") is True

def test_caesar_cipher() -> None:
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("XYZ", -1) == "WXY"
    assert sp.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
    assert sp.caesar_cipher("a", -1) == "z"

def test_most_common_word() -> None:
    sp = StringProcessor()
    assert sp.most_common_word("apple banana apple orange banana apple") == "apple"
    assert sp.most_common_word("cat dog cat dog") == "cat"  # Tied, returns first
    assert sp.most_common_word("  ... ,,, !!! ") is None
    assert sp.most_common_word("The quick brown fox jumps over the lazy dog") == "the"