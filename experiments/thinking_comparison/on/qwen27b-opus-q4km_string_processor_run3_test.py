from typing import Optional
import re
from collections import Counter


class StringProcessor:
    """A class for various string processing operations."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string (not the characters).
        Multiple spaces between words become a single space.
        Leading/trailing spaces are removed.
        
        Args:
            s: Input string
            
        Returns:
            String with word order reversed
        """
        if not s or not s.strip():
            return ""
        words = s.split()
        return ' '.join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count vowels (a, e, i, o, u) in the string, case-insensitive.
        
        Args:
            s: Input string
            
        Returns:
            Number of vowels in the string
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string
            
        Returns:
            True if palindrome, False otherwise
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift. Only shift a-z and A-Z,
        leave other characters unchanged. Supports negative shifts.
        
        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)
            
        Returns:
            Encrypted string
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                shifted = (ord(char) - ord('a') + shift) % 26 + ord('a')
                result.append(chr(shifted))
            elif 'A' <= char <= 'Z':
                shifted = (ord(char) - ord('A') + shift) % 26 + ord('A')
                result.append(chr(shifted))
            else:
                result.append(char)
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive).
        If tied, return the one that appears first.
        Return None for empty strings.
        
        Args:
            s: Input string
            
        Returns:
            Most common word or None if string is empty
        """
        if not s or not s.strip():
            return None
        words = s.split()
        if not words:
            return None
        word_counts = Counter(word.lower() for word in words)
        max_count = max(word_counts.values())
        for word in words:
            if word_counts[word.lower()] == max_count:
                return word.lower()
        return None

import pytest



class TestStringProcessor:
    """Test suite for StringProcessor class."""

    @pytest.fixture
    def processor(self):
        return StringProcessor()

    def test_reverse_words(self, processor):
        """Test reverse_words method."""
        assert processor.reverse_words("hello world") == "world hello"
        assert processor.reverse_words("  hello   world  ") == "world hello"
        assert processor.reverse_words("  ") == ""
        assert processor.reverse_words("single") == "single"
        assert processor.reverse_words("a b c") == "c b a"

    def test_count_vowels(self, processor):
        """Test count_vowels method."""
        assert processor.count_vowels("Hello World") == 3
        assert processor.count_vowels("xyz") == 0
        assert processor.count_vowels("AEIOU") == 5
        assert processor.count_vowels("") == 0
        assert processor.count_vowels("bcdfg") == 0

    def test_is_palindrome(self, processor):
        """Test is_palindrome method."""
        assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
        assert processor.is_palindrome("racecar") == True
        assert processor.is_palindrome("Hello") == False
        assert processor.is_palindrome("") == True
        assert processor.is_palindrome("No 'x' in Nixon") == True

    def test_caesar_cipher(self, processor):
        """Test caesar_cipher method."""
        assert processor.caesar_cipher("Hello", 1) == "Ifmmp"
        assert processor.caesar_cipher("Hello", -1) == "Gdkkn"
        assert processor.caesar_cipher("ABC xyz", 1) == "BCD yza"
        assert processor.caesar_cipher("Hello, World! 123", 3) == "Khoor, Zruog! 123"
        assert processor.caesar_cipher("z", 1) == "a"

    def test_most_common_word(self, processor):
        """Test most_common_word method."""
        assert processor.most_common_word("hello world hello") == "hello"
        assert processor.most_common_word("a b a c b") == "a"
        assert processor.most_common_word("Hello HELLO hello") == "hello"
        assert processor.most_common_word("") is None
        assert processor.most_common_word("  ") is None
        assert processor.most_common_word("one two three") == "one"