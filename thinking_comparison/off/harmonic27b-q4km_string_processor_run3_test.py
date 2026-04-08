from typing import Optional
import re

class StringProcessor:
    """A utility class for various string processing operations."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.
        
        Multiple spaces between words become a single space.
        Leading and trailing spaces are removed.
        
        Args:
            s: Input string
            
        Returns:
            String with word order reversed
        """
        # Split by whitespace, filter out empty strings, then reverse
        words = [word for word in s.split() if word]
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
            True if the string is a palindrome, False otherwise
        """
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift to alphabetic characters only.
        
        Only shifts a-z and A-Z, leaves other characters unchanged.
        Supports negative shifts.
        
        Args:
            s: Input string
            shift: Number of positions to shift
            
        Returns:
            Encrypted string
        """
        result = []
        
        for char in s:
            if 'a' <= char <= 'z':
                # Shift lowercase letters
                new_char = chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
                result.append(new_char)
            elif 'A' <= char <= 'Z':
                # Shift uppercase letters
                new_char = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
                result.append(new_char)
            else:
                # Leave non-alphabetic characters unchanged
                result.append(char)
        
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive).
        
        If tied, return the one that appears first.
        Returns None for empty strings.
        
        Args:
            s: Input string
            
        Returns:
            Most common word or None if string is empty
        """
        if not s or not s.strip():
            return None
        
        # Split into words and convert to lowercase
        words = [word.lower() for word in s.split() if word]
        
        if not words:
            return None
        
        # Count frequencies while preserving order
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # Find the word with maximum frequency (first occurrence in case of ties)
        max_count = 0
        most_common = None
        
        for word in words:
            if word_count[word] > max_count:
                max_count = word_count[word]
                most_common = word
        
        return most_common


# Pytest tests
def test_reverse_words():
    processor = StringProcessor()
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("  multiple   spaces  ") == "spaces multiple"
    assert processor.reverse_words("single") == "single"
    assert processor.reverse_words("") == ""

def test_count_vowels():
    processor = StringProcessor()
    assert processor.count_vowels("hello") == 2  # e, o
    assert processor.count_vowels("AEIOU") == 5
    assert processor.count_vowels("xyz") == 0
    assert processor.count_vowels("") == 0

def test_is_palindrome():
    processor = StringProcessor()
    assert processor.is_palindrome("racecar") == True
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
    assert processor.is_palindrome("hello") == False
    assert processor.is_palindrome("") == True

def test_caesar_cipher():
    processor = StringProcessor()
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("ABC", 1) == "BCD"
    assert processor.caesar_cipher("abc", -1) == "zab"
    assert processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"

def test_most_common_word():
    processor = StringProcessor()
    assert processor.most_common_word("hello world hello") == "hello"
    assert processor.most_common_word("a b a c a") == "a"
    assert processor.most_common_word("a b c") == "a"  # first in case of tie
    assert processor.most_common_word("") is None
    assert processor.most_common_word("   ") is None