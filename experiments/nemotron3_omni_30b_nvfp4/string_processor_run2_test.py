import string
from collections import Counter
from typing import Optional
import re

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in the input string.
        
        Args:
            s: Input string.
            
        Returns:
            String with words in reversed order.
        """
        words = s.split()
        return ' '.join(reversed(words))
    
    def count_vowels(self, s: str) -> int:
        """
        Counts the number of vowels (a, e, i, o, u) in the input string, case-insensitive.
        
        Args:
            s: Input string.
            
        Returns:
            Number of vowels in the string.
        """
        vowels = 'aeiou'
        return sum(1 for char in s.lower() if char in vowels)
    
    def is_palindrome(self, s: str) -> bool:
        """
        Checks if the input string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string.
            
        Returns:
            True if the string is a palindrome, False otherwise.
        """
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]
    
    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies a Caesar cipher to the input string, shifting letters by the specified amount.
        Only supports a-z and A-Z; other characters remain unchanged.
        
        Args:
            s: Input string.
            shift: Number of positions to shift (can be negative).
            
        Returns:
            Encrypted string with letters shifted by the specified amount.
        """
        result = []
        for char in s:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26
                result.append(chr(base + shifted))
            else:
                result.append(char)
        return ''.join(result)
    
    def most_common_word(self, s: str) -> Optional[str]:
        """
        Finds the most common word in the input string, case-insensitive.
        Returns the first word if there's a tie.
        
        Args:
            s: Input string.
            
        Returns:
            Most common word (in lowercase), or None if no words exist.
        """
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        word_counts = Counter(words)
        most_common = word_counts.most_common(1)
        return most_common[0][0] if most_common else None

# Pytest tests
def test_reverse_words():
    processor = StringProcessor()
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("a b c") == "c b a"
    assert processor.reverse_words("") == ""

def test_count_vowels():
    processor = StringProcessor()
    assert processor.count_vowels("Hello") == 2
    assert processor.count_vowels("AEIOU") == 5
    assert processor.count_vowels("xyz") == 0

def test_is_palindrome():
    processor = StringProcessor()
    assert processor.is_palindrome("A man a plan a canal Panama") == True
    assert processor.is_palindrome("race car") == True
    assert processor.is_palindrome("hello") == False
    assert processor.is_palindrome("") == True

def test_caesar_cipher():
    processor = StringProcessor()
    assert processor.caesar_cipher("abc", 1) == "bcd"
    assert processor.caesar_cipher("xyz", 3) == "abc"
    assert processor.caesar_cipher("ABC", -1) == "ZAB"
    assert processor.caesar_cipher("Hello, World!", 1) == "Ifmmp, Xpsme!"

def test_most_common_word():
    processor = StringProcessor()
    assert processor.most_common_word("Hello hello world") == "hello"
    assert processor.most_common_word("a a b b c") == "a"
    assert processor.most_common_word("") is None
    assert processor.most_common_word("123 456") is None