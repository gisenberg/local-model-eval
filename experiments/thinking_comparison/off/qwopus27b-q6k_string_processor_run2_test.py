from typing import Optional
import re


class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.
        
        Multiple spaces between words are collapsed to a single space.
        Leading and trailing spaces are removed.
        
        Args:
            s: Input string
            
        Returns:
            String with words in reversed order
        """
        words = s.split()
        return ' '.join(reversed(words))
    
    def count_vowels(self, s: str) -> int:
        """
        Count the number of vowels in the string.
        
        Vowels are a, e, i, o, u (case-insensitive).
        
        Args:
            s: Input string
            
        Returns:
            Count of vowels
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)
    
    def is_palindrome(self, s: str) -> bool:
        """
        Check if the string is a palindrome.
        
        Ignores case, spaces, and punctuation.
        
        Args:
            s: Input string
            
        Returns:
            True if palindrome, False otherwise
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]
    
    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift.
        
        Only shifts a-z and A-Z. Other characters remain unchanged.
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
                shifted = (ord(char) - ord('a') + shift) % 26
                result.append(chr(ord('a') + shifted))
            elif 'A' <= char <= 'Z':
                shifted = (ord(char) - ord('A') + shift) % 26
                result.append(chr(ord('A') + shifted))
            else:
                result.append(char)
        return ''.join(result)
    
    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word.
        
        Case-insensitive comparison. If tied, returns the one that appears first.
        Returns None for empty strings.
        
        Args:
            s: Input string
            
        Returns:
            Most common word or None if empty
        """
        if not s or not s.strip():
            return None
        
        words = re.findall(r'\b\w+\b', s.lower())
        if not words:
            return None
        
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        
        max_count = max(freq.values())
        
        for word in words:
            if freq[word] == max_count:
                return word
        
        return None


# Pytest tests
import pytest


class TestStringProcessor:
    def test_reverse_words(self):
        processor = StringProcessor()
        assert processor.reverse_words("hello world") == "world hello"
        assert processor.reverse_words("  hello   world  ") == "world hello"
        assert processor.reverse_words("single") == "single"
        assert processor.reverse_words("") == ""
        assert processor.reverse_words("   ") == ""
    
    def test_count_vowels(self):
        processor = StringProcessor()
        assert processor.count_vowels("hello") == 2
        assert processor.count_vowels("AEIOU") == 5
        assert processor.count_vowels("bcdfg") == 0
        assert processor.count_vowels("") == 0
        assert processor.count_vowels("Hello World") == 3
    
    def test_is_palindrome(self):
        processor = StringProcessor()
        assert processor.is_palindrome("racecar") is True
        assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
        assert processor.is_palindrome("hello") is False
        assert processor.is_palindrome("") is True
        assert processor.is_palindrome("Was it a car or a cat I saw?") is True
    
    def test_caesar_cipher(self):
        processor = StringProcessor()
        assert processor.caesar_cipher("abc", 1) == "bcd"
        assert processor.caesar_cipher("xyz", 1) == "yza"
        assert processor.caesar_cipher("bcd", -1) == "abc"
        assert processor.caesar_cipher("Hello", 1) == "Ifmmp"
        assert processor.caesar_cipher("Hello, World!", 1) == "Ifmmp, Xpsme!"
        assert processor.caesar_cipher("abc", 27) == "bcd"
    
    def test_most_common_word(self):
        processor = StringProcessor()
        assert processor.most_common_word("the cat and the dog") == "the"
        assert processor.most_common_word("The THE the") == "the"
        assert processor.most_common_word("cat dog cat dog") == "cat"
        assert processor.most_common_word("") is None
        assert processor.most_common_word("hello") == "hello"