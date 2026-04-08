from typing import Optional
import re


class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.
        
        Multiple spaces between words become a single space.
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
            Integer count of vowels
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
        Return the most frequently occurring word.
        
        Case-insensitive. If tied, returns the one that appears first.
        Returns None for empty strings.
        
        Args:
            s: Input string
            
        Returns:
            Most common word (lowercase) or None if empty
        """
        if not s or not s.strip():
            return None
            
        words = s.split()
        if not words:
            return None
            
        freq = {}
        for word in words:
            word_lower = word.lower()
            freq[word_lower] = freq.get(word_lower, 0) + 1
            
        max_count = max(freq.values())
        
        for word in words:
            if freq[word.lower()] == max_count:
                return word.lower()
                
        return None


# Pytest tests
import pytest

class TestStringProcessor:
    def test_reverse_words(self):
        processor = StringProcessor()
        assert processor.reverse_words("  hello   world  ") == "world hello"
        assert processor.reverse_words("single") == "single"
        assert processor.reverse_words("") == ""
        assert processor.reverse_words("   ") == ""
        assert processor.reverse_words("a b c") == "c b a"
    
    def test_count_vowels(self):
        processor = StringProcessor()
        assert processor.count_vowels("hello") == 2
        assert processor.count_vowels("AEIOU") == 5
        assert processor.count_vowels("bcdfg") == 0
        assert processor.count_vowels("") == 0
        assert processor.count_vowels("Rhythm") == 0
    
    def test_is_palindrome(self):
        processor = StringProcessor()
        assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
        assert processor.is_palindrome("race a car") == False
        assert processor.is_palindrome("Was it a car or a cat I saw?") == True
        assert processor.is_palindrome("") == True
        assert processor.is_palindrome("No 'x' in Nixon") == True
    
    def test_caesar_cipher(self):
        processor = StringProcessor()
        assert processor.caesar_cipher("abc", 1) == "bcd"
        assert processor.caesar_cipher("xyz", 1) == "yza"
        assert processor.caesar_cipher("ABC", 1) == "BCD"
        assert processor.caesar_cipher("abc", -1) == "zab"
        assert processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
        assert processor.caesar_cipher("abc", 26) == "abc"
        assert processor.caesar_cipher("123 !@#", 5) == "123 !@#"
    
    def test_most_common_word(self):
        processor = StringProcessor()
        assert processor.most_common_word("hello world hello") == "hello"
        assert processor.most_common_word("Hello hello HELLO") == "hello"
        assert processor.most_common_word("a b a c b") == "a"
        assert processor.most_common_word("") is None
        assert processor.most_common_word("   ") is None
        assert processor.most_common_word("the quick brown fox") == "the"