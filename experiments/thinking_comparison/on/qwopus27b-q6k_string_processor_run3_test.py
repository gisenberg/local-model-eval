from typing import Optional


class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.
        
        Multiple spaces between words become a single space.
        Leading and trailing spaces are removed.
        """
        words = s.split()
        return ' '.join(reversed(words))
    
    def count_vowels(self, s: str) -> int:
        """
        Count the number of vowels (a, e, i, o, u) in the string.
        
        Case-insensitive.
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)
    
    def is_palindrome(self, s: str) -> bool:
        """
        Check if the string is a palindrome.
        
        Ignores case, spaces, and punctuation.
        """
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]
    
    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift.
        
        Only shifts a-z and A-Z. Other characters remain unchanged.
        Supports negative shifts.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                shifted = (ord(char) - ord('a') + shift) % 26
                result.append(chr(shifted + ord('a')))
            elif 'A' <= char <= 'Z':
                shifted = (ord(char) - ord('A') + shift) % 26
                result.append(chr(shifted + ord('A')))
            else:
                result.append(char)
        return ''.join(result)
    
    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive).
        
        If tied, returns the one that appears first.
        Returns None for empty strings.
        """
        if not s or not s.strip():
            return None
        
        words = s.split()
        counts = {}
        order = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower not in counts:
                counts[word_lower] = 0
                order.append(word_lower)
            counts[word_lower] += 1
        
        max_count = max(counts.values())
        
        for word in order:
            if counts[word] == max_count:
                return word
        return None


# Pytest tests
import pytest


class TestStringProcessor:
    def setup_method(self):
        self.processor = StringProcessor()
    
    def test_reverse_words(self):
        """Test word reversal with various spacing scenarios."""
        assert self.processor.reverse_words("  hello   world  ") == "world hello"
        assert self.processor.reverse_words("single") == "single"
        assert self.processor.reverse_words("") == ""
        assert self.processor.reverse_words("   ") == ""
    
    def test_count_vowels(self):
        """Test vowel counting with mixed case and edge cases."""
        assert self.processor.count_vowels("Hello World") == 3  # e, o, o
        assert self.processor.count_vowels("AEIOUaeiou") == 10
        assert self.processor.count_vowels("bcdfg") == 0
        assert self.processor.count_vowels("") == 0
    
    def test_is_palindrome(self):
        """Test palindrome detection ignoring case, spaces, and punctuation."""
        assert self.processor.is_palindrome("A man, a plan, a canal: Panama") == True
        assert self.processor.is_palindrome("racecar") == True
        assert self.processor.is_palindrome("hello") == False
        assert self.processor.is_palindrome("") == True
        assert self.processor.is_palindrome("A") == True
    
    def test_caesar_cipher(self):
        """Test Caesar cipher with positive/negative shifts and non-alphabetic chars."""
        assert self.processor.caesar_cipher("abc", 1) == "bcd"
        assert self.processor.caesar_cipher("bcd", -1) == "abc"
        assert self.processor.caesar_cipher("Hello", 1) == "Ifmmp"
        assert self.processor.caesar_cipher("Hello, World!", 1) == "Ifmmp, Xpsme!"
        assert self.processor.caesar_cipher("xyz", 1) == "yza"
        assert self.processor.caesar_cipher("ABC", -1) == "ZAB"
    
    def test_most_common_word(self):
        """Test most common word detection with ties and empty inputs."""
        assert self.processor.most_common_word("hello world hello") == "hello"
        assert self.processor.most_common_word("Hello hello HELLO") == "hello"
        assert self.processor.most_common_word("a b a b") == "a"  # Tie: first appearance wins
        assert self.processor.most_common_word("") is None
        assert self.processor.most_common_word("   ") is None