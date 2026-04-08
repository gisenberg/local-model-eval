from typing import Optional

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string, normalizing spaces."""
        words = [word for word in s.split() if word]
        return ' '.join(reversed(words))
    
    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) case-insensitively."""
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)
    
    def is_palindrome(self, s: str) -> bool:
        """Check if string is palindrome, ignoring case, spaces, and punctuation."""
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]
    
    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift to alphabetic characters only."""
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
        """Return most frequently occurring word (case-insensitive)."""
        if not s or not s.strip():
            return None
        
        words = [word.lower() for word in s.split() if word]
        if not words:
            return None
        
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_count = max(word_counts.values())
        
        for word in words:
            if word_counts[word] == max_count:
                return word


# Pytest tests
import pytest

class TestStringProcessor:
    def test_reverse_words(self):
        processor = StringProcessor()
        assert processor.reverse_words("hello world") == "world hello"
        assert processor.reverse_words("  hello   world  ") == "world hello"
        assert processor.reverse_words("") == ""
    
    def test_count_vowels(self):
        processor = StringProcessor()
        assert processor.count_vowels("hello") == 2
        assert processor.count_vowels("AEIOU") == 5
        assert processor.count_vowels("bcdfg") == 0
    
    def test_is_palindrome(self):
        processor = StringProcessor()
        assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
        assert processor.is_palindrome("racecar") == True
        assert processor.is_palindrome("hello") == False
    
    def test_caesar_cipher(self):
        processor = StringProcessor()
        assert processor.caesar_cipher("abc", 1) == "bcd"
        assert processor.caesar_cipher("ABC", 1) == "BCD"
        assert processor.caesar_cipher("abc", -1) == "zab"
        assert processor.caesar_cipher("a1b", 1) == "b1c"
    
    def test_most_common_word(self):
        processor = StringProcessor()
        assert processor.most_common_word("hello world hello") == "hello"
        assert processor.most_common_word("a b a c b") == "a"
        assert processor.most_common_word("") is None