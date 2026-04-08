from typing import Optional
import re

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string, normalizing spaces."""
        if not s or not s.strip():
            return ""
        words = s.split()  # split() handles multiple spaces automatically
        return " ".join(reversed(words))
    
    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) case-insensitively."""
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)
    
    def is_palindrome(self, s: str) -> bool:
        """Check if string is palindrome, ignoring case, spaces, and punctuation."""
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        return cleaned == cleaned[::-1]
    
    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift to alphabetic characters only."""
        result = []
        for char in s:
            if char.isalpha():
                # Determine base ASCII value (65 for uppercase, 97 for lowercase)
                base = ord('A') if char.isupper() else ord('a')
                # Apply shift with wraparound
                shifted = (ord(char) - base + shift) % 26 + base
                result.append(chr(shifted))
            else:
                result.append(char)
        return ''.join(result)
    
    def most_common_word(self, s: str) -> Optional[str]:
        """Return most frequently occurring word (case-insensitive)."""
        if not s or not s.strip():
            return None
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\w+', s.lower())
        if not words:
            return None
        
        # Count frequencies and track first occurrence
        word_count = {}
        first_occurrence = {}
        for i, word in enumerate(words):
            word_count[word] = word_count.get(word, 0) + 1
            if word not in first_occurrence:
                first_occurrence[word] = i
        
        # Find maximum frequency
        max_count = max(word_count.values())
        
        # Among words with max frequency, return the one with earliest first occurrence
        candidates = [word for word, count in word_count.items() if count == max_count]
        return min(candidates, key=lambda w: first_occurrence[w])


# Pytest tests
import pytest

class TestStringProcessor:
    def test_reverse_words(self):
        processor = StringProcessor()
        assert processor.reverse_words("  hello   world  ") == "world hello"
        assert processor.reverse_words("") == ""
        assert processor.reverse_words("single") == "single"
    
    def test_count_vowels(self):
        processor = StringProcessor()
        assert processor.count_vowels("hello") == 2
        assert processor.count_vowels("AEIOU") == 5
        assert processor.count_vowels("xyz") == 0
    
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
        assert processor.most_common_word("hello world") == "hello"
        assert processor.most_common_word("") is None