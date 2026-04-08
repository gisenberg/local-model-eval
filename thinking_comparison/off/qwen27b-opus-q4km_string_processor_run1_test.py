from typing import Optional
import re

class StringProcessor:
    def reverse_words(self, s: str) -> str:
        """Reverse the order of words in a string.
        
        Multiple spaces between words become a single space.
        Leading/trailing spaces are removed.
        """
        if not s or not s.strip():
            return ""
        words = s.split()
        return " ".join(reversed(words))
    
    def count_vowels(self, s: str) -> int:
        """Count vowels (a, e, i, o, u) in the string, case-insensitive."""
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)
    
    def is_palindrome(self, s: str) -> bool:
        """Check if string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]
    
    def caesar_cipher(self, s: str, shift: int) -> str:
        """Apply Caesar cipher with given shift.
        
        Only shifts a-z and A-Z, leaves other characters unchanged.
        Supports negative shifts.
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
        """Return the most frequently occurring word (case-insensitive).
        
        If tied, returns the one that appears first.
        Returns None for empty strings.
        """
        if not s or not s.strip():
            return None
        
        words = s.split()
        word_counts = {}
        first_occurrence = {}
        
        for i, word in enumerate(words):
            lower_word = word.lower()
            if lower_word not in word_counts:
                word_counts[lower_word] = 0
                first_occurrence[lower_word] = i
            word_counts[lower_word] += 1
        
        max_count = max(word_counts.values())
        candidates = [word for word, count in word_counts.items() if count == max_count]
        
        candidates.sort(key=lambda w: first_occurrence[w])
        return candidates[0]


# Pytest tests
def test_reverse_words():
    processor = StringProcessor()
    assert processor.reverse_words("  hello   world  ") == "world hello"
    assert processor.reverse_words("single") == "single"
    assert processor.reverse_words("   ") == ""
    assert processor.reverse_words("a b c") == "c b a"

def test_count_vowels():
    processor = StringProcessor()
    assert processor.count_vowels("Hello World") == 3
    assert processor.count_vowels("xyz") == 0
    assert processor.count_vowels("AEIOU") == 5
    assert processor.count_vowels("") == 0

def test_is_palindrome():
    processor = StringProcessor()
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
    assert processor.is_palindrome("racecar") == True
    assert processor.is_palindrome("Hello") == False
    assert processor.is_palindrome("Was it a car or a cat I saw?") == True

def test_caesar_cipher():
    processor = StringProcessor()
    assert processor.caesar_cipher("Hello", 1) == "Ifmmp"
    assert processor.caesar_cipher("ABC", -1) == "ZAB"
    assert processor.caesar_cipher("Test123!", 5) == "Yjxy123!"
    assert processor.caesar_cipher("abc", 27) == "bcd"

def test_most_common_word():
    processor = StringProcessor()
    assert processor.most_common_word("hello world hello") == "hello"
    assert processor.most_common_word("a b a b") == "a"
    assert processor.most_common_word("Hello HELLO hello") == "hello"
    assert processor.most_common_word("") is None
    assert processor.most_common_word("   ") is None