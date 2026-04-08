from typing import Optional
import string

class StringProcessor:
    def __init__(self):
        pass

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string.
        Normalizes whitespace (multiple spaces become one) and strips edges.
        """
        # split() without arguments handles multiple spaces and stripping automatically
        words = s.split()
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count the number of vowels (a, e, i, o, u) in the string (case-insensitive).
        """
        vowels = set("aeiouAEIOU")
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        """
        # Create a translation table to remove punctuation and spaces
        translator = str.maketrans('', '', string.punctuation + ' ')
        cleaned = s.translate(translator).lower()
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift.
        Only shifts a-z and A-Z; other characters remain unchanged.
        Supports negative shifts.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                # Shift within lowercase range
                offset = ord('a')
                result.append(chr((ord(char) - offset + shift) % 26 + offset))
            elif 'A' <= char <= 'Z':
                # Shift within uppercase range
                offset = ord('A')
                result.append(chr((ord(char) - offset + shift) % 26 + offset))
            else:
                # Non-alphabetic characters remain unchanged
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive).
        If tied, return the one that appears first.
        Returns None for empty strings.
        """
        if not s:
            return None
        
        # Normalize to lowercase and split by whitespace
        words = s.lower().split()
        
        if not words:
            return None

        counts = {}
        first_appearance = {}
        
        for index, word in enumerate(words):
            if word not in counts:
                counts[word] = 0
                first_appearance[word] = index
            counts[word] += 1
            
        # Find max count
        max_count = max(counts.values())
        
        # Filter candidates with max count
        candidates = [word for word, count in counts.items() if count == max_count]
        
        # Return the one with the lowest first_appearance index
        return min(candidates, key=lambda w: first_appearance[w])

import pytest


@pytest.fixture
def processor():
    return StringProcessor()

class TestStringProcessor:

    def test_reverse_words(self, processor):
        # Test basic reversal
        assert processor.reverse_words("Hello World") == "World Hello"
        # Test multiple spaces normalization
        assert processor.reverse_words("  Python   is   great  ") == "great is Python"
        # Test single word
        assert processor.reverse_words("Single") == "Single"

    def test_count_vowels(self, processor):
        assert processor.count_vowels("Hello") == 2  # e, o
        assert processor.count_vowels("AEIOU") == 5  # Case insensitive
        assert processor.count_vowels("Rhythm") == 0 # No vowels
        assert processor.count_vowels("") == 0

    def test_is_palindrome(self, processor):
        assert processor.is_palindrome("Racecar") is True
        assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
        assert processor.is_palindrome("Hello") is False
        assert processor.is_palindrome("No 'x' in Nixon") is True

    def test_caesar_cipher(self, processor):
        # Shift by 1
        assert processor.caesar_cipher("abc", 1) == "bcd"
        # Shift wrapping around
        assert processor.caesar_cipher("xyz", 1) == "yza"
        # Negative shift
        assert processor.caesar_cipher("abc", -1) == "zab"
        # Mixed case and punctuation
        assert processor.caesar_cipher("Hello, World!", 1) == "Ifmmp, Xpsme!"

    def test_most_common_word(self, processor):
        # Basic frequency
        assert processor.most_common_word("apple banana apple cherry") == "apple"
        # Case insensitivity
        assert processor.most_common_word("Apple banana apple") == "apple"
        # Tie-breaking (first appearance)
        assert processor.most_common_word("one two one two") == "one"
        # Empty string
        assert processor.most_common_word("") is None
        # Single word
        assert processor.most_common_word("Only") == "only"