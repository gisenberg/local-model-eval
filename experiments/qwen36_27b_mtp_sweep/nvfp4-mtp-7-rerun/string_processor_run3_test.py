import re
from typing import Optional

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Reverse the order of words in the string.
        Extra whitespace is normalized to single spaces."""
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels in the string (case-insensitive)."""
        return sum(1 for char in s if char.lower() in 'aeiou')

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if the string is a palindrome, ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply a Caesar cipher to alphabetic characters only.
        Non-alphabetic characters remain unchanged. Supports negative shifts."""
        shift = shift % 26
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= char <= 'Z':
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Find the most common word in the string (case-insensitive).
        Returns the first word encountered if there's a tie. Returns None if no words exist."""
        words = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        if not words:
            return None

        counts: dict[str, int] = {}
        order: list[str] = []
        for word in words:
            if word not in counts:
                counts[word] = 0
                order.append(word)
            counts[word] += 1

        max_count = max(counts.values())
        for word in order:
            if counts[word] == max_count:
                return word
        return None


# ==================== PYTEST TESTS ====================

def test_reverse_words():
    assert StringProcessor.reverse_words("hello world") == "world hello"
    assert StringProcessor.reverse_words("  python is great  ") == "great is python"
    assert StringProcessor.reverse_words("single") == "single"

def test_count_vowels():
    assert StringProcessor.count_vowels("Hello World") == 3
    assert StringProcessor.count_vowels("Rhythm") == 0
    assert StringProcessor.count_vowels("AEIOU") == 5

def test_is_palindrome():
    assert StringProcessor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert StringProcessor.is_palindrome("Race car!") is True
    assert StringProcessor.is_palindrome("Hello") is False

def test_caesar_cipher():
    assert StringProcessor.caesar_cipher("abc", 1) == "bcd"
    assert StringProcessor.caesar_cipher("XYZ", -1) == "WXY"
    assert StringProcessor.caesar_cipher("a!b@c", 2) == "c!d@e"
    assert StringProcessor.caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"

def test_most_common_word():
    assert StringProcessor.most_common_word("apple banana apple orange banana apple") == "apple"
    assert StringProcessor.most_common_word("cat dog cat dog") == "cat"  # Tie broken by first occurrence
    assert StringProcessor.most_common_word("") is None
    assert StringProcessor.most_common_word("123 !@#") is None