from typing import Optional
import re

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """Return the input string with the order of words reversed."""
        return ' '.join(s.split()[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """Count the number of vowels (case-insensitive) in the string."""
        vowels = set('aeiouAEIOU')
        return sum(1 for ch in s if ch in vowels)

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Check if the string is a palindrome ignoring case, spaces, and punctuation."""
        cleaned = re.sub(r'[^A-Za-z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """Apply Caesar cipher to alphabetic characters with given shift (supports negative)."""
        result = []
        for ch in s:
            if 'a' <= ch <= 'z':
                result.append(chr((ord(ch) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= ch <= 'Z':
                result.append(chr((ord(ch) - ord('A') + shift) % 26 + ord('A')))
            else:
                result.append(ch)
        return ''.join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """Return the most common word (case-insensitive). If tie, return the first occurring word. Return None if no words."""
        words = re.findall(r'[A-Za-z]+', s)
        if not words:
            return None
        count = {}
        first_index = {}
        best_word = None
        best_count = 0
        best_first = float('inf')

        for i, w in enumerate(words):
            w_lower = w.lower()
            if w_lower not in count:
                count[w_lower] = 0
                first_index[w_lower] = i
            count[w_lower] += 1
            if count[w_lower] > best_count or (count[w_lower] == best_count and first_index[w_lower] < best_first):
                best_count = count[w_lower]
                best_first = first_index[w_lower]
                best_word = w_lower
        return best_word

# ─────────────── Pytest Tests ───────────────

def test_reverse_words():
    sp = StringProcessor()
    assert sp.reverse_words("hello world") == "world hello"
    assert sp.reverse_words("a  b  c") == "c b a"

def test_count_vowels():
    sp = StringProcessor()
    assert sp.count_vowels("Hello") == 2
    assert sp.count_vowels("AEIOU") == 5

def test_is_palindrome():
    sp = StringProcessor()
    assert sp.is_palindrome("A man, a plan, a canal: Panama") is True
    assert sp.is_palindrome("race a car") is False

def test_caesar_cipher():
    sp = StringProcessor()
    assert sp.caesar_cipher("abc", 1) == "bcd"
    assert sp.caesar_cipher("xyz", -1) == "wxy"
    assert sp.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"

def test_most_common_word():
    sp = StringProcessor()
    assert sp.most_common_word("the cat sat on the mat") == "the"
    assert sp.most_common_word("a b a b c c") == "a"          # tie, first occurrence wins
    assert sp.most_common_word("Hello hello") == "hello"      # case-insensitive