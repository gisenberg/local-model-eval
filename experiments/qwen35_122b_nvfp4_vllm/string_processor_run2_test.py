import re
from typing import Optional, List
from collections import Counter

class StringProcessor:
    @staticmethod
    def reverse_words(s: str) -> str:
        """
        Reverses the order of words in the input string.
        
        Consecutive whitespaces are normalized to single spaces.
        Leading/trailing whitespaces are removed.
        
        Example:
            "The quick brown" -> "brown quick The"
            
        Args:
            s: The input string.
            
        Returns:
            A string with words in reverse order.
        """
        # split() handles multiple spaces, leading/trailing spaces automatically
        words = s.split()
        return " ".join(words[::-1])

    @staticmethod
    def count_vowels(s: str) -> int:
        """
        Counts the number of vowels in the input string (case-insensitive).
        
        Vowels are defined as 'a', 'e', 'i', 'o', 'u'.
        
        Args:
            s: The input string.
            
        Returns:
            The integer count of vowels.
        """
        vowels = set("aeiouAEIOU")
        return sum(1 for char in s if char in vowels)

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """
        Determines if the string is a palindrome.
        
        Comparison ignores case, spaces, and punctuation. Only alphanumeric 
        characters are considered.
        
        Example:
            "A man, a plan..." -> True
            
        Args:
            s: The input string.
            
        Returns:
            True if the cleaned string is a palindrome, False otherwise.
        """
        # Remove non-alphanumeric characters and lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned == cleaned[::-1]

    @staticmethod
    def caesar_cipher(s: str, shift: int) -> str:
        """
        Encrypts the string using the Caesar Cipher algorithm.
        
        Supports shifting both positive and negative integers. Wraps around
        'z'/'Z' to 'a'/'A'. Non-alphabetical characters remain unchanged.
        
        Args:
            s: The input string.
            shift: The integer amount to shift (positive or negative).
            
        Returns:
            The encrypted string.
        """
        result = []
        for char in s:
            if char.isalpha():
                # Determine ASCII offset based on case
                ascii_offset = ord('a') if char.islower() else ord('A')
                # Calculate shifted index with wrap-around
                # Modulo arithmetic handles negative shifts correctly
                new_index = (ord(char) - ascii_offset + shift) % 26
                result.append(chr(new_index + ascii_offset))
            else:
                result.append(char)
        return "".join(result)

    @staticmethod
    def most_common_word(s: str) -> Optional[str]:
        """
        Finds the most common word in the string.
        
        Case-insensitive. Ignores non-word characters.
        In case of a tie, returns the word that appears first in the text.
        
        Args:
            s: The input string.
            
        Returns:
            The most common word string, or None if no words exist.
        """
        # Extract words, converting to lowercase
        words = re.findall(r'\w+', s.lower())
        
        if not words:
            return None
            
        counts = Counter(words)
        max_count = max(counts.values())
        
        # Find the first word in the Counter that matches the max count.
        # In Python 3.7+, Counter maintains insertion order, so iterating
        # items() yields words in order of first appearance in the text.
        for word, count in counts.items():
            if count == max_count:
                return word
        
        return None


# ==================== PYTEST TESTS ====================
# Run these tests using: pytest tests.py -v

def test_reverse_words_basic():
    processor = StringProcessor()
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("The quick brown") == "brown quick The"

def test_count_vowels_mixed_case():
    processor = StringProcessor()
    # AEIOU, aeiou = 10 vowels
    assert processor.count_vowels("AEIOU aeiou") == 10
    # y is not counted here
    assert processor.count_vowels("fly my sky") == 0
    assert processor.count_vowels("Hello World") == 3

def test_is_palindrome_with_punctuation():
    processor = StringProcessor()
    assert processor.is_palindrome("Racecar") is True
    assert processor.is_palindrome("A man, a plan, a canal: Panama") is True
    assert processor.is_palindrome("Not a palindrome") is False

def test_caesar_cipher_edge_cases():
    processor = StringProcessor()
    # Basic positive shift
    assert processor.caesar_cipher("abc", 1) == "bcd"
    # Wrap around positive
    assert processor.caesar_cipher("xyz", 3) == "abc"
    # Negative shift
    assert processor.caesar_cipher("cba", -1) == "baz"
    # Preserve non-alpha
    assert processor.caesar_cipher("hello, 123!", 1) == "ifmmp, 123!"

def test_most_common_word_tie_breaking():
    processor = StringProcessor()
    # No words
    assert processor.most_common_word("!@#$%") is None
    
    # Simple case
    assert processor.most_common_word("apple banana apple") == "apple"
    
    # Tie-breaker: 'cat' appears first in text than 'dog'
    # Both appear twice
    text = "cat dog cat dog"
    assert processor.most_common_word(text) == "cat"