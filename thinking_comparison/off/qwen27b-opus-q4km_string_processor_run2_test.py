from typing import Optional
import string


class StringProcessor:
    """A class providing various string manipulation and analysis methods."""

    def reverse_words(self, s: str) -> str:
        """
        Reverse the order of words in a string (not the characters).
        Multiple spaces between words become a single space.
        Leading/trailing spaces are removed.
        
        Args:
            s: Input string
            
        Returns:
            String with words in reversed order
        """
        if not s or not s.strip():
            return ""
        
        # Split by whitespace (handles multiple spaces), filter empty strings
        words = [word for word in s.split() if word]
        
        # Reverse the list of words and join with single space
        return " ".join(reversed(words))

    def count_vowels(self, s: str) -> int:
        """
        Count vowels (a, e, i, o, u) in the string, case-insensitive.
        
        Args:
            s: Input string
            
        Returns:
            Integer count of vowels
        """
        vowels = set('aeiouAEIOU')
        return sum(1 for char in s if char in vowels)

    def is_palindrome(self, s: str) -> bool:
        """
        Check if the string is a palindrome, ignoring case, spaces, and punctuation.
        
        Args:
            s: Input string
            
        Returns:
            True if palindrome, False otherwise
        """
        # Keep only alphanumeric characters and convert to lowercase
        cleaned = ''.join(char.lower() for char in s if char.isalnum())
        
        # Check if cleaned string equals its reverse
        return cleaned == cleaned[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Apply Caesar cipher with given shift. Only shift a-z and A-Z,
        leave other characters unchanged. Supports negative shifts.
        
        Args:
            s: Input string
            shift: Number of positions to shift (can be negative)
            
        Returns:
            Encrypted/decrypted string
        """
        result = []
        
        for char in s:
            if 'a' <= char <= 'z':
                # Shift lowercase letters
                shifted = (ord(char) - ord('a') + shift) % 26
                result.append(chr(ord('a') + shifted))
            elif 'A' <= char <= 'Z':
                # Shift uppercase letters
                shifted = (ord(char) - ord('A') + shift) % 26
                result.append(chr(ord('A') + shifted))
            else:
                # Leave other characters unchanged
                result.append(char)
        
        return ''.join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Return the most frequently occurring word (case-insensitive).
        If tied, return the one that appears first.
        Return None for empty strings.
        
        Args:
            s: Input string
            
        Returns:
            Most common word or None if string is empty
        """
        if not s or not s.strip():
            return None
        
        # Split into words and convert to lowercase
        words = s.split()
        
        if not words:
            return None
        
        # Count occurrences while preserving order
        word_count = {}
        word_order = []
        
        for word in words:
            lower_word = word.lower()
            if lower_word not in word_count:
                word_count[lower_word] = 0
                word_order.append(lower_word)
            word_count[lower_word] += 1
        
        # Find the word with maximum count (first one in case of tie)
        max_count = max(word_count.values())
        
        for word in word_order:
            if word_count[word] == max_count:
                return word
        
        return None


# Pytest tests
def test_reverse_words():
    processor = StringProcessor()
    
    # Test basic reversal
    assert processor.reverse_words("hello world") == "world hello"
    
    # Test multiple spaces
    assert processor.reverse_words("hello    world") == "world hello"
    
    # Test leading/trailing spaces
    assert processor.reverse_words("  hello world  ") == "world hello"
    
    # Test single word
    assert processor.reverse_words("single") == "single"
    
    # Test empty string
    assert processor.reverse_words("") == ""


def test_count_vowels():
    processor = StringProcessor()
    
    # Test basic counting
    assert processor.count_vowels("hello") == 2  # e, o
    
    # Test case insensitivity
    assert processor.count_vowels("HELLO") == 2
    
    # Test mixed case
    assert processor.count_vowels("HeLLo WoRLd") == 5  # e, o, o, o
    
    # Test no vowels
    assert processor.count_vowels("rhythm") == 0
    
    # Test empty string
    assert processor.count_vowels("") == 0


def test_is_palindrome():
    processor = StringProcessor()
    
    # Test simple palindrome
    assert processor.is_palindrome("racecar") == True
    
    # Test with spaces
    assert processor.is_palindrome("A man a plan a canal Panama") == True
    
    # Test with punctuation
    assert processor.is_palindrome("Able was I, ere I saw Elba!") == True
    
    # Test not a palindrome
    assert processor.is_palindrome("hello") == False
    
    # Test case insensitivity
    assert processor.is_palindrome("RaceCar") == True


def test_caesar_cipher():
    processor = StringProcessor()
    
    # Test basic encryption
    assert processor.caesar_cipher("abc", 1) == "bcd"
    
    # Test uppercase
    assert processor.caesar_cipher("ABC", 1) == "BCD"
    
    # Test wrapping (z -> a)
    assert processor.caesar_cipher("xyz", 1) == "yza"
    
    # Test negative shift (decryption)
    assert processor.caesar_cipher("bcd", -1) == "abc"
    
    # Test non-alphabetic characters preserved
    assert processor.caesar_cipher("Hello, World!", 3) == "Khoor, Zruog!"
    
    # Test shift of 26 (should be identity)
    assert processor.caesar_cipher("abc", 26) == "abc"


def test_most_common_word():
    processor = StringProcessor()
    
    # Test basic most common
    assert processor.most_common_word("the quick brown fox jumps over the lazy dog") == "the"
    
    # Test case insensitivity
    assert processor.most_common_word("The THE the") == "the"
    
    # Test tie - returns first occurrence
    assert processor.most_common_word("one two one two") == "one"
    
    # Test single word
    assert processor.most_common_word("single") == "single"
    
    # Test empty string returns None
    assert processor.most_common_word("") is None
    
    # Test whitespace only returns None
    assert processor.most_common_word("   ") is None