

import re

class StringProcessor:
    """
    A utility class for performing various string manipulations and analyses.
    """

    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in a string and collapses multiple spaces into single spaces.

        Args:
            s: The input string.

        Returns:
            A string with the word order reversed and normalized spacing.
        """
        # Split by any whitespace, filter out empty strings, reverse, and join with a single space
        words = s.split()
        return " ".join(words[::-1])

    def count_vowels(self, s: str) -> int:
        """
        Counts the total number of vowels (a, e, i, o, u) in a string, case-insensitively.

        Args:
            s: The input string.

        Returns:
            The total count of vowels.
        """
        vowels = "aeiouAEIOU"
        count = 0
        for char in s:
            if char in vowels:
                count += 1
        return count

    def is_palindrome(self, s: str) -> bool:
        """
        Checks if a string is a palindrome, ignoring case, spaces, and punctuation.

        Args:
            s: The input string.

        Returns:
            True if the string is a palindrome, False otherwise.
        """
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned_s = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        return cleaned_s == cleaned_s[::-1]

    def caesar_cipher(self, s: str, shift: int) -> str:
        """
        Applies the Caesar cipher to the string, only affecting letters (a-z, A-Z).
        Supports positive and negative shifts.

        Args:
            s: The input string.
            shift: The integer shift value.

        Returns:
            The encrypted or decrypted string.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                # Handle lowercase letters
                start = ord('a')
                new_ord = (ord(char) - start + shift) % 26 + start
                result.append(chr(new_ord))
            elif 'A' <= char <= 'Z':
                # Handle uppercase letters
                start = ord('A')
                new_ord = (ord(char) - start + shift) % 26 + start
                result.append(chr(new_ord))
            else:
                # Keep non-alphabetic characters unchanged
                result.append(char)
        return "".join(result)

    def most_common_word(self, s: str) -> Optional[str]:
        """
        Finds the most frequently occurring word in a string (case-insensitive).
        If there is a tie, the word that appears first in the text is returned.

        Args:
            s: The input string.

        Returns:
            The most common word (in its original casing from the first occurrence), 
            or None if the string is empty or contains no words.
        """
        # Tokenize and normalize words while preserving original case for tie-breaking
        words_with_case = re.findall(r'\b\w+\b', s)
        
        if not words_with_case:
            return None

        # Create a list of (lowercase_word, original_word) tuples
        normalized_words = [(word.lower(), word) for word in words_with_case]
        
        # Count frequencies of the lowercase words
        counts = Counter(item[0] for item in normalized_words)
        max_count = max(counts.values())
        
        # Find all words that match the max count
        most_common_lower_words = {
            word_lower for word_lower, _ in normalized_words 
            if counts[word_lower] == max_count
        }

        # Iterate through the original sequence to find the first occurrence 
        # of any word in the tied set
        for word_lower, original_word in normalized_words:
            if word_lower in most_common_lower_words:
                return original_word
        
        # Should not be reached if words_with_case is not empty
        return None

# Example Usage (Optional, for testing outside pytest)
if __name__ == '__main__':
    processor = StringProcessor()
    
    print("--- Reverse Words ---")
    print(f"Input: 'the quick brown fox' -> Output: '{processor.reverse_words('the quick brown fox')}'")
    print(f"Input: '  hello   world ' -> Output: '{processor.reverse_words('  hello   world ')}'")

    print("\n--- Count Vowels ---")
    print(f"Input: 'Programming' -> Output: {processor.count_vowels('Programming')}")

    print("\n--- Is Palindrome ---")
    print(f"Input: 'A man, a plan, a canal: Panama' -> Output: {processor.is_palindrome('A man, a plan, a canal: Panama')}")
    print(f"Input: 'hello' -> Output: {processor.is_palindrome('hello')}")

    print("\n--- Caesar Cipher ---")
    print(f"Input: 'abc' shift 3 -> Output: '{processor.caesar_cipher('abc', 3)}'")
    print(f"Input: 'xyz' shift -3 -> Output: '{processor.caesar_cipher('xyz', -3)}'")
    print(f"Input: 'Hello World!' shift 1 -> Output: '{processor.caesar_cipher('Hello World!', 1)}'")

    print("\n--- Most Common Word ---")
    print(f"Input: 'apple banana apple orange banana' -> Output: '{processor.most_common_word('apple banana apple orange banana')}'")
    print(f"Input: 'a b a' -> Output: '{processor.most_common_word('a b a')}'")
    print(f"Input: 'tie tie' -> Output: '{processor.most_common_word('tie tie')}'")

import pytest



@pytest.fixture
def processor():
    """Fixture to provide a fresh instance of StringProcessor for each test."""
    return StringProcessor()

# 1. reverse_words Test
def test_reverse_words_basic(processor: StringProcessor):
    """Tests basic word reversal and space collapsing."""
    assert processor.reverse_words("hello world") == "world hello"
    assert processor.reverse_words("one two three") == "three two one"

def test_reverse_words_extra_spaces(processor: StringProcessor):
    """Tests handling of leading, trailing, and multiple internal spaces."""
    input_str = "  leading   middle trailing  "
    expected = "trailing middle leading"
    assert processor.reverse_words(input_str) == expected

# 2. count_vowels Test
def test_count_vowels_case_insensitivity(processor: StringProcessor):
    """Tests vowel counting with mixed case."""
    assert processor.count_vowels("AEIOUaeiou") == 10
    assert processor.count_vowels("Rhythm") == 0
    assert processor.count_vowels("Programming") == 3 # o, a, i

# 3. is_palindrome Test
def test_is_palindrome_standard(processor: StringProcessor):
    """Tests a standard, clean palindrome."""
    assert processor.is_palindrome("racecar") == True

def test_is_palindrome_with_noise(processor: StringProcessor):
    """Tests palindrome ignoring case, spaces, and punctuation."""
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True
    assert processor.is_palindrome("Madam") == True

# 4. caesar_cipher Test
def test_caesar_cipher_positive_shift(processor: StringProcessor):
    """Tests a standard positive shift."""
    # a -> d (shift 3)
    assert processor.caesar_cipher("abc", 3) == "def"
    # Z -> C (wraps around)
    assert processor.caesar_cipher("XYZ", 3) == "ABC"
    # Mixed case and non-alpha characters
    assert processor.caesar_cipher("Hello World!", 1) == "Ifmmp Xpsme!"

def test_caesar_cipher_negative_shift(processor: StringProcessor):
    """Tests a negative shift (decryption)."""
    # d -> a (shift -3)
    assert processor.caesar_cipher("def", -3) == "abc"
    # A -> X (wraps around)
    assert processor.caesar_cipher("ABC", -3) == "XYZ"

# 5. most_common_word Test
def test_most_common_word_tie_breaking(processor: StringProcessor):
    """Tests that in case of a tie, the word appearing first is returned."""
    # 'apple' appears first, even though 'banana' appears later and also appears twice
    # Wait, 'apple' appears twice, 'banana' appears twice. 'apple' comes first.
    assert processor.most_common_word("apple banana apple orange banana") == "apple"
    
    # Clear tie: 'tie' appears first
    assert processor.most_common_word("tie tie other") == "tie"

def test_most_common_word_empty_or_no_words(processor: StringProcessor):
    """Tests edge cases for word counting."""
    assert processor.most_common_word("") is None
    assert processor.most_common_word("!@#$ %^&") is None