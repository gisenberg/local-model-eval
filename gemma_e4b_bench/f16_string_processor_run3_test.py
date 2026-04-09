import re



class StringProcessor:
    """
    A utility class for performing various string manipulation tasks.
    """

    def reverse_words(self, s: str) -> str:
        """
        Reverses the order of words in a string and collapses multiple spaces into a single space.

        Args:
            s: The input string.

        Returns:
            The string with words reversed and normalized spacing.
        """
        words = s.split()
        return " ".join(words[::-1])

    def count_vowels(self, s: str) -> int:
        """
        Counts the total number of vowels (a, e, i, o, u) in a string, case-insensitive.

        Args:
            s: The input string.

        Returns:
            The total count of vowels.
        """
        vowels = "aeiouAEIOU"
        return sum(1 for char in s if char in vowels)

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
        Applies the Caesar cipher to alphabetic characters in a string.

        Args:
            s: The input string.
            shift: The integer shift value (positive or negative).

        Returns:
            The encrypted/decrypted string.
        """
        result = []
        for char in s:
            if 'a' <= char <= 'z':
                start = ord('a')
                new_ord = (ord(char) - start + shift) % 26 + start
                result.append(chr(new_ord))
            elif 'A' <= char <= 'Z':
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
        Returns the first word encountered if there is a tie.

        Args:
            s: The input string.

        Returns:
            The most common word (lowercase), or None if the string is empty or contains no words.
        """
        # Use regex to find all sequences of word characters
        words = re.findall(r'\b\w+\b', s.lower())
        
        if not words:
            return None
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Find the maximum frequency
        max_count = max(word_counts.values())
        
        # Find all words that match the max frequency
        most_common = [word for word, count in word_counts.items() if count == max_count]
        
        # Return the first one encountered (as per requirement)
        return most_common[0]

# Example Usage (optional, for testing outside pytest)
if __name__ == '__main__':
    processor = StringProcessor()
    
    print("--- Reverse Words ---")
    print(f"'Hello   world' -> '{processor.reverse_words('Hello   world')}'") # dlrow Hello
    
    print("\n--- Count Vowels ---")
    print(f"'Programming' -> {processor.count_vowels('Programming')}") # 3 (o, a, i)
    
    print("\n--- Is Palindrome ---")
    print(f"'A man, a plan, a canal: Panama' -> {processor.is_palindrome('A man, a plan, a canal: Panama')}") # True
    
    print("\n--- Caesar Cipher ---")
    print(f"'abc' shifted by 3 -> '{processor.caesar_cipher('abc', 3)}'") # def
    print(f"'xyz' shifted by -3 -> '{processor.caesar_cipher('xyz', -3)}'") # abc
    
    print("\n--- Most Common Word ---")
    print(f"'The quick brown fox jumps over the lazy fox' -> {processor.most_common_word('The quick brown fox jumps over the lazy fox')}") # the

import pytest



@pytest.fixture
def processor() -> StringProcessor:
    """Fixture to provide a fresh instance of StringProcessor for each test."""
    return StringProcessor()

# 1. Test reverse_words
def test_reverse_words_basic(processor: StringProcessor):
    """Test basic word reversal and space collapsing."""
    assert processor.reverse_words("one two three") == "three two one"

def test_reverse_words_extra_spaces(processor: StringProcessor):
    """Test handling of multiple spaces between words."""
    assert processor.reverse_words("  leading   and   trailing  ") == "trailing and leading"

# 2. Test count_vowels
def test_count_vowels_case_insensitive(processor: StringProcessor):
    """Test vowel counting with mixed case."""
    assert processor.count_vowels("AEIOUaeiou") == 10
    assert processor.count_vowels("Rhythm") == 0
    assert processor.count_vowels("Hello World") == 3 # e, o, o

# 3. Test is_palindrome
def test_is_palindrome_standard(processor: StringProcessor):
    """Test a classic palindrome."""
    assert processor.is_palindrome("Racecar") == True

def test_is_palindrome_with_punctuation_and_spaces(processor: StringProcessor):
    """Test palindrome ignoring case, spaces, and punctuation."""
    assert processor.is_palindrome("A man, a plan, a canal: Panama") == True

# 4. Test caesar_cipher
def test_caesar_cipher_positive_shift(processor: StringProcessor):
    """Test positive shift (encryption)."""
    assert processor.caesar_cipher("abc", 3) == "def"
    assert processor.caesar_cipher("XYZ", 1) == "YZA" # Wraps around

def test_caesar_cipher_negative_shift(processor: StringProcessor):
    """Test negative shift (decryption)."""
    assert processor.caesar_cipher("def", -3) == "abc"
    assert processor.caesar_cipher("ABC", -1) == "ZAB" # Wraps around

# 5. Test most_common_word
def test_most_common_word_basic(processor: StringProcessor):
    """Test finding the single most common word."""
    text = "apple banana apple orange banana apple"
    assert processor.most_common_word(text) == "apple"

def test_most_common_word_tie_breaker(processor: StringProcessor):
    """Test tie-breaking (should return the first one encountered)."""
    # 'the' appears first, and both 'the' and 'a' appear twice.
    text = "the a the a"
    assert processor.most_common_word(text) == "the"

def test_most_common_word_empty_string(processor: StringProcessor):
    """Test handling of an empty input string."""
    assert processor.most_common_word("") is None

def test_most_common_word_no_words(processor: StringProcessor):
    """Test handling of a string with only punctuation/spaces."""
    assert processor.most_common_word("!@#$%^&*()") is None