from typing import List, Dict, Tuple
from collections import defaultdict

# Sample data
data = [
    ("ABACUS", ["AE", "B", "AH", "K", "AH", "S"]),
    ("BOOK", ["B", "UH", "K"]),
    ("THEIR", ["DH", "EH", "R"]),
    ("THERE", ["DH", "EH", "R"]),
    ("TOMATO", ["T", "AH", "M", "AA", "T", "OW"]),
    ("TOMATO", ["T", "AH", "M", "EY", "T", "OW"])
]
# Main fun to find the combinations.
def find_word_combos_with_pronunciation(phonemes: List[str]) -> List[List[str]]:
    d = preprocess_pronunciation_dictionary(data)
    results = []
    find_combinations(tuple(phonemes), 0, [], results, d)
    return results
  
# Preprocess the dictionary
def preprocess_pronunciation_dictionary(data: List[Tuple[str, List[str]]]) -> Dict[Tuple[str, ...], List[str]]:
    d = defaultdict(list)
    for word, phonemes in data:
        d[tuple(phonemes)].append(word)

    return d

# Recursive function to find all possible combinations
def find_combinations(sequence: Tuple[str, ...], start: int, current: List[str], results: List[List[str]], d: Dict[Tuple[str, ...], List[str]]):
    if start == len(sequence):
        results.append(current.copy())
        return
    for i in range(start, len(sequence)):
        phonemes = sequence[start:i+1]
        if phonemes in d:
            for word in d[phonemes]:
                current.append(word)

                find_combinations(sequence, i + 1, current, results, d)
                current.pop()
