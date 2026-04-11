import functools
import os
import random
import re
from pathlib import Path

_nltk_data_dir = Path(__file__).parent / ".nltk_data"
_nltk_data_dir.mkdir(exist_ok=True)
os.environ["NLTK_DATA"] = str(_nltk_data_dir)

import nltk

nltk.data.path.insert(0, str(_nltk_data_dir))

def count_words(text):
    """Counts the number of words."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    num_words = len(tokens)
    return num_words

if __name__ == "__main__":
    print(count_words("Solve the problem first, not the pattern, Choose patterns only when they clearly fit the problem, Keep code simple and readable above all, Refactor iteratively to apply patterns naturally, Study patterns in context through real code examples, Understand the *why* behind each pattern's trade-offs. (58 words)"))
  