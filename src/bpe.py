from collections import Counter, deque
from functools import lru_cache
import json

class Tokenizer:
    def __init__(self):
        # Map id -> str
        self.vocab = {}
        # Map str -> id
        self.inverse_vocab = {}
        # BPE merges (str1, str2) -> str_merged$
        self.bpe_merges = {}
        self.bpe_ranks = {}
    
    