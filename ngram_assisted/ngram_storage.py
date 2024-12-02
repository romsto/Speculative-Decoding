import abc
from typing import Tuple
import torch

class INgramStorage(abc.ABC):
    """
    Interface of Ngram-Storage. It adapts dynamically on the seen ngrams, and return the most likely token based on the (n-1) tokens.
    """
    
    def __init__(self, n: int, vocab_size: int):
        assert n > 1, "n should be greater than 1"
        
        self.n = n
        self.vocab_size = vocab_size
    
    @abc.abstractmethod
    def next_token(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the next token based on the input_ids

        Args:
            input_ids: torch.Tensor of shape (batch_size, seq_len)

        Returns:
            torch.Tensor: torch.Tensor of shape (batch_size, 1) with the most likely token
            torch.Tensor: torch.Tensor of shape (batch_size, 1) with whether the token is known or not
        """
        pass
    
    @abc.abstractmethod
    def has_gram(self, ngram: torch.Tensor) -> bool:
        """
        Check if the input_ids has been seen before

        Args:
            mgram: torch.Tensor of shape (self.n)

        Returns:
            bool: True if the ngram has been seen before, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def update(self, input_ids: torch.Tensor, next_tokens: torch.Tensor):
        """
        Update the model with the input_ids and next_tokens

        Args:
            input_ids: torch.Tensor of shape (batch_size, seq_len)
            next_tokens: torch.Tensor of shape (batch_size, 1 or more)
        """
        pass
    
    @abc.abstractmethod
    def initialize(self, input_ids: torch.Tensor):
        """
        Initialize the ngrams with the given input_ids.

        Args:
            input_ids: torch.Tensor of shape (batch_size, seq_len)
        """
        pass
    
    @abc.abstractmethod
    def reset(self):
        """
        Reset the model
        """
        pass
    


class OneLevelNGramStorage(INgramStorage):
    """
    Implementation of Ngram-Storage. It only handles n-grams and not k-grams with k in [2, n].
    """

    def __init__(self, n: int, vocab_size: int):
        super().__init__(n, vocab_size)
        self.counts = {}  # gram -> {token -> count}
        self.ngrams = {}  # gram -> token with the highest count

    def next_token(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = torch.randint(self.vocab_size, size=(input_ids.shape[0],))
        known = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for i, seq in enumerate(input_ids):
            if seq.shape[0] < self.n - 1:
                continue

            gram = tuple(seq[-(self.n - 1) :].tolist())

            if gram in self.ngrams:
                out[i] = self.ngrams[gram]
                known[i] = True

        return out, known

    def has_gram(self, ngram: torch.Tensor) -> bool:
        if ngram.shape[0] < self.n:
            return False

        gram = tuple(ngram[-(self.n - 1) :].tolist())
        if not gram in self.counts:
            return False

        return ngram[-1].item() in self.counts[gram]

    def update(self, input_ids: torch.Tensor, next_tokens: torch.Tensor):
        for i, seq in enumerate(input_ids):
            if seq.shape[0] < self.n:
                continue

            gram = tuple(seq[-(self.n - 1) :].tolist())
            tokens = next_tokens[i].tolist()

            if gram not in self.counts:
                self.counts[gram] = {}
            if gram not in self.ngrams:
                self.ngrams[gram] = tokens[0]

            for token in tokens:
                if token not in self.counts[gram]:
                    self.counts[gram][token] = 1
                else:
                    current = self.counts[gram][token] + 1
                    self.counts[gram][token] = current
                    if current > self.counts[gram][self.ngrams[gram]]:
                        self.ngrams[gram] = token

    def initialize(self, input_ids: torch.Tensor):
        for seq in input_ids:
            for i in range(seq.shape[0] - self.n + 1):
                gram = tuple(seq[i : i + self.n - 1].tolist())
                token = seq[i + self.n - 1].item()

                if gram not in self.counts:
                    self.counts[gram] = {}
                if gram not in self.ngrams:
                    self.ngrams[gram] = token

                if token not in self.counts[gram]:
                    self.counts[gram][token] = 1
                else:
                    self.counts[gram][token] += 1
                    if self.counts[gram][token] > self.counts[gram][self.ngrams[gram]]:
                        self.ngrams[gram] = token

    def reset(self):
        self.counts = {}
        self.ngrams = {}



class NGramStorage(INgramStorage):
    """
    Implementation of Ngram-Storage.
    """
    
    def __init__(self, n: int, vocab_size: int):
        super().__init__(n, vocab_size)
        self.counts = {} # i-gram (i in [2, n]) -> {gram -> {token -> count}}
        self.ngrams = {} # i-gram (i in [2, n]) -> {gram -> token with the highest count}
        
    def next_token(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = torch.randint(self.vocab_size, size=(input_ids.shape[0],))
        known = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for i, seq in enumerate(input_ids):
            if seq.shape[0] < 1:
                continue
            
            for j in range(min(self.n - 1, seq.shape[0]), 1, -1):    
                gram = tuple(seq[-j:].tolist())
                
                if gram in self.ngrams[j]:
                    out[i] = self.ngrams[j][gram]
                    known[i] = True
                    break
        
        return out, known
    
    def has_gram(self, ngram: torch.Tensor) -> bool:
        if ngram.shape[0] < 1:
            return False
        
        for j in range(min(self.n - 1, ngram.shape[0]), 1, -1):            
            gram = tuple(ngram[-j:].tolist())
            if not gram in self.counts[j]:
                continue
            
            if ngram[-1].item() in self.counts[j][gram]:
                return True
        
        return False
    
    def update(self, input_ids: torch.Tensor, next_tokens: torch.Tensor):
        for i, seq in enumerate(input_ids):
            if seq.shape[0] < 1:
                continue
            
            for j in range(min(self.n - 1, seq.shape[0]), 1, -1):
                gram = tuple(seq[-j:].tolist())
                tokens = next_tokens[i].tolist()
                
                if j not in self.counts:
                    self.counts[j] = {}
                if j not in self.ngrams:
                    self.ngrams[j] = {}
                
                if gram not in self.counts[j]:
                    self.counts[j][gram] = {}
                if gram not in self.ngrams[j]:
                    self.ngrams[j][gram] = tokens[0]
                
                for token in tokens:
                    if token not in self.counts[j][gram]:
                        self.counts[j][gram][token] = 1
                    else:
                        current = self.counts[j][gram][token] + 1
                        self.counts[j][gram][token] = current
                        if current > self.counts[j][gram][self.ngrams[j][gram]]:
                            self.ngrams[j][gram] = token
                            
    def initialize(self, input_ids: torch.Tensor):
        for seq in input_ids:
            for i in range(seq.shape[0]):
                for j in range(min(self.n - 1, i), 1, -1):
                    gram = tuple(seq[i - j : i].tolist())
                    token = seq[i].item()
                    
                    if j not in self.counts:
                        self.counts[j] = {}
                    if j not in self.ngrams:
                        self.ngrams[j] = {}
                    
                    if gram not in self.counts[j]:
                        self.counts[j][gram] = {}
                    if gram not in self.ngrams[j]:
                        self.ngrams[j][gram] = token
                    
                    if token not in self.counts[j][gram]:
                        self.counts[j][gram][token] = 1
                    else:
                        self.counts[j][gram][token] += 1
                        if self.counts[j][gram][token] > self.counts[j][gram][self.ngrams[j][gram]]:
                            self.ngrams[j][gram] = token
    
    def reset(self):
        self.counts = {}
        self.ngrams = {}
