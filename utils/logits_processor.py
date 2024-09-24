import abc
import torch
from torch import Tensor
from torch.nn import functional as F


class LogitsProcessor(abc.ABC):
    """Logits processors for sampling."""

    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, logits: Tensor) -> Tensor:
        proc = self._process(logits)
        return F.softmax(proc / self.temperature, dim=-1)

    @abc.abstractmethod
    def _process(self, logits: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def sample(self, probs: Tensor) -> Tensor:
        pass


class GreedyProcessor(LogitsProcessor):
    """Greedy: Most probable token."""

    def __init__(self, temperature: float = 1):
        super().__init__(temperature)

    def _process(self, logits: Tensor) -> Tensor:
        return logits

    def sample(self, probs: Tensor) -> Tensor:
        return torch.argmax(probs, dim=-1).unsqueeze(-1)


class MultinomialProcessor(LogitsProcessor):
    """Multinomial: Random sampling."""

    def __init__(self, temperature: float):
        super().__init__(temperature)

    def _process(self, logits: Tensor) -> Tensor:
        return logits

    def sample(self, probs: Tensor) -> Tensor:
        return torch.multinomial(probs, num_samples=1)


class TopKProcessor(MultinomialProcessor):
    """Top-k: Top-k sampling."""

    def __init__(self, temperature: float, top_k: int):
        super().__init__(temperature)
        self.top_k = top_k

    def _process(self, logits: Tensor) -> Tensor:
        top_k = min(self.top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = -1e20
        return logits


class NucleusProcessor(MultinomialProcessor):
    """Nucleus: Top-p sampling."""

    def __init__(self, temperature: float, top_p: float):
        super().__init__(temperature)
        self.top_p = top_p

    def _process(self, logits: Tensor) -> Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_logits[sorted_indices_to_remove] = -1e20
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))
        return logits
    
    
class TopKNucleusProcessor(MultinomialProcessor):
    """Top-k and nucleus: Top-k sampling with top-p fallback."""

    def __init__(self, temperature: float, top_k: int, top_p: float):
        super().__init__(temperature)
        self.top_k = top_k
        self.top_p = top_p

    def _process(self, logits: Tensor) -> Tensor:
        top_k = min(self.top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = -1e20
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_logits[sorted_indices_to_remove] = -1e20
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))
        return logits
