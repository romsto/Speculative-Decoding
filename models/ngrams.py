from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm.rich import tqdm
import os
import pickle
import gc
from transformers.file_utils import ModelOutput


@dataclass
class NgramModelOutput(ModelOutput):
    """
    Base class for ngram language model outputs.

    Args:
        logits (:obj:`torch.FloatTensor` of shape :obj:`(1, config.vocab_size)`):
            Prediction scores of the language modeling (scores for each vocabulary token).
        past_key_values (`optional`):
            Always `None` for ngram models.
    """

    logits: torch.FloatTensor = None
    past_key_values: None = None


class NgramModel(nn.Module):
    def __init__(
        self,
        n: int,
        vocab_size: int,
        device: str = "cuda",
        resume: str = None,
        fit_mode: bool = False,
        *args,
        **kwargs,
    ):
        """
        This class implements a ngram model. It is used to compute the probability of the next token given the previous prefix.
        :param n: The n in ngram. It is the number of previous tokens to consider.
        :param vocab_size: The size of the vocabulary
        :param device: The device to use for the computations
        :param resume: The path to a saved model folder to resume training, if None, the model is built from scratch
        :param fit_mode: If True, the model will not load the existing checkpoints. However the model couldn't be used for inference.
        """
        super(NgramModel, self).__init__(*args, **kwargs)

        assert n >= 1, "n must be greater than 0"

        self.__name__ = "{}-gram model".format(n)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n = n
        self.is_unigram = n == 1
        self._fitted = False
        self._fit_mode = fit_mode
        self.vocab_size = float(vocab_size)
        self.int_vocab_size = int(vocab_size)
        self.unigram_probabilities = None
        self.ngram_probabilities = defaultdict(
            lambda: torch.zeros(self.int_vocab_size, device=self.device)
        )
        self.total_counts = defaultdict(float)
        self.in_memory_ones = torch.ones(self.int_vocab_size, device=self.device)

        if resume is not None:
            self._load(resume)

        self.backoff = (
            None
            if self.is_unigram
            else NgramModel(n - 1, vocab_size, device, resume, fit_mode)
        )

    @torch.no_grad()
    def forward(self, input_ids: Tensor, *args, **kwargs) -> NgramModelOutput:
        """
        This function is used for model inference.
        :param input_ids: input_ids of the tokenized prefix
        :return: logits of the next tokens
        """

        input_ids = input_ids[0]

        # Case 1: Uni-gram model
        if self.is_unigram:
            return NgramModelOutput(logits=self.unigram_probabilities.unsqueeze(0).unsqueeze(0))

        # Case where prefix is lower than n
        # Use a backoff model
        if input_ids.shape[0] < self.n - 1 and not self.is_unigram:
            return self.backoff(input_ids)

        # Case 2: n-gram model with n > 1

        # Fetch the ending ngram from the prefix (last n-1 tokens)
        ngram = tuple(input_ids.tolist()[-(self.n - 1) :])

        # TODO check if it is really necessary
        if ngram[-1] == 1:
            return NgramModelOutput(logits=self.in_memory_ones.unsqueeze(0).unsqueeze(0))
        # End TODO

        return NgramModelOutput(
            logits=(
                torch.add(self.in_memory_ones, self.ngram_probabilities[ngram])
                / (self.total_counts[ngram] + self.vocab_size)
            ).unsqueeze(0).unsqueeze(0)
        )

    def generate(self, input_ids: Tensor) -> NgramModelOutput:
        """
        This function is used for model inference. (Same as a forward pass)
        :param input_ids: input_ids of the tokenized prefix
        :return: logits of the next token
        """
        return self.forward(input_ids)

    def fit(
        self,
        dataset,
        tokenizer,
        label_finder,
        path: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        """
        This function is used to fit the model on the data. It is not a training since the model is not trainable.
        It only builds the ngram counts based on the given data.
        :param dataset: The data to fit the model on. List of tokenized sentences on the give vocabulary
        :param tokenizer: The tokenizer used to tokenize the sentences
        :param label_finder: A function to extract the label from each element of the dataset
        :param path: The path to the folder where the model will be saved
        :param batch_size: The batch size to use for the data loader
        :param num_workers: The number of workers to use for the data loader
        """

        if self._fitted:
            print(
                f"The {self.n}-gram is already fitted. If you want to fit it again, please create a new instance."
            )
            self.backoff.fit(
                dataset, tokenizer, label_finder, path, batch_size, num_workers
            )
            return

        # ================== Counting ngrams ==================
        ngram_counts = defaultdict(lambda: defaultdict(int))

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=lambda batch: tokenizer(
                [label_finder(example) for example in batch],
                truncation=True,
                max_length=512,
                return_tensors="pt",
                padding=True,
            ),
        )

        for batch in tqdm(
            data_loader,
            desc="Counting {}-grams".format(self.n),
            total=len(data_loader),
            miniters=100,
        ):
            input_ids = batch["input_ids"]
            for sequence in input_ids:
                if not self.is_unigram:
                    for i in range(len(sequence) - self.n + 1):
                        ngram = tuple(sequence.tolist()[i : i + self.n - 1])
                        next_token = sequence[i + self.n - 1]
                        ngram_counts[ngram][next_token.item()] += 1
                        self.total_counts[ngram] += 1.0
                else:
                    for i in range(len(sequence)):
                        next_token = sequence[i].item()
                        self.total_counts[next_token] += 1.0
        # =====================================================

        # ================== Pruning ==================
        # if len(ngram_counts) > 200000:
        #     THRESOLD_PRUNING = 1
        #     count_pruned = 0
        #     print(
        #         f"{THRESOLD_PRUNING}-pruning the ngram counts... /!\\ This is a test feature. Use with caution."
        #     )
        #     ngrams_as_keys = list(ngram_counts.keys())
        #     for ngram in ngrams_as_keys:
        #         keys = list(ngram_counts[ngram].keys())
        #         if len(keys) <= THRESOLD_PRUNING:
        #             for key in keys:
        #                 self.total_counts[tuple(list(ngram[1:]) + [key])] -= 1
        #             ngram_counts.pop(ngram)
        #             count_pruned += 1
        #     print(f"{count_pruned} ngrams pruned.")
        # ============================================

        # Compute the probabilities before run-time inference
        print("Converting counts...")
        if not self.is_unigram:
            keys = ngram_counts.keys()
            for ngram in tqdm(
                keys, total=len(keys), desc="Converting counts", miniters=400
            ):
                counts = torch.zeros(self.int_vocab_size, device=self.device)
                for token, count in ngram_counts[ngram].items():
                    counts[token] = count
                counts = counts.to_sparse()
                self.ngram_probabilities[ngram] = counts

            self._save(path)

            # Clearing memory
            if self._fit_mode:
                ngram_counts = None
                self.total_counts = None
                self.ngram_probabilities = None
                torch.cuda.empty_cache()
                gc.collect()

            print("Fitting backoff model of size {}...".format(self.n - 1))
            self.backoff.fit(
                dataset, tokenizer, label_finder, path, batch_size, num_workers
            )
        else:
            counts = torch.zeros(self.int_vocab_size, device=self.device)
            for key, count in self.total_counts.items():
                counts[key] = count
            self.unigram_probabilities = (counts + 1.0) / self.vocab_size

            self._save(path)

        self._fitted = True

        if self._fit_mode:
            print(
                f"Fitting mode enabled. The {self.n}-gram model is not loaded and can't be used for inference."
            )

    def _save(self, path: str) -> None:
        """
        This function is used to save the model probabilities to disk.
        :param path: The path to the folder where the model will be saved
        """
        print("Saving {}-gram model...".format(self.n))
        if not os.path.exists(path):
            os.mkdir(path)

        if self.is_unigram:
            unigram_probabilities = self.unigram_probabilities.detach()
            with open(os.path.join(path, "{}.ncounts".format(self.n)), "wb") as f:
                pickle.dump(unigram_probabilities, f)
        else:
            ngram_probabilities = {}
            for ngram in self.ngram_probabilities.keys():
                ngram_probabilities[ngram] = self.ngram_probabilities[ngram].detach()
            with open(os.path.join(path, "{}.ncounts".format(self.n)), "wb") as f:
                pickle.dump(ngram_probabilities, f)

            total_counts = {}
            for total_count in self.total_counts.keys():
                total_counts[total_count] = self.total_counts[total_count]
            with open(os.path.join(path, "{}.tcounts".format(self.n)), "wb") as f:
                pickle.dump(total_counts, f)

    def _load(self, path: str) -> None:
        """
        This function is used to load the model probabilities from disk.
        :param path: The path to the folder where the model is saved
        :return: The ngram counts and total counts
        """
        if not os.path.exists(path):
            print(
                "Ngram folder not found. The {}-gram model will be built from scratch.".format(
                    self.n
                )
            )
            return
        if (not os.path.exists(os.path.join(path, "{}.ncounts".format(self.n)))) or (
            not os.path.exists(os.path.join(path, "{}.ncounts".format(self.n)))
            and not self.is_unigram
        ):
            print(
                "Resume file not found. The {}-gram model will be built from scratch.".format(
                    self.n
                )
            )
            return

        if self._fit_mode:
            print(f"Fitting mode enabled. Skipping the loading of {self.n}-gram.")
            self._fitted = True
            return

        print("Loading existing {}-gram model...".format(self.n))

        if self.is_unigram:
            with open(os.path.join(path, "{}.ncounts".format(self.n)), "rb") as f:
                self.unigram_probabilities = pickle.load(f).to(self.device)
        else:
            with open(os.path.join(path, "{}.ncounts".format(self.n)), "rb") as f:
                ngram_probabilities = pickle.load(f)
                keys = list(ngram_probabilities.keys())
                for ngram in tqdm(
                    keys, total=len(keys), desc="Loading ngram counts", miniters=100
                ):
                    ngram_probabilities[ngram] = ngram_probabilities[ngram].to(
                        self.device
                    )
                self.ngram_probabilities = defaultdict(
                    lambda: torch.ones(self.int_vocab_size, device=self.device),
                    ngram_probabilities,
                )
            with open(os.path.join(path, "{}.tcounts".format(self.n)), "rb") as f:
                print("Loading total counts...")
                total_counts = pickle.load(f)
                self.total_counts = defaultdict(float, total_counts)

        self._fitted = True
