import argparse
import random
import torch
from sampling import speculative_decoding, autoregressive_decoding
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    QuantoConfig,
)
from constants import DATASET
from models import NgramModel
import time
import os
from termcolor import colored


class InferenceCLI:

    def __init__(self, device: str = "cuda"):
        print(
            colored("Speculative Decoding", "red"),
            colored("CLI", on_color="on_red", color="white"),
            "\n",
        )
        self.device = device

        self.top_k = 1
        self.top_p = 0.0
        self.temperature = 1.0
        self.gamma = 4
        self.gen_len = 35
        self.debug = False
        self.spec = True
        self.dr = False
        self.cache = True
        self.target_gen = True

        self._load_models()
        self._run()

    def _load_models(self):
        # Target model
        target_model = DATASET["cnndm"]["target"]  # "google/gemma-7b"
        target_quantize = None  # QuantoConfig(weights="int8")
        target_model_type = "seq2seq"  # "causal"

        # Drafter model
        drafter_model = "2-gram"  # "google/gemma-2b"
        drafter_ngram = 2  # if 0, it won't load ngram model
        drafter_model_ckpt = "/workspace/models/nasd/cnndm_target"
        drafter_quantize = None  # QuantoConfig(weights="int8")
        drafter_model_type = "ngram"  # "causal", "seq2seq"

        print(colored("Target model:", on_color="on_yellow"), target_model)
        print(colored("Drafter model:", on_color="on_yellow"), drafter_model)
        print(colored("Loading models...", "light_grey"))

        target_class = (
            AutoModelForSeq2SeqLM
            if target_model_type == "seq2seq"
            else AutoModelForCausalLM
        )
        self.target = target_class.from_pretrained(
            target_model, quantization_config=target_quantize, device_map=self.device
        )
        self.target.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(target_model)

        if drafter_ngram == 0 or drafter_model_type != "ngram":
            drafter_class = (
                AutoModelForSeq2SeqLM
                if drafter_model_type == "seq2seq"
                else AutoModelForCausalLM
            )
            self.drafter = drafter_class.from_pretrained(
                drafter_model,
                quantization_config=drafter_quantize,
                device_map=self.device,
            )
        else:
            self.drafter = NgramModel(
                n=drafter_ngram,
                vocab_size=len(self.tokenizer),
                device=self.device,
                resume=drafter_model_ckpt,
            )
            self.drafter.to(self.device)
        self.drafter.eval()

    def _perform_command(self, command: str):
        args = command.split(" ")
        if args[0] == "/quit":
            print(colored("Goodbye!", on_color="on_red"))
            exit(0)
        if args[0] == "/debug":
            self.debug = not self.debug
            print(colored(f"Debug mode: {self.debug}", on_color="on_blue"))
            return
        if args[0] == "/speculative":
            self.spec = not self.spec
            print(
                colored(
                    f"Speculative Decoding generation: {self.spec}", on_color="on_blue"
                )
            )
            return
        if args[0] == "/drafter":
            self.dr = not self.dr
            print(colored(f"Drafter generation: {self.dr}", on_color="on_blue"))
            return
        if args[0] == "/cache":
            self.cache = not self.cache
            print(colored(f"Cache: {self.cache}", on_color="on_blue"))
            return
        if args[0] == "/target":
            self.target_gen = not self.target_gen
            print(colored(f"Target generation: {self.target_gen}", on_color="on_blue"))
            return
        if args[0] == "/temperature":
            if len(args) < 2:
                print(colored("Usage: /temperature <value>", "red"))
                return
            self.temperature = float(args[1])
            print(colored(f"Temperature: {float(args[1])}", on_color="on_blue"))
            return
        if args[0] == "/top_k":
            if len(args) < 2:
                print(colored("Usage: /top_k <value>", "red"))
                return
            self.top_k = int(args[1])
            print(colored(f"Top-k: {int(args[1])}", on_color="on_blue"))
            return
        if args[0] == "/top_p":
            if len(args) < 2:
                print(colored("Usage: /top_p <value>", "red"))
                return
            self.top_p = float(args[1])
            print(colored(f"Top-p: {float(args[1])}", on_color="on_blue"))
            return
        if args[0] == "/length":
            if len(args) < 2:
                print(colored("Usage: /length <value>", "red"))
                return
            self.gen_len = int(args[1])
            print(colored(f"Generation length: {int(args[1])}", on_color="on_blue"))
            return
        if args[0] == "/gamma":
            if len(args) < 2:
                print(colored("Usage: /gamma <value>", "red"))
                return
            self.gamma = int(args[1])
            print(colored(f"Gamma: {int(args[1])}", on_color="on_blue"))
            return
        if args[0] == "/greedy":
            self.top_k = 1
            self.top_p = 0.0
            print(colored("Greedy decoding", "green"))
            return
        if args[0] == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            return

        self._help()

    def _help(self):
        print(colored("Commands:", on_color="on_blue"))
        print("/quit: quit the program")
        print("/debug: toggle speculative debug mode")
        print(colored(f"\t{self.debug}", "green" if self.debug else "red"))
        print("/clear: clear the screen")
        print("/speculative: toggle speculative decoding")
        print(colored(f"\t{self.spec}", "green" if self.spec else "red"))
        print("/target: toggle target generation")
        print(colored(f"\t{self.target_gen}", "green" if self.target_gen else "red"))
        print("/drafter: toggle drafter generation")
        print(colored(f"\t{self.dr}", "green" if self.dr else "red"))
        print("/cache: toggle cache")
        print(colored(f"\t{self.cache}", "green" if self.cache else "red"))
        print("/temperature <value>: set temperature")
        print(colored(f"\t{self.temperature}", "blue"))
        print("/top_k <value>: set top-k")
        print(colored(f"\t{self.top_k}", "blue"))
        print("/top_p <value>: set top-p")
        print(colored(f"\t{self.top_p}", "blue"))
        print("/length <value>: set generation length")
        print(colored(f"\t{self.gen_len}", "blue"))
        print("/gamma <value>: set gamma")
        print(colored(f"\t{self.gamma}", "blue"))
        print("/greedy: set greedy decoding")

    def _infer(self, prefix: str):
        tokenized = self.tokenizer.encode(prefix, return_tensors="pt").to(self.device)

        spec_throughput = 0.0
        base_throughput = 0.0
        drafter_throughput = 0.0

        if self.spec:
            random.seed(42)
            torch.manual_seed(42)
            speculative_decoding(
                tokenized,
                self.drafter,
                self.target,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                debug=self.debug,
                debug_tokenizer=self.tokenizer,
                min_len=self.gen_len,
                gamma=self.gamma,
                end_token_id=self.tokenizer.eos_token_id,
                use_cache=self.cache,
            )
            random.seed(42)
            torch.manual_seed(42)
            spec_start_time = time.time()
            output_ids, accept_rate = speculative_decoding(
                tokenized,
                self.drafter,
                self.target,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                debug=self.debug,
                debug_tokenizer=self.tokenizer,
                min_len=self.gen_len,
                gamma=self.gamma,
                end_token_id=self.tokenizer.eos_token_id,
                use_cache=self.cache,
            )
            spec_end_time = time.time()
            spec_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(colored("========== Speculative ==========", "green"))
            print(colored("Out:", "green"), spec_output)
            print(colored(f"Accept rate: {accept_rate:.3f}", "green"))
            spec_throughput = len(spec_output) / (spec_end_time - spec_start_time)
            print(colored(f"Throughput: {spec_throughput:.1f} tokens/s", "green"))
            print(colored("========== Speculative ==========", "green"))

        if self.target_gen:
            random.seed(42)
            torch.manual_seed(42)
            autoregressive_decoding(
                tokenized,
                self.target,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_len=self.gen_len,
                end_token_id=self.tokenizer.eos_token_id,
                use_cache=self.cache,
            )
            random.seed(42)
            torch.manual_seed(42)
            start_time = time.time()
            output_ids = autoregressive_decoding(
                tokenized,
                self.target,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_len=self.gen_len,
                end_token_id=self.tokenizer.eos_token_id,
                use_cache=self.cache,
            )
            end_time = time.time()
            output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(colored("=========== Target AR ===========", "blue"))
            print(colored("Out:", "blue"), output)
            base_throughput = len(output) / (end_time - start_time)
            print(colored(f"Throughput: {base_throughput:.1f} tokens/s", "blue"))
            print(colored("=========== Target AR ===========", "blue"))
            if self.spec:
                print(
                    colored(
                        f"Throughput increase: {((spec_throughput / base_throughput)) * 100:.1f}%",
                        "magenta",
                    )
                )

        if self.dr:
            random.seed(42)
            torch.manual_seed(42)
            output_ids = autoregressive_decoding(
                tokenized,
                self.drafter,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_len=self.gen_len,
                end_token_id=self.tokenizer.eos_token_id,
                use_cache=self.cache,
            )
            output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print(colored("========== Drafter AR ==========", "cyan"))
            drafter_throughput = len(output) / (end_time - start_time)
            print(colored("Out:", "cyan"), output)
            print(colored(f"Throughput: {drafter_throughput:.1f} tokens/s", "cyan"))
            print(colored("========== Drafter AR ==========", "cyan"))

    def _run(self):
        while True:
            command = input("> ")
            if command.startswith("/"):
                self._perform_command(command)
                continue

            self._infer(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Decoding CLI")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for inference"
    )
    args = parser.parse_args()

    InferenceCLI(device=args.device)
