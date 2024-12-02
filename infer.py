import argparse
import random
import numpy as np
import torch
from sampling import autoregressive_generate, speculative_generate
from ngram_assisted import OneLevelNGramStorage, NGramStorage, ngram_assisted_speculative_generate
from utils.logits_processor import GreedyProcessor, MultinomialProcessor, TopKProcessor, NucleusProcessor, TopKNucleusProcessor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    QuantoConfig,
)
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

        self.gamma = 4
        self.gen_len = 35
        self.debug = False
        self.spec = True
        self.dr = False
        self.cache = False
        self.target_gen = True
        # Ngram Assisted Generation
        self.ngram_gen = True
        self.ngram = None
        self.top_k_filler = 3
        self.ngram_n = 3
        self.reset_in_between = True
        
        self.chat = True # If using a chat instructed model, set to True
        
        self.processors = {
            "greedy": {
                "processor": GreedyProcessor,
                "building_args": {"temperature": float},
            },
            "multinomial": {
                "processor": MultinomialProcessor,
                "building_args": {"temperature": float},
            },
            "topk": {
                "processor": TopKProcessor,
                "building_args": {"temperature": float, "top_k": int},
            },
            "nucleus": {
                "processor": NucleusProcessor,
                "building_args": {"temperature": float, "top_p": float},
            },
            "topknucleus": {
                "processor": TopKNucleusProcessor,
                "building_args": {"temperature": float, "top_k": int, "top_p": float},
            },
        }
        self.selected_processor = {
            "name": "greedy",
            "processor": GreedyProcessor,
            "args": {"temperature": 1.0},
        }
        self.processor = GreedyProcessor()

        self._load_models()
        self._run()

    def _load_models(self):
        # Target model
        target_model = "meta-llama/Llama-3.2-3B-Instruct"
        target_quantize = QuantoConfig(weights="int8")  # QuantoConfig(weights="int8")  None
        
        # Drafter model
        drafter_model = "meta-llama/Llama-3.2-1B-Instruct"
        drafter_quantize = QuantoConfig(weights="int8")  # QuantoConfig(weights="int8") None

        print(colored("Target model:", on_color="on_yellow"), target_model)
        print(colored("Drafter model:", on_color="on_yellow"), drafter_model)
        print(colored("Loading models...", "light_grey"))

        self.target = AutoModelForCausalLM.from_pretrained(
            target_model,
            quantization_config=target_quantize,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.target.eval()

        tokenizer_name = target_model
        if tokenizer_name != target_model:
            print(colored("Warning: Tokenizer is different from target model. Use with caution.", "red"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        self.drafter = AutoModelForCausalLM.from_pretrained(
            drafter_model,
            quantization_config=drafter_quantize,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.drafter.eval()
        
        self.ngram = NGramStorage(n=3, vocab_size=self.target.config.vocab_size)
        
        self.end_tokens = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")] # "<|eot_id|>" is the end of turn token for Llama model.

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
            print(colored(f"Speculative Decoding generation: {self.spec}", on_color="on_blue"))
            return
        if args[0] == "/drafter":
            self.dr = not self.dr
            print(colored(f"Drafter generation: {self.dr}", on_color="on_blue"))
            return
        if args[0] == "/cache":
            self.cache = not self.cache
            print(colored(f"Cache: {self.cache}", on_color="on_blue"))
            if self.cache:
                print(colored("Warning, cache feature is very unstable accross different models. It might generate errors or just perturb the generation. Use with caution.", "red"))
            return
        if args[0] == "/target":
            self.target_gen = not self.target_gen
            print(colored(f"Target generation: {self.target_gen}", on_color="on_blue"))
            return
        if args[0] == "/chat":
            self.chat = not self.chat
            print(colored(f"Chat mode: {self.chat}", on_color="on_blue"))
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
        if args[0] == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            return
        if args[0] == "/processor":
            # /processor <processor_name> <args0> <args1> ...
            if len(args) < 2:
                print(colored("Usage: /processor <processor_name> <args0> <args1> ...", "red"))
                return
            processor_name = args[1]
            if processor_name not in self.processors:
                print(colored("Invalid processor name", "red"))
                print(colored("Available processors:", "red"))
                for processor in self.processors.keys():
                    print(colored(f"\t{processor}", "red"))
                return
            processor = self.processors[processor_name]
            print(colored(f"Selected processor: {processor_name}", "blue"))
            building_args = processor["building_args"]
            args = args[2:]
            processor_args = {}
            for arg_name, arg_type in building_args.items():
                if len(args) == 0:
                    print(colored(f"Missing argument {arg_name}", "red"))
                    return
                try:
                    processor_args[arg_name] = arg_type(args[0])
                    print(colored(f"\t{arg_name}: {arg_type(args[0])}", "blue"))
                except ValueError:
                    print(colored(f"Invalid argument {arg_name} of type {arg_type}", "red"))
                    return
                args = args[1:]
            self.selected_processor = {
                "name": processor_name,
                "processor": processor["processor"],
                "args": processor_args,
            }
            self.processor = processor["processor"](**processor_args)
            return
        # Ngram Assisted Generation
        if args[0] == "/ngram":
            self.ngram_gen = not self.ngram_gen
            print(colored(f"Ngram assisted generation: {self.ngram_gen}", on_color="on_blue"))
            return
        if args[0] == "/top_k_filler":
            if len(args) < 2:
                print(colored("Usage: /top_k_filler <value>", "red"))
                return
            self.top_k_filler = int(args[1])
            print(colored(f"Top k filler: {int(args[1])}", on_color="on_blue"))
            return
        if args[0] == "/set_ngramstorage":
            if len(args) < 3:
                print(colored("Usage: /set_ngramstorage <basic/onelevel> <n>", "red"))
                return
            if args[1] == "onelevel":
                ntype = OneLevelNGramStorage
            elif args[1] == "basic":
                ntype = NGramStorage
            else:
                print(colored("Invalid ngram type", "red"))
                return
            self.ngram = ntype(n=int(args[2]), vocab_size=self.target.config.vocab_size)
            self.ngram_n = int(args[2])
            print(colored(f"Ngram type: {args[1]}", "blue"))
            print(colored(f"Ngram n: {int(args[2])}", "blue"))
            return
        if args[0] == "/reset_in_between":
            self.reset_in_between = not self.reset_in_between
            print(colored(f"Reset ngram in between each generation: {self.reset_in_between}", on_color="on_blue"))
            return
        print(colored("Unknown command", "red"))
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
        print("/chat: toggle chat mode")
        print(colored(f"\t{self.chat}", "green" if self.chat else "red"))
        print("/length <value>: set generation length")
        print(colored(f"\t{self.gen_len}", "blue"))
        print("/gamma <value>: set gamma")
        print(colored(f"\t{self.gamma}", "blue"))
        print("/processor <processor_name> [args0] [args1] ...: set processor")
        print(colored(f"\t{self.selected_processor['name']}", "blue"))
        for arg_name, arg_value in self.selected_processor["args"].items():
            print(colored(f"\t\t{arg_name}: {arg_value}", "blue"))
        # Ngram Assisted Generation
        print("/ngram: toggle ngram assisted generation")
        print(colored(f"\t{self.ngram_gen}", "green" if self.ngram_gen else "red"))
        print("/top_k_filler <value>: set top k filler for ngram update")
        print(colored(f"\t{self.top_k_filler}", "blue"))
        print("/set_ngramstorage <basic/onelevel> <n>: set ngramstorage drafter")
        print(colored(f"\t{self.ngram.__class__.__name__} {self.ngram_n}", "blue"))
        print("/reset_in_between: toggle reset ngram in between each generation")
        print(colored(f"\t{self.reset_in_between}", "green" if self.reset_in_between else "red"))
        

    def _infer(self, prefix: str):
        if self.chat:
            prefix = self.tokenizer.apply_chat_template([{"role": "user", "content": prefix}], add_generation_prompt=True, tokenize=False)
            
        tokenized = self.tokenizer(prefix, return_tensors="pt").input_ids[0].tolist()
        
        if self.reset_in_between:
            self.ngram.reset()
        
        spec_throughput = 0.0
        base_throughput = 0.0
        drafter_throughput = 0.0

        if self.spec:
            self._set_seed(42)
            spec_start_time = time.time()
            output_ids, accept_rate = speculative_generate(
                tokenized,
                self.drafter,
                self.target,
                tokenizer=self.tokenizer,
                logits_processor=self.processor,
                gamma=self.gamma,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                debug=self.debug,
                use_cache=self.cache,
            )
            spec_end_time = time.time()
            spec_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("========== Speculative ==========", "green"))
            print(colored("Out:", "green"), spec_output)
            print(colored(f"Acceptance rate: {accept_rate:.3f}", "green"))
            spec_throughput = len(spec_output) / (spec_end_time - spec_start_time)
            print(colored(f"Throughput: {spec_throughput:.1f} tokens/s", "green"))
            print(colored("========== Speculative ==========", "green"))
            
        if self.ngram_gen:
            self._set_seed(42)
            ngram_start_time = time.time()
            output_ids, accept_rate = ngram_assisted_speculative_generate(
                tokenized,
                self.ngram,
                self.target,
                tokenizer=self.tokenizer,
                gamma=self.gamma,
                filler_top_k=self.top_k_filler,
                logits_processor=self.processor,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                debug=self.debug,
                use_cache=self.cache,
                first_target=True,
                stop_if_unknown=True,
            )
            ngram_end_time = time.time()
            ngram_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("========== Ngram Assisted ==========", "yellow"))
            print(colored("Out:", "yellow"), ngram_output)
            print(colored(f"Acceptance rate: {accept_rate:.3f}", "yellow"))
            ngram_throughput = len(ngram_output) / (ngram_end_time - ngram_start_time)
            print(colored(f"Throughput: {ngram_throughput:.1f} tokens/s", "yellow"))
            print(colored("========== Ngram Assisted ==========", "yellow"))
            if self.spec and ngram_throughput > 0.0:
                print(colored(f"Throughput increase: {((spec_throughput / ngram_throughput)) * 100:.1f}%", "magenta"))

        if self.target_gen:
            self._set_seed(42)
            start_time = time.time()
            output_ids = autoregressive_generate(
                tokenized,
                self.target,
                use_cache=self.cache,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.processor,
                debug=self.debug,
            )
            end_time = time.time()
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("=========== Target AR ===========", "blue"))
            print(colored("Out:", "blue"), output)
            base_throughput = len(output) / (end_time - start_time)
            print(colored(f"Throughput: {base_throughput:.1f} tokens/s", "blue"))
            print(colored("=========== Target AR ===========", "blue"))
            if self.spec and base_throughput > 0.0:
                print(colored(f"Throughput increase: {((spec_throughput / base_throughput)) * 100:.1f}%", "magenta"))

        if self.dr:
            self._set_seed(42)
            output_ids = autoregressive_generate(
                tokenized,
                self.drafter,
                use_cache=self.cache,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.processor,
                debug=self.debug,
            )
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            print(colored("========== Drafter AR ==========", "cyan"))
            drafter_throughput = len(output) / (end_time - start_time)
            print(colored("Out:", "cyan"), output)
            print(colored(f"Throughput: {drafter_throughput:.1f} tokens/s", "cyan"))
            print(colored("========== Drafter AR ==========", "cyan"))

    def _run(self):
        while True:
            command = input("> ").replace('\\n', '\n').replace('\\t', '\t')
            if command.startswith("/"):
                self._perform_command(command)
                continue

            self._infer(command)
            
    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Decoding CLI")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    args = parser.parse_args()

    InferenceCLI(device=args.device)
