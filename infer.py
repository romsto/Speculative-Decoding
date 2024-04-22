import argparse
import random
import torch
from sampling import speculative_decoding, autoregressive_decoding
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, QuantoConfig
from constants import DATASET
from models import NgramModel
import time
import os
from termcolor import colored


def main(target_dataset, temperature, top_k, top_p, gamma, ngram_n, ngram_ckpt, device):
    # top_k = 1: greedy decoding
    # top_k = 0: no top-k filtering, multinomial sampling
    # top_p > 0: top-p sampling

    target_name = DATASET[target_dataset]["target"]
    # target_name = "google/gemma-7b"
    print(
        colored(
            f"Speculative Decoding (temperature={temperature}, top_k={top_k}, top_p={top_p}, gamma={gamma})",
            "red",
        )
    )
    print(colored(f"Target: {target_name}\tDrafter: {str(ngram_n) + '-gram'}", "red"))
    print(colored("Loading models...", "yellow"))
    tokenizer = AutoTokenizer.from_pretrained(target_name)
    print(colored("Tokenizer loaded.", "yellow"))
    target = AutoModelForSeq2SeqLM.from_pretrained(target_name).to(device)
    #quantization_config = QuantoConfig(weights="int8")
    #target = AutoModelForCausalLM.from_pretrained(target_name, quantization_config=quantization_config, device_map="cuda")
    target.eval()
    print(colored("Target model loaded.", "yellow"))
    drafter = NgramModel(
        n=ngram_n,
        vocab_size=len(tokenizer),
        device=device,
        resume=ngram_ckpt,
    )
    drafter.to(device)
    # drafter = AutoModelForSeq2SeqLM.from_pretrained("ubikpt/t5-small-finetuned-cnn").to(device)
    #quantization_config_2 = QuantoConfig(weights="int8")
    #drafter = AutoModelForCausalLM.from_pretrained("google/gemma-2b", quantization_config=quantization_config_2, device_map="cuda")
    drafter.eval()
    print(colored("Drafter model loaded.", "yellow"))

    debug = False
    spec = True
    dr = False
    cache = True
    target_gen = True
    gen_len = 35

    # Set random seed of everything
    random.seed(42)
    torch.manual_seed(42)

    while True:
        prefix = input("> ")
        if prefix.startswith("/"):
            args = prefix.split(" ")
            if args[0] == "/quit":
                print(colored("Goodbye!", on_color="on_red"))
                break
            if args[0] == "/debug":
                debug = not debug
                print(colored(f"Debug mode: {debug}", on_color="on_blue"))
                continue
            if args[0] == "/speculative":
                spec = not spec
                print(
                    colored(
                        f"Speculative Decoding generation: {spec}", on_color="on_blue"
                    )
                )
                continue
            if args[0] == "/drafter":
                dr = not dr
                print(colored(f"Drafter generation: {dr}", on_color="on_blue"))
                continue
            if args[0] == "/cache":
                cache = not cache
                print(colored(f"Cache: {cache}", on_color="on_blue"))
                continue
            if args[0] == "/target":
                target_gen = not target_gen
                print(colored(f"Target generation: {target_gen}", on_color="on_blue"))
                continue
            if args[0] == "/temperature":
                if len(args) < 2:
                    print(colored("Usage: /temperature <value>", "red"))
                    continue
                temperature = float(args[1])
                print(colored(f"Temperature: {float(args[1])}", on_color="on_blue"))
                continue
            if args[0] == "/top_k":
                if len(args) < 2:
                    print(colored("Usage: /top_k <value>", "red"))
                    continue
                top_k = int(args[1])
                print(colored(f"Top-k: {int(args[1])}", on_color="on_blue"))
                continue
            if args[0] == "/top_p":
                if len(args) < 2:
                    print(colored("Usage: /top_p <value>", "red"))
                    continue
                top_p = float(args[1])
                print(colored(f"Top-p: {float(args[1])}", on_color="on_blue"))
                continue
            if args[0] == "/length":
                if len(args) < 2:
                    print(colored("Usage: /length <value>", "red"))
                    continue
                gen_len = int(args[1])
                print(colored(f"Generation length: {int(args[1])}", on_color="on_blue"))
                continue
            if args[0] == "/gamma":
                if len(args) < 2:
                    print(colored("Usage: /gamma <value>", "red"))
                    continue
                gamma = int(args[1])
                print(colored(f"Gamma: {int(args[1])}", on_color="on_blue"))
                continue
            if args[0] == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
                continue

            print(colored("Commands:", on_color="on_blue"))
            print("/quit: quit the program")
            print("/help: show this help message")
            print("/debug: toggle speculative debug mode")
            print(colored(f"\t{debug}", "green" if debug else "red"))
            print("/clear: clear the screen")
            print("/speculative: toggle speculative decoding")
            print(colored(f"\t{spec}", "green" if spec else "red"))
            print("/target: toggle target generation")
            print(colored(f"\t{target_gen}", "green" if target_gen else "red"))
            print("/drafter: toggle drafter generation")
            print(colored(f"\t{dr}", "green" if dr else "red"))
            print("/cache: toggle cache")
            print(colored(f"\t{cache}", "green" if cache else "red"))
            print("/temperature <value>: set temperature")
            print(colored(f"\t{temperature}", "blue"))
            print("/top_k <value>: set top-k")
            print(colored(f"\t{top_k}", "blue"))
            print("/top_p <value>: set top-p")
            print(colored(f"\t{top_p}", "blue"))
            print("/length <value>: set generation length")
            print(colored(f"\t{gen_len}", "blue"))
            print("/gamma <value>: set gamma")
            print(colored(f"\t{gamma}", "blue"))
            continue

        input_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
        if spec:
            random.seed(42)
            torch.manual_seed(42)
            spec_start_time = time.time()
            output_ids, accept_rate = speculative_decoding(
                input_ids,
                drafter,
                target,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                debug=debug,
                debug_tokenizer=tokenizer,
                min_len=gen_len,
                gamma=gamma,
                end_token_id=tokenizer.eos_token_id,
                use_cache=cache,
            )
            spec_end_time = time.time()
            spec_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(colored("SpecDec:", "green"), spec_output)
            print(colored("========== Spec ==========", "green"))
            print(colored(f"Accept rate: {accept_rate:.3f}", "green"))
            print(
                colored(
                    f"Execution time: {spec_end_time - spec_start_time:.3f} seconds",
                    "green",
                )
            )
            print(
                colored(
                    f"Throughput: {len(spec_output) / (spec_end_time - spec_start_time):.1f} tokens/s",
                    "green",
                )
            )
            print(colored("========== Spec ==========", "green"))
        if target_gen:
            random.seed(42)
            torch.manual_seed(42)
            start_time = time.time()
            output_ids = autoregressive_decoding(
                input_ids,
                target,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_len=gen_len,
                end_token_id=tokenizer.eos_token_id,
                use_cache=cache,
            )
            end_time = time.time()
            output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(colored("BaseDec:", "blue"), output)
            print(colored("========== Base ==========", "blue"))
            print(
                colored(f"Execution time: {end_time - start_time:.3f} seconds", "blue")
            )
            print(
                colored(
                    f"Throughput: {len(output) / (end_time - start_time):.1f} tokens/s",
                    "blue",
                )
            )
            print(colored("========== Base ==========", "blue"))
            if spec:
                print(
                    colored(
                        f"Speed increase: {((end_time - start_time) / (spec_end_time - spec_start_time)) * 100:.1f}%",
                        "magenta",
                    )
                )
                print(
                    colored(
                        f"Throughput increase: {((len(spec_output) / (spec_end_time - spec_start_time)) / (len(output) / (end_time - start_time))) * 100:.1f}%",
                        "magenta",
                    )
                )

        if dr:
            random.seed(42)
            torch.manual_seed(42)
            output_ids = autoregressive_decoding(
                input_ids,
                drafter,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_len=gen_len,
                end_token_id=tokenizer.eos_token_id,
                use_cache=cache,
            )
            output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(colored("Drafter BaseDec:", "cyan"), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Decoding")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling"
    )
    parser.add_argument("--top_k", type=int, default=1, help="Top-k value for sampling")
    parser.add_argument(
        "--top_p", type=float, default=0.0, help="Top-p value for sampling"
    )
    parser.add_argument(
        "--gamma", type=int, default=4, help="Gamma value for speculative decoding"
    )
    parser.add_argument(
        "--target", type=str, default="cnndm", help="Target model dataset name"
    )
    parser.add_argument(
        "--ngram_n", type=int, default=2, help="N for n-gram drafter model."
    )
    parser.add_argument(
        "--ngram_ckpt",
        type=str,
        default="/workspace/models/nasd/cnndm_target",
        help="Path to n-gram drafter model checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    main(
        args.target,
        args.temperature,
        args.top_k,
        args.top_p,
        args.gamma,
        args.ngram_n,
        args.ngram_ckpt,
        args.device,
    )
