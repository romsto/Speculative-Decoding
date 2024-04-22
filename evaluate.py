import logging
from datetime import datetime
import argparse
import torch
import random
import torchmetrics.text
from tqdm.rich import tqdm
from datasets import load_dataset
from sampling import autoregressive_decoding, speculative_decoding
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from constants import DATASET
from models import NgramModel
from functools import wraps, partial
from torch.cuda import Event


def benchmark(fn, timer):
    @wraps(fn)
    def inner(*args, **kwargs):
        start_event = timer()
        end_event = timer()
        start_event.record()

        out = fn(*args, **kwargs)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        return out, elapsed_time_ms

    return inner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Decoding Evaluator")
    parser.add_argument(
        "--dataset", type=str, default="cnndm", help="Dataset to use for evaluation."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling"
    )
    parser.add_argument("--top_k", type=int, default=1, help="Top-k value for sampling")
    parser.add_argument(
        "--top_p", type=float, default=0.0, help="Top-p value for sampling"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=200,
        help="Maximum length of the generated sequence",
    )
    parser.add_argument(
        "--gamma", type=int, default=4, help="Gamma value for speculative decoding"
    )
    parser.add_argument(
        "--ngram_n",
        type=int,
        default=2,
        help="N for n-gram drafter model.",
    )
    parser.add_argument(
        "--ngram_ckpt",
        type=str,
        default="/workspace/models/nasd/cnndm_target",
        help="Path to n-gram drafter model checkpoint.",
    )
    args = parser.parse_args()

    # ================== Set up logging ==================
    ngram_type = args.ngram_ckpt.split("/")[-1]
    experiment_name = f"{args.dataset}_{args.ngram_n}_{ngram_type}_{args.temperature}_{args.top_k}_{args.top_p}_{args.max_len}_{args.gamma}"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    handler = logging.FileHandler(
        f"{experiment_name}.{current_time}.log",
    )
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    # ====================================================

    logger.info(f"Launching experiment {experiment_name}")
    logger.info(f"Sampling parameters:")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Top-k: {args.top_k}")
    logger.info(f"Top-p: {args.top_p}")
    logger.info(f"Maximum length: {args.max_len}")
    logger.info(f"Speculative decoding parameters:")
    logger.info(f"Gamma: {args.gamma}")
    logger.info(
        f"Drafter: {args.ngram_n}gram ({ngram_type})"
    )

    # ================== Load dataset ==================
    if args.dataset not in DATASET:
        raise ValueError(f"Dataset {args.dataset} not found in DATASET constant.")

    dataset_name = DATASET[args.dataset]["name"]
    dataset_version = DATASET[args.dataset]["version"]

    dataset = load_dataset(dataset_name, dataset_version, split="test")
    logger.info("Dataset loaded.")
    # =================================================

    # ================== Load tokenizer ==================
    target_name = DATASET[args.dataset]["target"]
    tokenizer = AutoTokenizer.from_pretrained(target_name)
    logger.info("Tokenizer loaded.")
    # ==================================================

    # ================== Load target model ==================
    target_model = AutoModelForSeq2SeqLM.from_pretrained(target_name)
    target_model.to("cuda")
    logger.info("Target model loaded.")
    # =====================================================

    # ================== Load drafter model ==================
    drafter = NgramModel(
        n=args.ngram_n,
        vocab_size=tokenizer.vocab_size,
        resume=args.ngram_ckpt,
        device="cuda",
    )
    drafter.to("cuda")
    logger.info(f"Loaded n-gram drafter model with n={args.ngram_n}")
    logger.info(f"Ngram type: {ngram_type}")
    # =======================================================

    # ================== Set up random seeds ==================
    random.seed(42)
    torch.manual_seed(42)
    # =======================================================

    # ================== Set up models ==================
    if not drafter is None:
        drafter.eval()
    target_model.eval()
    # =================================================

    # ================== Set up metrics ==================
    timer = partial(Event, enable_timing=True)
    target_time = []
    target_throughput = []
    spec_time = []
    spec_throughput = []
    spec_alphas = []

    count_times_spec_longer = 0

    target_metric = torchmetrics.text.ROUGEScore(accumulate="avg")
    spec_metric = torchmetrics.text.ROUGEScore(accumulate="avg")
    # ==================================================

    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    max_len = args.max_len
    gamma = args.gamma

    dataset_length = len(dataset)
    logger.info(f"Dataset length: {dataset_length}")

    # ================== Inference loop ==================
    logger.info("Starting inference loop...")
    input_finder = DATASET[args.dataset]["input"]
    label_finder = DATASET[args.dataset]["label"]
    for i in tqdm(
        range(dataset_length),
        desc="Inference",
        total=dataset_length,
        mininterval=30,
        maxinterval=50,
    ):
        sample = dataset[i]
        input_text = input_finder(sample)
        target_text = label_finder(sample)

        input_tokens = tokenizer.encode(
            input_text, return_tensors="pt", max_length=512
        ).to("cuda")

        target_out, target_elapsed = benchmark(autoregressive_decoding, timer)(
            input_tokens,
            target_model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_len=max_len,
            use_cache=True,
        )

        (spec_out, spec_alpha), spec_elapsed = benchmark(speculative_decoding, timer)(
            input_tokens,
            drafter,
            target_model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_len=max_len,
            use_cache=True,
        )

        target_time.append(target_elapsed)
        spec_time.append(spec_elapsed)
        spec_alphas.append(spec_alpha)

        if spec_elapsed > target_elapsed:
            count_times_spec_longer += 1

        target_decode = tokenizer.decode(target_out[0], skip_special_tokens=True)
        spec_decode = tokenizer.decode(spec_out[0], skip_special_tokens=True)

        target_thr = len(target_decode) / (target_elapsed / 1000)
        spec_thr = len(spec_decode) / (spec_elapsed / 1000)

        target_throughput.append(target_thr)
        spec_throughput.append(spec_thr)

        target_metric.update(target_decode, target_text)
        spec_metric.update(spec_decode, target_text)

        logger.debug(f"Input text: {input_text}")
        logger.debug(
            f"Target decoding ({target_elapsed:.1f}ms / {target_thr:.1f} t.s-1): {target_decode}"
        )
        logger.debug(
            f"Speculative decoding ({spec_elapsed:.1f}ms / {spec_thr:.1f} t.s-1) @ {spec_alpha:.2f}: {spec_decode}\n"
        )
    # ===================================================

    # ================== Finalize metrics ==================
    target_metric.compute()
    spec_metric.compute()

    average_target_time = sum(target_time) / len(target_time)
    average_target_throughput = sum(target_throughput) / len(target_throughput)
    average_spec_time = sum(spec_time) / len(spec_time)
    average_spec_throughput = sum(spec_throughput) / len(spec_throughput)
    average_spec_alpha = sum(spec_alphas) / len(spec_alphas)
    # =====================================================

    logger.info("Experiment finished.")

    # ================== Log data ==================
    logger.info("Experiment results:")
    logger.info("Target throughput: ")
    logger.info(target_throughput)
    logger.info("Speculative throughput: ")
    logger.info(spec_throughput)
    logger.info("Target times: ")
    logger.info(target_time)
    logger.info("Speculative times: ")
    logger.info(spec_time)
    logger.info("Speculative alphas: ")
    logger.info(spec_alphas)

    # ================== Log metrics ==================
    for metric in ["rouge2_fmeasure", "rouge2_precision", "rouge2_recall"]:
        logger.info(f"{metric}\tTarget:\t{target_metric.compute()[metric]:.3f}")
        logger.info(f"{metric}\tSpec:\t{spec_metric.compute()[metric]:.3f}")
    logger.info(f"Average target decoding time: {average_target_time:.3f} ms")
    logger.info(f"Average speculative decoding time: {average_spec_time:.3f} ms")
    logger.info(f"Average acceptance rate: {average_spec_alpha:.3f}")
    logger.info(
        f"Average target decoding throughput: {average_target_throughput:.3f} tokens/s"
    )
    logger.info(
        f"Average speculative decoding throughput: {average_spec_throughput:.3f} tokens/s"
    )
    logger.info(f"Speculative decoding was longer {count_times_spec_longer} times.")
    # =====================================================
