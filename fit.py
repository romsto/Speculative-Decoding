import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from constants import DATASET
from models import NgramModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ngram Trainer")
    parser.add_argument(
        "--dataset", type=str, default="cnndm", help="Dataset to use for training."
    )
    parser.add_argument(
        "--ngram_n", type=int, default=2, help="N for n-gram drafter model."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="target",
        help="Tokenizer to use for the model. ('target' to use the one from the target model of the dataset)",
    )
    parser.add_argument(
        "--ngram_output",
        type=str,
        default="/workspace/models/nasd/cnndm_target",
        help="Path to output n-gram drafter model checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training. ('cuda' or 'cpu')",
    )
    args = parser.parse_args()

    print(f"Launching training of {args.ngram_n}-gram model")

    # ================== Load dataset ==================
    if args.dataset not in DATASET:
        raise ValueError(f"Dataset {args.dataset} not found in DATASET constant.")

    dataset_name = DATASET[args.dataset]["name"]
    dataset_version = DATASET[args.dataset]["version"]

    print(f"Loading {dataset_name} ({dataset_version}) dataset...")
    dataset = load_dataset(dataset_name, dataset_version, split="train")
    print("Dataset loaded.")
    # =================================================

    # ================== Load tokenizer ==================
    if args.tokenizer == "target":
        args.tokenizer = DATASET[args.dataset]["target"]

    print(f"Loading {args.tokenizer} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print("Tokenizer loaded.")
    # ==================================================

    # ================== Load drafter model ==================
    drafter_model = NgramModel(
        args.ngram_n,
        tokenizer.vocab_size,
        resume=args.ngram_output,
        device=args.device,
        fit_mode=True,
    )
    drafter_model.to(args.device)
    drafter_model.train()
    # =======================================================

    dataset_length = len(dataset)
    print(f"Dataset length: {dataset_length}")

    # ================== Training loop ==================
    label_loader = DATASET[args.dataset]["label"]
    drafter_model.fit(dataset, tokenizer, label_loader, path=args.ngram_output)
    # ===================================================

    print("Fitting finished.")
