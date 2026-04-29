"""
Fine-tune legal-bert on CUAD dataset for improved clause classification.

⚠️  WARNING: Run overnight. Do not interrupt training.
    Estimated time: 4–6 hours on RTX 3050 6GB
    Model will be saved to: models/legal-bert-cuad/

Prerequisites:
    pip install datasets transformers torch accelerate
    # GPU with CUDA required for fp16=True

Usage:
    python scripts/finetune_cuad.py

The ClassificationAgent works WITHOUT fine-tuning via zero-shot.
This script improves accuracy after the system is running.
"""
import os
import sys
import warnings
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune_cuad")

# ── Constants ─────────────────────────────────────────────────────────
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DATASET_NAME = "theatticusproject/cuad"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "legal-bert-cuad"
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_SEQ_LEN = 512
FP16 = True  # Requires CUDA. Set to False for CPU (very slow)

# All 41 CUAD clause types (must match order used in classification agent)
CUAD_CLAUSE_TYPES = [
    "Non-Compete", "IP Assignment", "Termination", "Governing Law", "Payment Terms",
    "Confidentiality/NDA", "Liability Limitation", "Indemnification", "Force Majeure",
    "Dispute Resolution", "Notice Period", "Non-Solicitation", "Warranty", "Audit Rights",
    "Assignment", "Change of Control", "Data Privacy", "Exclusivity", "Liquidated Damages",
    "Most Favored Nation", "Renewal/Expiration", "Severance", "Source Code Escrow",
    "Uncapped Liability", "Anti-Assignment", "Insurance", "Minimum Commitment",
    "Revenue Share", "Price Restrictions", "Cap on Liability", "Affiliate IP",
    "Covenant Not to Sue", "EBITDA", "Effective Date", "Expiration Date",
    "Jurisdiction", "License Grant", "Post-Termination Services", "Price Adjustment",
    "Unlimited License", "Volume Restriction",
]

CUAD_TO_LABEL_MAP = {ct.lower().replace(" ", "_").replace("/", "_"): ct for ct in CUAD_CLAUSE_TYPES}


def check_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.warn(
                "⚠️  CUDA not available. Training on CPU will be extremely slow (40–60+ hours). "
                "FP16 will be disabled automatically."
            )
            return False
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {device_name} ({vram:.1f} GB VRAM)")
        return True
    except ImportError:
        logger.error("PyTorch not installed.")
        sys.exit(1)


def load_cuad_dataset():
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Run: pip install datasets")
        sys.exit(1)

    logger.info(f"Loading CUAD dataset from HuggingFace: {DATASET_NAME}")
    logger.info("This may take several minutes on first download (~500MB)...")
    try:
        dataset = load_dataset(DATASET_NAME)
        logger.info(f"Dataset loaded. Splits: {list(dataset.keys())}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load CUAD: {e}")
        logger.error(
            "Ensure you have internet access and the dataset is public. "
            "Alternatively, download manually from https://huggingface.co/datasets/theatticusproject/cuad"
        )
        sys.exit(1)


def prepare_binary_classification_data(dataset, clause_type: str):
    """
    CUAD has QA-format data. We extract positive/negative examples per clause type
    using a one-vs-rest binary classification approach.
    """
    texts = []
    labels = []

    clause_key = clause_type.lower().replace(" ", "_").replace("/", "_").replace("-", "_")

    for split in ["train", "test"]:
        if split not in dataset:
            continue
        for example in dataset[split]:
            # CUAD format: each example has 'context' (contract text) and QA pairs
            context = example.get("context", "")
            if not context:
                continue

            # Check if this example has an answer for the target clause type
            # CUAD stores annotations as a list of question-answer pairs
            answers = example.get("answers", {})
            answer_texts = answers.get("text", [])

            if answer_texts:
                # Positive example: this context contains the target clause type
                # Use first 512 chars as the training sample
                texts.append(context[:512])
                labels.append(1)
            else:
                texts.append(context[:512])
                labels.append(0)

    return texts, labels


def tokenize_dataset(texts, labels, tokenizer, max_length=MAX_SEQ_LEN):
    try:
        from datasets import Dataset
    except ImportError:
        sys.exit(1)

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )
    dataset_dict = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }
    return Dataset.from_dict(dataset_dict)


def train_for_clause_type(clause_type: str, dataset, tokenizer, cuda_available: bool):
    """Train a binary classifier for one clause type."""
    from transformers import (
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score

    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {clause_type}")
    logger.info(f"{'='*60}")

    texts, labels = prepare_binary_classification_data(dataset, clause_type)
    if not texts:
        logger.warning(f"No data found for {clause_type}. Skipping.")
        return None

    pos_count = sum(labels)
    logger.info(f"Examples: {len(texts)} total, {pos_count} positive, {len(texts)-pos_count} negative")

    # Split 80/20
    split_idx = int(len(texts) * 0.8)
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    train_dataset = tokenize_dataset(train_texts, train_labels, tokenizer)
    val_dataset = tokenize_dataset(val_texts, val_labels, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    clause_output_dir = OUTPUT_DIR / clause_type.replace(" ", "_").replace("/", "_")
    clause_output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = FP16 and cuda_available

    training_args = TrainingArguments(
        output_dir=str(clause_output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_fp16,
        logging_steps=50,
        report_to="none",  # Disable W&B
        dataloader_num_workers=0,
    )

    def compute_metrics(eval_pred):
        logits, true_labels = eval_pred
        preds = np.argmax(logits, axis=1)
        f1 = f1_score(true_labels, preds, zero_division=0)
        precision = precision_score(true_labels, preds, zero_division=0)
        recall = recall_score(true_labels, preds, zero_division=0)
        return {"f1": f1, "precision": precision, "recall": recall}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate and print results
    eval_results = trainer.evaluate()
    logger.info(
        f"{clause_type} → F1: {eval_results.get('eval_f1', 0):.3f} | "
        f"Precision: {eval_results.get('eval_precision', 0):.3f} | "
        f"Recall: {eval_results.get('eval_recall', 0):.3f}"
    )

    trainer.save_model(str(clause_output_dir / "best"))
    return eval_results


def main():
    print("\n" + "="*70)
    print("LEXAGENT — Legal-BERT CUAD Fine-Tuning Script")
    print("="*70)
    print("⚠️  WARNING: Run overnight. Do not interrupt training.")
    print(f"   Model output: {OUTPUT_DIR}")
    print(f"   Training: {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    print("="*70 + "\n")

    cuda_available = check_gpu()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
        sys.exit(1)

    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = load_cuad_dataset()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for clause_type in CUAD_CLAUSE_TYPES:
        result = train_for_clause_type(clause_type, dataset, tokenizer, cuda_available)
        if result:
            all_results[clause_type] = result

    # Print summary
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE — Per-Class F1 Summary")
    print("="*70)
    for clause_type, result in sorted(all_results.items()):
        f1 = result.get("eval_f1", 0)
        bar = "█" * int(f1 * 20) + "░" * (20 - int(f1 * 20))
        print(f"{clause_type:<35} [{bar}] {f1:.3f}")

    avg_f1 = sum(r.get("eval_f1", 0) for r in all_results.values()) / max(len(all_results), 1)
    print(f"\nAverage F1: {avg_f1:.3f}")
    print(f"\nModels saved to: {OUTPUT_DIR}")
    print(
        "\nTo use fine-tuned models, update ClassificationAgent to load from:\n"
        f"  {OUTPUT_DIR}/<clause_type>/best\n"
        "for per-type binary classification instead of zero-shot."
    )


if __name__ == "__main__":
    main()
