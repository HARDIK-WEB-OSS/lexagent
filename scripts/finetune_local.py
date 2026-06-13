import json, logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune_local")

DATASET_PATH  = Path("evaluation/test_dataset.json")
OUTPUT_DIR    = Path("models/legal-bert-local")
MODEL_NAME    = "nlpaueb/legal-bert-base-uncased"
NUM_EPOCHS    = 80
BATCH_SIZE    = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
MAX_SEQ_LEN   = 256

def main():
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    with open(DATASET_PATH) as f:
        data = json.load(f)

    labels   = sorted(set(d["true_label"] for d in data))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    logger.info(f"Classes: {len(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # KEY FIX: build config from scratch with correct num_labels
    # This prevents the base model's id2label(len=2) from overriding ours
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = len(labels)
    config.id2label   = id2label
    config.label2id   = label2id

    class ClauseDataset(Dataset):
        def __init__(self, data):
            self.encodings = tokenizer(
                [d["clause_text"] for d in data],
                truncation=True, padding=True,
                max_length=MAX_SEQ_LEN, return_tensors="pt"
            )
            self.labels = torch.tensor([label2id[d["true_label"]] for d in data])
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.encodings.items()}, self.labels[i]

    dataset = ClauseDataset(data)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,                  # pass our fixed config explicitly
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Phase 1: train classifier head only (frozen BERT)
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    def make_optimizer(lr):
        return AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=WEIGHT_DECAY
        )

    optimizer = make_optimizer(LEARNING_RATE)
    total_steps = len(loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    best_acc, best_epoch, best_state = 0.0, 0, None

    for epoch in range(NUM_EPOCHS):
        # Phase 2 at epoch 30: unfreeze full model with lower LR
        if epoch == 30:
            logger.info("Unfreezing full model...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = make_optimizer(LEARNING_RATE / 10)

        model.train()
        total_loss, correct = 0, 0

        for batch_inputs, batch_labels in loader:
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            batch_labels = batch_labels.to(device)
            outputs      = model(**batch_inputs, labels=batch_labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += outputs.loss.item()
            correct    += (outputs.logits.argmax(-1) == batch_labels).sum().item()

        acc = correct / len(dataset)
        if acc > best_acc:
            best_acc   = acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:2d}/{NUM_EPOCHS}  "
                        f"loss={total_loss/len(loader):.4f}  "
                        f"acc={acc:.1%}  best={best_acc:.1%}@ep{best_epoch}")

    logger.info(f"Restoring best checkpoint: epoch {best_epoch}, acc={best_acc:.1%}")
    model.load_state_dict(best_state)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Done. Model saved → {OUTPUT_DIR}  best_acc={best_acc:.1%}")

if __name__ == "__main__":
    main()
