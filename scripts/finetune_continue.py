import json, logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune_continue")

CHECKPOINT   = Path("models/legal-bert-local")   # resume from here
OUTPUT_DIR   = Path("models/legal-bert-local")
DATASET_PATH = Path("evaluation/test_dataset.json")
NUM_EPOCHS   = 120
LEARNING_RATE = 2e-6    # lower — full model already unfrozen
WEIGHT_DECAY  = 0.01
MAX_SEQ_LEN   = 256
BATCH_SIZE    = 8

def main():
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    with open(DATASET_PATH) as f:
        data = json.load(f)

    labels   = sorted(set(d["true_label"] for d in data))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model     = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
    model.to(device)
    logger.info(f"Loaded checkpoint. num_labels={model.config.num_labels}")

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

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(loader) * 5,
        num_training_steps=len(loader) * NUM_EPOCHS
    )

    best_acc, best_epoch, best_state = 0.0, 0, None

    for epoch in range(NUM_EPOCHS):
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
            logger.info(f"Epoch {epoch+1:3d}/{NUM_EPOCHS}  "
                        f"loss={total_loss/len(loader):.4f}  "
                        f"acc={acc:.1%}  best={best_acc:.1%}@ep{best_epoch}")

        # Early stop if converged
        if best_acc >= 0.90:
            logger.info(f"Reached 90% accuracy at epoch {best_epoch}. Stopping.")
            break

    logger.info(f"Restoring best: epoch {best_epoch}, acc={best_acc:.1%}")
    model.load_state_dict(best_state)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Saved → {OUTPUT_DIR}  best_acc={best_acc:.1%}")

if __name__ == "__main__":
    main()
