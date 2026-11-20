"""Frozen GPT-2 features and a trainable adapter with one output vector per label token."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, GPT2Model, Trainer, TrainingArguments, set_seed

from app.examples import DATASET, LABELS, label_token

BASE = "gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def build_model(base_model=BASE):
    options = {"revision": REVISION} if base_model == BASE else {}
    tok = AutoTokenizer.from_pretrained(base_model, **options)
    tok.add_special_tokens({"additional_special_tokens": [label_token(label) for label in LABELS]})
    tok.pad_token = tok.eos_token
    encoder = GPT2Model.from_pretrained(base_model, **options)
    encoder.resize_token_embeddings(len(tok), mean_resizing=False)
    # Only the first four transformer blocks supply features for this adapter.
    encoder.h = encoder.h[:4]
    encoder.config.n_layer = len(encoder.h)
    with torch.no_grad():
        for label in LABELS:
            ids = tok.encode(label.replace("_", " "), add_special_tokens=False)
            row = tok.convert_tokens_to_ids(label_token(label))
            encoder.wte.weight[row].copy_(encoder.wte.weight[ids].mean(dim=0))
    return tok, encoder.eval().requires_grad_(False)


@torch.no_grad()
def features(texts, tok, encoder):
    rows = tok(texts, padding=True, truncation=True, max_length=96, return_tensors="pt")
    hidden = encoder(**rows, use_cache=False).last_hidden_state
    mask = rows.attention_mask.unsqueeze(-1)
    return (hidden * mask).sum(1) / mask.sum(1)


class LabelAdapter(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer("center", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        self.down = nn.Linear(width, 32)
        self.up = nn.Linear(32, width, bias=False)
        nn.init.zeros_(self.up.weight)
        self.label_vectors = nn.Parameter(torch.zeros(len(LABELS), width))

    def forward(self, inputs, labels=None):
        x = (inputs - self.center) / self.scale
        x = x + self.up(F.gelu(self.down(x)))
        logits = 12 * F.normalize(x) @ F.normalize(self.label_vectors).T
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels)
        return result


def train(output_dir, base_model=BASE, epochs=15):
    if epochs < 1:
        raise ValueError("Epochs must be positive.")
    set_seed(42)
    tok, encoder = build_model(base_model)
    requests, labels = zip(*DATASET)
    vectors = torch.cat([features(list(requests[i:i+8]), tok, encoder)
                         for i in range(0, len(requests), 8)])
    targets = torch.tensor([LABELS.index(label) for label in labels])
    head = LabelAdapter(vectors.shape[1])
    with torch.no_grad():
        head.center.copy_(vectors.mean(0))
        head.scale.copy_(vectors.std(0).clamp_min(0.05))
        normalized = (vectors - head.center) / head.scale
        for i in range(len(LABELS)):
            head.label_vectors[i].copy_(normalized[targets == i].mean(0))
    rows = [{"inputs": v, "labels": y} for v, y in zip(vectors, targets)]
    args = TrainingArguments(output_dir=str(output_dir), use_cpu=True, num_train_epochs=epochs,
                             per_device_train_batch_size=12, learning_rate=0.002,
                             save_strategy="no", report_to=[], disable_tqdm=True, seed=42)
    trainer = Trainer(model=head, args=args, train_dataset=rows)
    result = trainer.train()
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    save_file(head.cpu().state_dict(), str(folder / "adapter.safetensors"))
    (folder / "model.json").write_text(json.dumps({"base_model": base_model, "labels": LABELS,
        "label_tokens": [label_token(x) for x in LABELS]}, indent=2), encoding="utf-8")
    return result.training_loss


def load(model_dir):
    folder = Path(model_dir)
    config = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    if config["labels"] != LABELS:
        raise ValueError("Saved model labels do not match the application.")
    tok, encoder = build_model(config["base_model"])
    head = LabelAdapter(encoder.config.n_embd)
    head.load_state_dict(load_file(str(folder / "adapter.safetensors")))
    return tok, (encoder, head.eval())


@torch.inference_mode()
def classify(text, tokenizer, model):
    if not text.strip():
        raise ValueError("Enter a non-empty request.")
    encoder, head = model
    scores = head(features([text], tokenizer, encoder))["logits"]
    return LABELS[scores.argmax().item()]

