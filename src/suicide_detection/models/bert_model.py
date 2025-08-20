from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class BertConfig:
    model_name: str = "mental/mental-bert-base-uncased"
    num_labels: int = 2


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(
            str(self.texts[idx]),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in item.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def build_model_and_tokenizer(cfg: BertConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels
    )
    return model, tokenizer
