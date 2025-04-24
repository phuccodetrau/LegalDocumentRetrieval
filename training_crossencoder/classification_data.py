from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import json
import torch


class ClassificationDataset(Dataset):
    def __init__(self, data_points, tokenizer, max_length=255):
        self.data_points = data_points
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.model_max_length

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        entry = self.data_points[idx]
        question = entry["question"]
        context = entry["context"]
        target = entry["label"]
        
        entry_encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False
        )
        entry_input = {
            'input_ids': entry_encoding['input_ids'].squeeze(),
            'attention_mask': entry_encoding['attention_mask'].squeeze(),
        }
        return entry_input, torch.tensor(int(target))