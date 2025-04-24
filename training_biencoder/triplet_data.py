from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import json


class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_question_len=25, max_ctx_len=400):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_question_len = max_question_len

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        question = triplet["question"]
        pos_context = triplet["positive"]["segmented_text"]
        neg_context = triplet["negative"]["segmented_text"]
        
        anchor_encoding = self.tokenizer(
            question,
            max_length=self.max_question_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False
        )
        pos_encoding = self.tokenizer(
            pos_context,
            max_length=self.max_ctx_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False
        )
        neg_encoding = self.tokenizer(
            neg_context,
            max_length=self.max_ctx_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False
        )

        return {
            'anchor_input_ids': anchor_encoding['input_ids'].squeeze(),
            'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(),
            'pos_input_ids': pos_encoding['input_ids'].squeeze(),
            'pos_attention_mask': pos_encoding['attention_mask'].squeeze(),
            'neg_input_ids': neg_encoding['input_ids'].squeeze(),
            'neg_attention_mask': neg_encoding['attention_mask'].squeeze()
        }