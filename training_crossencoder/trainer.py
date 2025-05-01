import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer
from modeling_phobert import LegalDocumentCrossEncoder
from classification_data import ClassificationDataset
import json
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
class Trainer:
    def __init__(self, model_name, inputfile, train_ratio=0.7, val_ratio=0.2, max_length=255, train_type="triplet", 
                 num_epochs=3, lr=0.00005, weight_decay=0.1, batch_size=32, eval_freq=5, eval_iter=5, save_path="/content/drive/MyDrive/BKAI_Legal_dataset/model", device="cuda"):
        self.model = LegalDocumentCrossEncoder(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.save_path = save_path

        with open(inputfile, "r", encoding="utf-8") as f:
            data = json.load(f)

        train_dataset = data[:int(train_ratio * len(data))]
        val_dataset = data[int(train_ratio * len(data)):int((train_ratio + val_ratio) * len(data))]
        test_dataset = data[int((train_ratio + val_ratio) * len(data)):]

        self.train_type = train_type
        self.device = device
        torch.manual_seed(123)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.num_epochs = num_epochs
        self.eval_freq = eval_freq
        self.eval_iter = eval_iter
        self.batch_size = batch_size

        self.train_loader = DataLoader(ClassificationDataset(train_dataset, self.tokenizer, max_length=max_length),
                                       batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0)
        self.val_loader = DataLoader(ClassificationDataset(val_dataset, self.tokenizer, max_length=max_length),
                                     batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0)
        self.test_loader = DataLoader(ClassificationDataset(test_dataset, self.tokenizer, max_length=max_length),
                                      batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0)
        self.total_steps = len(self.train_loader) * self.num_epochs
        self.warmup_steps = int(0.1 * self.total_steps)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        self.loss_fn = F.CrossEntropyLoss()
        
    def calc_loss_batch(self, inputs, targets):
        self.model.train()
        logits, _ = self.model(inputs["input_ids"].to(self.device), inputs["attention_mask"].to(self.device))
        CE_loss = self.loss_fn(logits, targets.to(self.device))
        return CE_loss

    def calc_loss_loader(self, data_loader, num_batches=None):
        total_loss = 0
        if len(data_loader) == 0:
            return float("nan")
        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (inputs, targets) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(inputs, targets)
                total_loss += loss.item()
            else:
                break

        return total_loss / num_batches

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            train_loss = self.calc_loss_loader(self.train_loader, num_batches=self.eval_iter)
            val_loss = self.calc_loss_loader(self.val_loader, num_batches=self.eval_iter)
        self.model.train()
        return train_loss, val_loss

    def train(self):
        train_losses, val_losses = [], []
        global_step = 0
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.model.train()
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.calc_loss_batch(inputs, targets)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                global_step += 1

                if global_step % self.eval_freq == 0:
                    train_loss, val_loss = self.evaluate()
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                    
        self.save_model()

    def save_model(self):
        os.makedirs(self.save_path, exist_ok=True)
        model_path = os.path.join(self.save_path, "model.pt")
        tokenizer_path = os.path.join(self.save_path, "tokenizer")

        torch.save(self.model.state_dict(), model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        print(f"Model saved to {model_path}")
        print(f"Tokenizer saved to {tokenizer_path}")