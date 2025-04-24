import torch
import torch.nn as nn
from transformers import AutoModel

class LegalDocumentBiEncoder(nn.Module):
    def __init__(self, model_name):
        super(LegalDocumentBiEncoder, self).__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.base_model.config.type_vocab_size = 0

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )
        last_hidden_state = outputs[0]
        cls_embedding = last_hidden_state[:, 0, :]
        return cls_embedding