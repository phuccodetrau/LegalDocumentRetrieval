import torch
import torch.nn as nn
from transformers import AutoModel

class LegalDocumentCrossEncoder(nn.Module):
    def __init__(self, model_name):
        super(LegalDocumentCrossEncoder, self).__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.base_model.config.type_vocab_size = 0
        self.classifier = nn.Sequential(nn.Dropout(p=self.base_model.config.hidden_dropout_prob),
                                        nn.Linear(self.base_model.config.hidden_size, 2))

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=None)

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        
        return logits, outputs[1]