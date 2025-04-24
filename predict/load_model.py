from transformers import AutoTokenizer
import torch
from training_biencoder.modeling_phobert import LegalDocumentBiEncoder
from training_crossencoder.modeling_phobert import LegalDocumentCrossEncoder
def load_biencoder(model_path):
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model_name = "vinai/phobert-base"
    model = LegalDocumentBiEncoder(model_name)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer

def load_crossencoder(model_path):
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model_name = "vinai/phobert-base"
    model = LegalDocumentCrossEncoder(model_name)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer