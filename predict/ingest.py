from tqdm import tqdm
import json
from preprocessing.utils import tokenize_text
import torch
from model import CorpusDocument
from es_query import ingest_corpus_doc
from load_model import load_biencoder
def ingest_corpus(data_path, model, tokenizer):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    corpus_documents = []

    for entry in data:
        print(entry)
        segmented_text = tokenize_text(data[entry]["text"])
        
        inputs = tokenizer(
            segmented_text,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs)
        
        embeddings = logits[0].tolist()
        corpus_documents.append({
            "cid": data[entry]["cid"],
            "text": data[entry]["text"],
            "segmented_text": segmented_text,
            "embeddings": embeddings
        })
    for doc in tqdm(corpus_documents):
        corpus_doc = CorpusDocument(**doc)
        ingest_corpus_doc(corpus_doc)
        
if __name__ == "__main__":
    model, tokenizer = load_biencoder("../training_biencoder/model/model.pt")
    ingest_corpus("../data/clean_corpus.json", model, tokenizer)
