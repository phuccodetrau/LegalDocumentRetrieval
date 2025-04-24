from tqdm import tqdm
import numpy as np
import json
from strategy import fuse_bm25_and_embedding, rerank_candidate
from es_query import bm25_search, embedding_search
from load_model import load_biencoder, load_crossencoder

def predict(data_path, bi_path, cross_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bi_model, bi_tokenizer = load_biencoder(bi_path)
    cross_model, cross_tokenizer = load_crossencoder(cross_path)
    mrr10 = []
    for entry in tqdm(data):
        bm25_results = bm25_search(entry["raw_text"])
        embedding_results = embedding_search(entry["raw_text"], bi_model, bi_tokenizer)
        fusion_results = fuse_bm25_and_embedding(bm25_results, embedding_results)
        final_results = rerank_candidate(fusion_results, entry["raw_text"], tokenizer=cross_tokenizer, cross_model=cross_model)
        
        flag = True
        for i, r in enumerate(final_results):
            if r["cid"] in entry["cid"]:
                mrr10.append(1/(i+1))
                flag = False
                break
        if flag:
            mrr10.append(0)
        
        tqdm.write(f"Current mean MRR@10: {np.mean(mrr10):.4f}")
        
if __name__ == "__main__":
    predict("../data/test_set.json", "../training_biencoder/model/model.pt", "../training_crossencoder/model/model.pt")