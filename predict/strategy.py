import numpy as np
import torch
from preprocessing.utils import preprocess_question, tokenize_text
from es_query import bm25_search, embedding_search
from load_model import load_biencoder, load_crossencoder

def normalize_scores(results):
    scores = np.array([r["score"] for r in results])
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s == 0:
        return [1.0] * len(results)
    return [(s - min_s) / (max_s - min_s) for s in scores]

def fuse_bm25_and_embedding(bm25_results, embedding_results, top_k=20, alpha=0.5):
    bm25_scores = {r["cid"]: s for r, s in zip(bm25_results, normalize_scores(bm25_results))}
    emb_scores = {r["cid"]: s for r, s in zip(embedding_results, normalize_scores(embedding_results))}

    fused_scores = {}

    for cid in set(bm25_scores.keys()).union(emb_scores.keys()):
        score_bm25 = bm25_scores.get(cid, 0.0)
        score_emb = emb_scores.get(cid, 0.0)
        fused_scores[cid] = alpha * score_bm25 + (1 - alpha) * score_emb

    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    doc_map = {r["cid"]: r["source"] for r in bm25_results + embedding_results}
    return [{"cid": cid, "score": score, "source": doc_map[cid]} for cid, score in sorted_results]

def rerank_candidate(candidate, query, tokenizer, cross_model):
    query = tokenize_text(preprocess_question(query, remove_end_phrase=False))
    contexts = [tokenize_text(c['source']['text']) for c in candidate]
    batch = tokenizer(
        [(query, cand) for cand in contexts],
        max_length=255,
        padding='max_length',
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False
    )
    batch = {k: v.to("cuda") for k, v in batch.items()}
    with torch.no_grad():
        logits, _ = cross_model(**batch)
    scores = torch.softmax(logits, dim=1)[:, 1]
    sorted_pairs = sorted(zip(candidate, scores.tolist()), key=lambda x: x[1], reverse=True)
    candidate = [x[0] for x in sorted_pairs]
    return candidate[0:10]

if __name__ == "__main__":
    query = "Chủ đầu tư không công khai báo cáo đánh giá tác động môi trường đã được phê duyệt kết quả thẩm định báo cáo sẽ bị xử lý như thế nào?"

    bm25_results = bm25_search(query)
    # print(bm25_results)
    bi_model, bi_tokenizer = load_biencoder("../training_biencoder/model/model.pt")
    embedding_results = embedding_search(query, bi_model, bi_tokenizer)
    fusion_results = fuse_bm25_and_embedding(bm25_results, embedding_results)
    cross_model, cross_tokenizer = load_crossencoder("../training_crossencoder/model/model.pt")
    final_results = rerank_candidate(candidate=fusion_results, cross_model=cross_model, query=query, tokenizer=cross_tokenizer)

    for r in final_results:
        print(f"CID: {r['cid']} - Score: {r['score']:.4f}")
        print(f"Text: {r['source']['text'][:200]}...")
        print("-" * 50)