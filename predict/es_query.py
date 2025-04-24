from elasticsearch import Elasticsearch
import torch
from preprocessing.utils import preprocess_question, tokenize_text
import numpy as np
from model import CorpusDocument
es = Elasticsearch(['http://your-instance'])

def create_corpus_index():
    index_name = "corpus"
    index_config = {
        "mappings": {
            "properties": {
                "cid": {"type": "keyword"},
                "text": {"type": "text"},
                "segmented_text": {"type": "text"},
                "embeddings": {
                    "type": "dense_vector",
                    "dims": 768
                }
            }
        }
    }

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=index_config)
        print("✅ Đã tạo index 'corpus'")
    else:
        print("⚠️ Index 'corpus' đã tồn tại")
        
def ingest_corpus_doc(doc: CorpusDocument):
    es.index(index="corpus", id=doc.cid, document=doc.dict())
    
def bm25_search(query: str, top_k=20):
    # segmented_query = tokenize_text(query)
    q = preprocess_question(query, remove_end_phrase=False)

    response = es.search(
        index="corpus",
        size=top_k,
        query={
            "match": {
                "text": {
                    "query": q
                }
            }
        }
    )

    results = [
        {
            "cid": hit["_source"]["cid"],
            "score": hit["_score"],
            "source": hit["_source"]
        }
        for hit in response["hits"]["hits"]
    ]
    return results


def embedding_search(query: str, model, tokenizer, top_k=20):
    query = preprocess_question(query, remove_end_phrase=False)
    segmented_query = tokenize_text(query)

    inputs = tokenizer(
        segmented_query,
        padding="max_length",
        max_length=25,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        query_vector = model(**inputs)[0].squeeze().cpu().numpy()

    response = es.search(
        index="corpus",
        size=top_k,
        query={
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        },
        request_timeout=60 
    )

    results = [
        {
            "cid": hit["_source"]["cid"],
            "score": hit["_score"],
            "source": hit["_source"]
        }
        for hit in response["hits"]["hits"]
    ]
    return results

