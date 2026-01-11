import pickle
import faiss
import numpy as np
import re
import requests
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

#loading the vector db
with open("db/chunks.pkl", "rb") as f:
    CHUNKS = pickle.load(f)

with open("db/meta.pkl", "rb") as f:
    META = pickle.load(f)

INDEX = faiss.read_index("db/index.faiss")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

#building Bm25 index
def tokenize(text):
    return re.findall(r"\w+", text.lower())

BM25_CORPUS = [tokenize(c) for c in CHUNKS]
BM25 = BM25Okapi(BM25_CORPUS)

#using reciprocal ranked fusion 
def rrf(faiss_ids, bm25_ids, k=60):
    scores = {}

    for rank, i in enumerate(faiss_ids):
        scores[i] = scores.get(i, 0) + 1 / (k + rank)

    for rank, i in enumerate(bm25_ids):
        scores[i] = scores.get(i, 0) + 1 / (k + rank)

    return sorted(scores, key=scores.get, reverse=True)

#using hybrid retrieval strategies 
def retrieve(query, top_k=6):
    # FAISS
    q_emb = MODEL.encode([query]).astype("float32")
    _, faiss_ids = INDEX.search(q_emb, 20)
    faiss_ids = faiss_ids[0].tolist()

    # BM25
    tokens = tokenize(query)
    bm25_scores = BM25.get_scores(tokens)
    bm25_ids = np.argsort(bm25_scores)[::-1][:20].tolist()

    
    fused = rrf(faiss_ids, bm25_ids)

    return fused[:top_k]

#using ollama chat 
def ask_llm(prompt):
    try:
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3",
                "messages": [
                    {"role": "system", "content": "You are a financial document assistant. Answer ONLY using the provided context."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            },
            timeout=120
        )
        return r.json()["message"]["content"]
    except Exception as e:
        return "LLM error: " + str(e)

#main rag 
def ask_rag(question):
    ids = retrieve(question)

    context = ""
    citations = set()

    for i in ids:
        context += CHUNKS[i] + "\n\n"
        meta = META[i]
        citations.add(f"Page {meta['page']} ({meta['type']})")

    prompt = f"""
Answer ONLY from this context.
If the answer is not in context, say you don't know.

Context:
{context}

Question: {question}

Give numbers exactly and do not guess.
"""

    answer = ask_llm(prompt)

    return answer.strip(), ", ".join(sorted(citations))