# app.py

import streamlit as st
import json
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read HF token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Show token loaded (for debugging only — delete later)
# st.write("Token starts with:", HUGGINGFACE_TOKEN[:8])

# -----------------------------
# Load Chunks
# -----------------------------
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -----------------------------
# Load embeddings
# -----------------------------
embeddings = np.load("embeddings.npy")
embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# -----------------------------
# Create FAISS index
# -----------------------------
d = embeddings_norm.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings_norm)

# -----------------------------
# Load embedding model
# -----------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# -----------------------------
# Load HF Chat Model
# -----------------------------
client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=HUGGINGFACE_TOKEN
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="RAG Demo — ML Knowledge Base",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAG Demo — ML Knowledge Base")
st.caption(
    "A lightweight Retrieval-Augmented Generation (RAG) demo that answers questions "
    "from a curated ML knowledge base with citations."
)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown(
        """
        **What this project does (short and complete):**
        - Builds a compact ML knowledge base from curated articles.
        - Retrieves the most relevant text chunks using FAISS vector search.
        - Uses an open-source LLM to generate a concise answer with citations.
        - Shows the exact context used so you can verify the response.

        **Built for ML-only questions:**
        - Trained on ML-related content only, so ask ML topics (models, metrics, training).
        - For other domains, it may respond with "I don't know".

        **How to use:**
        - Type a natural-language question in the box below.
        - Review the cited answer and the retrieved chunks.
        """
    )

with col_right:
    st.markdown(
        """
        **Tech stack:**
        - SentenceTransformer embeddings (all-MiniLM-L6-v2)
        - FAISS (cosine similarity)
        - Llama-3.2-3B-Instruct via Hugging Face API

        **Notes:**
        - If the answer is not in context, it says "I don't know".
        - Requires internet access for LLM inference.
        """
    )

st.info(
    "This is a mini project for an AI/ML practical assessment and demonstrates an "
    "end-to-end RAG pipeline: preprocess -> embed -> retrieve -> generate."
)

query = st.text_input("Ask an ML question (models, metrics, training, or techniques):")


# -----------------------------
# 1️⃣ Retrieve Relevant Chunks
# -----------------------------
def retrieve_chunks(query, k=5):
    q_emb = embedder.encode([query])
    q_emb = q_emb / np.linalg.norm(q_emb)
    scores, ids = index.search(q_emb.astype(np.float32), k)

    results = []
    for i, idx in enumerate(ids[0]):
        results.append({
            "chunk_id": int(idx),
            "score": float(scores[0][i]),
            "text": chunks[idx]["text"]
        })

    return results


# -----------------------------
# 2️⃣ Build Prompt for LLM
# -----------------------------
def build_prompt(query, retrieved):
    ctx = ""
    for i, c in enumerate(retrieved):
        ctx += f"[{i}] {c['text']}\n\n"

    return f"""You are an AI assistant specialized in Machine Learning. Use the context below to answer the question.

Context:
{ctx}

Question: {query}

Instructions:
- Answer based on the provided context.
- If you use specific chunks, cite them like [0], [1], etc.
- If the context doesn't fully answer the question, provide the best answer from what's available and mention what information is missing.
- Keep your answer concise and clear.

Answer:"""


# -----------------------------
# 3️⃣ Final RAG Answer
# -----------------------------
if query:
    retrieved = retrieve_chunks(query)
    prompt = build_prompt(query, retrieved)

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250
    )

    st.write("### 📘 Response:")
    st.write(response.choices[0].message["content"])

    st.write("---")
    st.write("### 📄 Retrieved Chunks (Context Used):")
    st.json(retrieved)

