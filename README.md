
# RAG-based Q&A Bot (MVP)

This is a simple RAG (Retrieval-Augmented Generation) project that answers
questions based on a custom knowledge base created from text articles.

---

## 🚀 Features
- Text cleaning and chunking
- Embedding generation using Sentence Transformers
- FAISS vector search
- RAG pipeline (retrieve → prompt → answer)
- Llama-3.2 (3B) model via HuggingFace API
- Streamlit web interface

---

## 🧠 How It Works
1. Articles are loaded and converted into clean text.
2. Text is split into small chunks.
3. Each chunk is converted into an embedding vector.
4. FAISS is used to store and search similar chunks.
5. For each query:
   - Relevant chunks are retrieved
   - A prompt is created
   - LLM generates the final answer

---


