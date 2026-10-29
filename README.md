# 🤖 Enterprise Hybrid RAG Chatbot & Autonomous Agent

A Retrieval-Augmented Generation (RAG) chatbot built with Python and Streamlit. It combines a **hybrid retrieval engine** (FAISS + BM25), **cross-encoder re-ranking**, and an **automatic web-search fallback** so answers come from your documents whenever possible and from the live web only when they don't.

## ✨ Key Features

* **Hybrid Retrieval:** Dense vector search (FAISS) finds semantically similar passages, and sparse keyword search (BM25) finds exact term matches. Results are merged with Reciprocal Rank Fusion.
* **Cross-Encoder Re-ranking:** An MS-MARCO cross-encoder scores each candidate against the question, so only the most relevant chunks reach the LLM.
* **Strict Document Answers:** The model is instructed to answer only from the retrieved context and to return a fixed `NO_ANSWER_FOUND` token when the context doesn't contain the answer.
* **Web Search Fallback (DuckDuckGo):** When the documents don't contain the answer, the question is rewritten into a standalone query using recent chat history, searched on the web, and answered from the results. The UI labels these answers clearly.
* **Dynamic Ingestion:** Upload a PDF in the sidebar to chunk, embed, and index it into the existing databases immediately.
* **Conversation Context:** The last few messages are passed to the model, so follow-up questions with words like "it" or "they" resolve correctly.
* **Source Citations:** Local answers list the PDF pages they came from. Web answers link to their sources.
* **Explicit Error States:** Failed LLM calls show a red error box instead of being presented as answers.

---

## 🛠️ Technology Stack

* **Frontend:** Streamlit
* **LLM:** Groq (default model `openai/gpt-oss-120b`, configurable in `.env`)
* **Orchestration:** LangChain (`langchain-core`, `langchain-groq`, `langchain-community`)
* **Dense Index:** FAISS (`faiss-cpu`)
* **Sparse Index:** BM25 (`rank-bm25`)
* **Embeddings & Re-ranking:** `sentence-transformers` (`all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`)
* **Web Search:** DuckDuckGo (`ddgs`)
* **Document Processing:** `pypdf`, `pymupdf`, `tiktoken`

---

## 🏗️ How It Works

1. **Ingestion:** An uploaded PDF is read page by page, split into overlapping chunks, and embedded with `all-MiniLM-L6-v2`.
2. **Indexing:** Embeddings go into a FAISS index, and the chunk text goes into a BM25 index. Both are saved to `vector_store/`.
3. **Hybrid Retrieval:** The question is searched in both indexes, and the results are fused with Reciprocal Rank Fusion. The top 6 candidates are kept.
4. **Re-ranking:** The cross-encoder scores each candidate against the question, and the top 3 go to the LLM.
5. **Generation:** The LLM answers from those chunks. If it returns `NO_ANSWER_FOUND`, the pipeline rewrites the question, searches DuckDuckGo, and generates an answer from the web results.
6. **Display:** The answer appears in the chat with its sources and a label showing whether it came from your PDFs or the web.

---

## 🚀 Setup

### Step 1: Prerequisites

1. **Python 3.9+**
2. **A Groq API key** from [console.groq.com/keys](https://console.groq.com/keys)

### Step 2: Create a Virtual Environment

```bash
git clone https://github.com/Shanavas15/hybrid-rag-chatbot.git
cd hybrid-rag-chatbot
python -m venv venv

# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Your API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

> ⚠️ Never commit `.env`. It's listed in `.gitignore`. If a key is ever exposed, delete it at console.groq.com and create a new one.

To see which models your key can use, check the model list at [console.groq.com/docs/models](https://console.groq.com/docs/models) or query the API with your key.

---

## 💻 Usage

```bash
streamlit run app.py
```

1. **Upload:** Drag a PDF into the sidebar and click **Process Documents**. The index updates immediately.
2. **Ask:** Type a question in the chat box. Answers from your documents show the source pages. Answers from the web show a notice and a fact-checking panel.
3. **Clear:** Use **Clear Chat** in the sidebar to reset the conversation. Your indexed documents are kept.

### Changing Settings

* **Model:** set `GROQ_MODEL` in `.env`.
* **Temperature:** set in `RAGGenerator.__init__` in `rag/generator.py` (default `0.2`, which suits factual answers).
* **Number of chunks:** set `top_k` in `EnterpriseRAGPipeline.ask()` in `rag/pipeline.py` (default `3`).

---

## 📁 Project Structure

```text
hybrid-rag-chatbot/
│
├── app.py                     # Streamlit UI
├── requirements.txt           # Dependencies
├── .env                       # API key and model (not committed)
├── .streamlit/
│   └── config.toml            # Theme settings
├── Dockerfile                 # Container build
├── data/                      # Sample PDFs
├── tests/
│   └── eval_questions.py      # Evaluation questions
│
├── vector_store/              # Generated FAISS and BM25 indexes (not committed)
│
└── rag/
    ├── __init__.py
    ├── loader.py              # PDF loading
    ├── splitter.py            # Chunking
    ├── embeddings.py          # Sentence-transformer wrapper
    ├── faiss_store.py         # Dense vector index
    ├── bm25_store.py          # Keyword index
    ├── hybrid_retriever.py    # Reciprocal Rank Fusion
    ├── reranker.py            # Cross-encoder re-ranking
    ├── generator.py           # Groq LLM prompts and calls
    └── pipeline.py            # Orchestration, including web fallback
```

`vector_store/` is rebuilt automatically from the PDFs you upload, so it isn't stored in the repository.

---

## 🔒 Privacy

Document chunks and questions are sent to Groq to generate answers. When an answer isn't in your documents, the rewritten question is sent to DuckDuckGo. Documents are stored locally in `vector_store/`.