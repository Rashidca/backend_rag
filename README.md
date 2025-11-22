
# 📘 Camera Manual RAG — Local LLM (Ollama + Metadata-Aware Retrieval)

This project implements an **advanced Retrieval-Augmented Generation (RAG)** system that answers questions from a **technical PDF manual** with **chapter-aware metadata filtering**.
Unlike standard RAG pipelines that search blindly across the whole document, this system detects the **relevant chapter first** and retrieves only those chunks — delivering far more accurate answers.

🧠 **Embeddings:** Hugging Face BGE Small
🤖 **LLM:** **Ollama — Qwen 2.5 3B Instruct (local CPU inference)**
📄 **Dataset:** Camera user manual (but you can replace with ANY manual)

---

## 🚀 Features

* Chapter-aware vector search (metadata filtering)
* Rejects hallucination — answers **only from provided chunks**
* Streamlit UI for user queries
* FastAPI backend
* Performance logging:

  * Retrieval latency
  * Generation latency
* Works fully **offline** once dependencies + Ollama model installed

---

## 📺 Demo Video

📌 *YouTube Link:*
▶️ [*(paste your link here)*](https://youtu.be/zT2QTyO-Ezw)

---

## 📂 Project Structure

```
rag-manual/
│
├── data/
│   └── manual.pdf              ← place your PDF here
│
├── ingestion.py                ← one-time ingestion pipeline
├── retriever.py                ← hybrid semantic + chapter filtering
├── models.py                   ← embeddings + local LLM caller
├── query_api.py                ← FastAPI runtime service
├── chat_ui.py                  ← Streamlit UI
│
├── requirements.txt
```



## 🛠 Setup

### 1️⃣ Install Ollama

[https://ollama.com/](https://ollama.com/)

Then pull the model:

```bash
ollama pull qwen2.5:3b-instruct
```

### 2️⃣ Create and activate virtual environment

```bash
python -m venv venv
```

```bash
# Windows
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Ingest the Manual

Add your manual here:

```
data/manual.pdf
```

Then run:

```bash
python ingestion.py
```

This creates the Chroma vector database.

---

## ▶️ Run Backend API

```bash
python query_api.py
```

Backend runs at:

```
http://localhost:8000/query
```

---

## 💻 Launch UI

```bash
streamlit run chat_ui.py
```

Ask your questions — example:

```
What are the steps in the Quick Start Guide for the D3300?
```

If the answer isn't present in the retrieved context, the system replies:

```
"I don't know based on this manual."
```

---

## ⚡ Performance on My Machine (No GPU)

| Component  | Latency                     |
| ---------- | --------------------------- |
| Retrieval  | ⏳ ~3–4 seconds              |
| Generation | ⏳ ~20-30 seconds (CPU only) |

Generation can be much faster with GPU models.

---

## 🖼 Screenshots 

```
/assets/streamlit.png
/assets/logs.png
/assets/context.png
```

---

## 🔮 Future Improvements

* GPU inference for faster generation
* OCR for manuals containing images
* Support for multiple manuals
* Dashboard for testing multiple models

---

## ⭐ Final Notes

* You can use *any* manual — just rename it to **manual.pdf** and place it in `/data`
* Ollama must be running while using the system
* Run ingestion only once unless you change the PDF

