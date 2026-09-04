# 🎬 YouTube RAG Chatbot

An intelligent video conversation agent built with **LangChain**, **Google Gemini**, **FastAPI**, and a retro HTML/CSS/JavaScript client. Paste a YouTube link, ingest its transcript, and ask questions about the content with precise timestamp references.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-orange.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)

---

## 🌟 Features

- **Instant Ingestion**: Extract transcripts from any YouTube video via ID or URL.
- **Smart Chunking**: Implements a custom 30-second windowing strategy to preserve semantic context and significantly reduce API overhead.
- **Timestamped Citations**: Answers include direct "Jump-to" links to the exact moments in the video where the information was discussed.
- **Multilingual Support**: Automatically fetches transcripts in English, Hindi, and regional variations.
- **Rate-Limit Resilience**: Built-in exponential backoff and retry logic to handle Google Gemini API `429 RESOURCE_EXHAUSTED` errors gracefully.
- **Minimal UI**: A lightweight retro terminal interface with full in-browser chat history.

## 🛠️ Tech Stack

- **LLM**: Google Gemini 1.5 Flash (via `langchain-google-genai`)
- **Embeddings**: Google Generative AI Embeddings (`models/gemini-embedding-001`)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Orchestration**: LangChain Expression Language (LCEL)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: HTML, CSS, and vanilla JavaScript

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- A Google AI Studio API Key ([Get it here](https://aistudio.google.com/app/apikey))

### 2. Installation
bash git clone https://github.com/ankitghosal/youtube_chat_bot_langchain.git cd youtube_chat_bot_langchain pip install -r requirements.txt

### 3. Configuration
Create a `.env` file in the root directory and add your Gemini API key:
env GEMINI_API_KEY=your_api_key_here

### 4. Running the App

Start the API and frontend with:

```bash
uvicorn backend:app --reload
```

Open http://127.0.0.1:8000 in your browser.
## 📖 How it Works

1. **Transcript Fetching**: Uses `youtube-transcript-api` to retrieve raw text segments.
2. **Context Windowing**: Merges tiny raw transcript segments into 30-second blocks. This preserves the flow of conversation and ensures the retriever finds meaningful context.
3. **Vector Indexing**: Chunks are embedded and stored in a local FAISS index for high-speed similarity search.
4. **RAG Chain**: 
   - The system retrieves the top relevant 30-second segments for a query.
   - Gemini synthesizes a response constrained *strictly* to the provided transcript context.
   - Metadata is mapped back to YouTube timestamps for interactive citation buttons.

## 🤝 Contributing
Contributions are welcome! Whether it's improving the chunking logic, adding support for more vector databases, or enhancing the UI, feel free to open a PR.

