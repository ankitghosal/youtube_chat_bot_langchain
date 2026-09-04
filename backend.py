import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from youtube_transcript_api import TranscriptsDisabled, VideoUnavailable, YouTubeTranscriptApi

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="YouTube RAG Chatbot", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

video_cache: dict[str, dict[str, Any]] = {}


class LoadRequest(BaseModel):
    video: str = Field(min_length=1, max_length=500)


class ChatRequest(BaseModel):
    video_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=2000)


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured. Add it to your .env file.")
    os.environ["GEMINI_API_KEY"] = api_key
    return api_key


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def parse_video_id(raw: str) -> str:
    value = raw.strip()
    parsed = urlparse(value)
    hostname = parsed.hostname or ""
    if hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        return parse_qs(parsed.query).get("v", [value])[0]
    if hostname == "youtu.be":
        return parsed.path.strip("/").split("/")[0]
    return value.split("?")[0].split("&")[0]


def build_chain(video_id: str) -> dict[str, Any]:
    get_api_key()
    transcript = YouTubeTranscriptApi().fetch(
        video_id, languages=["en", "hi", "en-US", "en-GB"]
    )

    raw_chunks = list(transcript)
    if not raw_chunks:
        raise RuntimeError("No transcript segments were found for this video.")

    merged_docs: list[Document] = []
    window_texts: list[str] = []
    window_start = raw_chunks[0].start
    for chunk in raw_chunks:
        if chunk.start - window_start >= 30 and window_texts:
            merged_docs.append(
                Document(page_content=" ".join(window_texts), metadata={"start": window_start})
            )
            window_texts = []
            window_start = chunk.start
        window_texts.append(chunk.text)
    if window_texts:
        merged_docs.append(
            Document(page_content=" ".join(window_texts), metadata={"start": window_start})
        )

    docs = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_documents(
        merged_docs
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    batch_size = 50
    vectors: list[list[float]] = []
    for offset in range(0, len(docs), batch_size):
        batch = [document.page_content for document in docs[offset : offset + batch_size]]
        for attempt in range(6):
            try:
                vectors.extend(embeddings.embed_documents(batch))
                break
            except Exception as exc:
                message = str(exc)
                if "429" not in message and "RESOURCE_EXHAUSTED" not in message:
                    raise
                if attempt == 5:
                    raise RuntimeError("Embedding failed after repeated rate-limit retries.") from exc
                wait = 2.0 ** (attempt + 1)
                retry_delay = re.search(r"retryDelay.*?(\d+)s", message)
                if retry_delay:
                    wait = int(retry_delay.group(1)) + 2
                time.sleep(wait)

    vector_store = FAISS.from_embeddings(
        text_embeddings=[(document.page_content, vector) for document, vector in zip(docs, vectors)],
        embedding=embeddings,
        metadatas=[document.metadata for document in docs],
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    prompt = PromptTemplate(
        template="""You are a helpful assistant. Answer only from the provided transcript context.
If the context doesn't contain enough information, say you don't know.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2)
    chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            context=retriever
            | RunnableLambda(lambda found: "\n\n".join(document.page_content for document in found)),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return {
        "chain": chain,
        "retriever": retriever,
        "chunks": len(docs),
        "preview": " ".join(chunk.text for chunk in raw_chunks)[:300],
    }


def timestamp_links(video_id: str, documents: list[Any]) -> list[dict[str, str | int]]:
    seen: set[int] = set()
    links = []
    for document in sorted(documents, key=lambda item: item.metadata.get("start", 0)):
        start = document.metadata.get("start")
        if start is None or int(start) in seen:
            continue
        seconds = int(start)
        seen.add(seconds)
        links.append(
            {
                "label": format_timestamp(seconds),
                "seconds": seconds,
                "url": f"https://www.youtube.com/watch?v={video_id}&t={seconds}s",
            }
        )
    return links


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/load")
async def load_video(request: LoadRequest) -> dict[str, Any]:
    video_id = parse_video_id(request.video)
    if not video_id:
        raise HTTPException(status_code=400, detail="Enter a YouTube video ID or URL.")
    if video_id not in video_cache:
        try:
            video_cache[video_id] = await run_in_threadpool(build_chain, video_id)
        except TranscriptsDisabled as exc:
            raise HTTPException(status_code=422, detail="Transcripts are disabled for this video.") from exc
        except VideoUnavailable as exc:
            raise HTTPException(status_code=404, detail="Video unavailable or transcript cannot be retrieved.") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    cached = video_cache[video_id]
    return {"video_id": video_id, "chunks": cached["chunks"], "preview": cached["preview"]}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    cached = video_cache.get(request.video_id)
    if cached is None:
        raise HTTPException(status_code=409, detail="Load a video before asking a question.")
    try:
        documents = await run_in_threadpool(cached["retriever"].invoke, request.question)
        answer = await run_in_threadpool(cached["chain"].invoke, request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"answer": answer, "timestamps": timestamp_links(request.video_id, documents)}
