import os
import re
import time
import math
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, VideoUnavailable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


load_dotenv()

# os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

api_key = os.getenv("GEMINI_API_KEY")

# fallback to Streamlit secrets
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        st.error("API key not found. Set it in .env or Streamlit secrets.")
        st.stop()

os.environ["GEMINI_API_KEY"] = api_key


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 YouTube RAG Chatbot")
st.caption("Paste a YouTube Video ID, load its transcript, and ask anything about the video.")


# ── Helper: format seconds → MM:SS or HH:MM:SS ────────────────────────────────
def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ── Helper: extract video ID from URL or bare ID ───────────────────────────────
def parse_video_id(raw: str) -> str:
    """Accept a bare ID or a youtube.com / youtu.be URL and return just the ID."""
    raw = raw.strip()
    if "youtube.com/watch" in raw:
        import urllib.parse as urlp
        params = urlp.parse_qs(urlp.urlparse(raw).query)
        return params.get("v", [raw])[0]
    if "youtu.be/" in raw:
        return raw.split("youtu.be/")[-1].split("?")[0]
    return raw  # assume it's already a bare ID


# ── Helper: build the RAG chain ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_chain(video_id: str):
    """Fetch transcript → merge → split → embed → build chain. Cached by video_id."""

    # 1. Fetch transcript
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.fetch(video_id, languages=["en", "hi", "en-US", "en-GB"])
    raw_chunks = list(transcript_list)  # each has .text and .start

    # 2. Merge raw chunks into ~30-second windows before splitting.
    #    Raw API chunks are typically 1–3 words (~5–15 chars) each.
    #    Creating one Document per raw chunk produces thousands of tiny docs
    #    and massively inflates embedding call count.
    #    Merging into 30s windows reduces doc count ~15x while keeping
    #    timestamp accuracy within 30 seconds.
    WINDOW_SECONDS = 30
    merged_docs = []
    window_texts = []
    window_start = raw_chunks[0].start if raw_chunks else 0.0

    for chunk in raw_chunks:
        if chunk.start - window_start >= WINDOW_SECONDS and window_texts:
            merged_docs.append(Document(
                page_content=" ".join(window_texts),
                metadata={"start": window_start},
            ))
            window_texts = []
            window_start = chunk.start
        window_texts.append(chunk.text)

    if window_texts:  # flush last window
        merged_docs.append(Document(
            page_content=" ".join(window_texts),
            metadata={"start": window_start},
        ))

    # 3. Split merged windows into embedding-sized chunks.
    #    2000-char chunks (vs old 1000) halves the number of embedding calls.
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = splitter.split_documents(merged_docs)

    # 4. Embed + vector store
    #    Free tier: 100 embed requests/min. Batching + retry with back-off.
    BATCH_SIZE = 50
    MAX_RETRIES = 6
    BACKOFF_BASE = 2.0

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    progress_text = st.empty()
    progress_bar = st.progress(0)

    all_texts = [d.page_content for d in docs]
    all_metas = [d.metadata for d in docs]
    n_batches = math.ceil(len(all_texts) / BATCH_SIZE)
    all_vectors = []

    for i in range(n_batches):
        batch_texts = all_texts[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        progress_bar.progress(int(i / n_batches * 100))
        progress_text.text(f"Embedding batch {i + 1}/{n_batches}…")

        for attempt in range(MAX_RETRIES):
            try:
                vecs = embeddings.embed_documents(batch_texts)
                all_vectors.extend(vecs)
                break
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = BACKOFF_BASE ** (attempt + 1)
                    m = re.search(r"retryDelay.*?(\d+)s", err_str)
                    if m:
                        wait = int(m.group(1)) + 2
                    if attempt < MAX_RETRIES - 1:
                        progress_text.text(
                            f"Rate limited on batch {i + 1}. "
                            f"Retrying in {int(wait)}s (attempt {attempt + 1}/{MAX_RETRIES})…"
                        )
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"Embedding failed after {MAX_RETRIES} retries. "
                            "Try a shorter video or wait a minute before reloading."
                        ) from exc
                else:
                    raise

    progress_bar.progress(100)
    progress_text.text("Building vector index…")

    text_embedding_pairs = list(zip(all_texts, all_vectors))
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embeddings,
        metadatas=all_metas,
    )

    progress_bar.empty()
    progress_text.empty()

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 5. Prompt
    prompt = PromptTemplate(
        template="""You are a helpful assistant. Answer only from the provided transcript context.
If the context doesn't contain enough information, say you don't know.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )

    # 6. Chain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
    parallel_chain = RunnableParallel({
        "question": RunnablePassthrough(),
        "context": retriever | RunnableLambda(
            lambda docs: "\n\n".join(d.page_content for d in docs)
        ),
    })
    chain = parallel_chain | prompt | llm | StrOutputParser()

    full_transcript = " ".join(chunk.text for chunk in raw_chunks)
    return chain, retriever, len(docs), full_transcript[:300]


# ── Main UI ─────────────────────────────────────────────────────────────────────
video_input = st.text_input(
    "YouTube Video ID or URL",
    placeholder="e.g. dQw4w9WgXcQ  or  https://www.youtube.com/watch?v=dQw4w9WgXcQ",
)

load_btn = st.button("🚀 Load Video", use_container_width=True, type="primary")

# Session-state to persist the chain across reruns
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "loaded_id" not in st.session_state:
    st.session_state.loaded_id = ""

# ── Load video ──────────────────────────────────────────────────────────────────
if load_btn:
    if not video_input.strip():
        st.error("Please enter a YouTube video ID or URL.")
    else:
        vid = parse_video_id(video_input)
        if vid == st.session_state.loaded_id and st.session_state.chain is not None:
            st.info("This video is already loaded. Ask your question below!")
        else:
            with st.spinner("Fetching transcript and building vector index…"):
                try:
                    chain, retriever, n_chunks, preview = build_chain(vid)
                    st.session_state.chain = chain
                    st.session_state.retriever = retriever
                    st.session_state.loaded_id = vid
                    st.session_state.chat_history = []
                    st.success(f"✅ Loaded **{n_chunks}** chunks from video `{vid}`")
                    with st.expander("Transcript preview"):
                        st.write(preview + " …")
                except TranscriptsDisabled:
                    st.error("❌ Transcripts are disabled for this video.")
                except VideoUnavailable:
                    st.error("❌ Video unavailable or transcript cannot be retrieved.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ── Chat interface ───────────────────────────────────────────────────────────────
if st.session_state.chain is not None:
    st.markdown("---")
    st.subheader("💬 Ask about the video")

    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])
            if entry["role"] == "assistant" and entry.get("timestamps"):
                st.markdown("**🕐 Discussed at:**")
                cols = st.columns(len(entry["timestamps"]))
                for col, (label, url) in zip(cols, entry["timestamps"]):
                    col.link_button(label, url)

    user_question = st.chat_input("Type your question here…")
    if user_question:
        # Show user message
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Retrieve source docs to extract timestamps
        source_docs = st.session_state.retriever.invoke(user_question)

        # Deduplicate timestamps (sort by start time, keep unique seconds)
        seen = set()
        timestamps = []
        for doc in sorted(source_docs, key=lambda d: d.metadata.get("start", 0)):
            start = doc.metadata.get("start")
            if start is None:
                continue
            start_sec = int(start)
            if start_sec not in seen:
                seen.add(start_sec)
                label = f"▶ {format_timestamp(start_sec)}"
                url = f"https://www.youtube.com/watch?v={st.session_state.loaded_id}&t={start_sec}s"
                timestamps.append((label, url))

        # Get RAG answer
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            with st.spinner("Thinking…"):
                full_answer = st.session_state.chain.invoke(user_question)
            answer_placeholder.write(full_answer)

            # Show timestamp buttons below the answer
            if timestamps:
                st.markdown("**🕐 Discussed at:**")
                cols = st.columns(len(timestamps))
                for col, (label, url) in zip(cols, timestamps):
                    col.link_button(label, url)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_answer,
            "timestamps": timestamps,
        })

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

elif not load_btn:
    st.info("👆 Enter a video ID or URL and click **Load Video** to get started.")