import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import faiss
import numpy as np

from src.models.model_manager import get_embedding_model, get_llm
from src.retrieval.vector_store import search_index
from src.retrieval.reranker import rerank_chunks
from src.models.llm_interface import generate_answer


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Research Assistant")
st.markdown("Chat with your research paper using AI")

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider("FAISS Top-K", 3, 10, 5)
    rerank_k = st.slider("Final Chunks", 1, 5, 3)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.write(
        "RAG Pipeline:\n"
        "- FAISS retrieval\n"
        "- Cross-encoder reranking\n"
        "- TinyLlama generation"
    )

    max_tokens = st.slider("Answer Length (tokens)", 100, 600, 300)


# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    embedding_model = get_embedding_model()
    tokenizer, llm_model = get_llm()
    return embedding_model, tokenizer, llm_model


@st.cache_resource
def load_data():
    index = faiss.read_index("data/faiss.index")
    chunks = np.load("data/chunks.npy", allow_pickle=True).tolist()
    return index, chunks


embedding_model, tokenizer, llm_model = load_models()
index, chunks = load_data()


# -------------------- PDF UPLOAD --------------------
uploaded_file = st.file_uploader("📄 Upload a PDF (optional)", type="pdf")

if uploaded_file:
    with open("data/raw/temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF uploaded! (Re-index manually for now)")


# -------------------- CHAT STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------- DISPLAY CHAT --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -------------------- INPUT --------------------
user_input = st.chat_input("Ask a question...")


# -------------------- MAIN LOGIC --------------------
if user_input:

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking... 🤔"):

            # 1️⃣ Embed query
            query_embedding = embedding_model.encode([user_input])

            # 2️⃣ FAISS search
            distances, indices = search_index(index, query_embedding, k=top_k)
            retrieved_chunks = [chunks[i] for i in indices[0]]

            # 3️⃣ Rerank
            reranked_chunks = rerank_chunks(user_input, retrieved_chunks)
            final_chunks = reranked_chunks[:rerank_k]

            # 4️⃣ Generate answer
            answer = generate_answer(
                user_input,
                final_chunks,
                tokenizer,
                llm_model,
                max_tokens=max_tokens
            )

        st.write(answer)

        # Confidence info
        st.caption(f"Retrieved {len(final_chunks)} relevant chunks")

        # Show sources
        st.markdown("### 📚 Sources")
        for i, chunk in enumerate(final_chunks):
            with st.expander(f"Source {i+1}"):
                st.write(chunk)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})


# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<center>RAG Pipeline</center>",
    unsafe_allow_html=True
)