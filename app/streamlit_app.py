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
st.markdown("Chat with research papers using AI")


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider("FAISS Top-K", 3, 10, 5)
    rerank_k = st.slider("Final Chunks", 1, 5, 3)
    max_tokens = st.slider("Answer Length", 100, 600, 300)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.write(
        "RAG Pipeline:\n"
        "- FAISS retrieval\n"
        "- Cross-encoder reranking\n"
        "- TinyLlama generation"
    )


# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    embedding_model = get_embedding_model()
    tokenizer, llm_model = get_llm()
    return embedding_model, tokenizer, llm_model


embedding_model, tokenizer, llm_model = load_models()


# -------------------- PAPER SELECTION --------------------
st.subheader("📚 Choose a Paper")

paper_option = st.selectbox(
    "Select a research paper",
    ["Transformer", "ResNet"]
)

# Paper descriptions
if paper_option == "Transformer":
    st.caption("Attention Is All You Need (Vaswani et al., 2017)")

elif paper_option == "ResNet":
    st.caption("Deep Residual Learning for Image Recognition (He et al., 2015)")


# -------------------- RESET CHAT ON SWITCH --------------------
if "last_paper" not in st.session_state:
    st.session_state.last_paper = paper_option

if st.session_state.last_paper != paper_option:
    st.session_state.messages = []
    st.session_state.last_paper = paper_option


# -------------------- MAP PAPER TO PATHS --------------------
def get_paper_paths(option):

    if option == "Transformer":
        return (
            "data/papers/transformer/faiss.index",
            "data/papers/transformer/chunks.npy"
        )

    elif option == "ResNet":
        return (
            "data/papers/resnet/faiss.index",
            "data/papers/resnet/chunks.npy"
        )


# -------------------- LOAD DATA --------------------
@st.cache_resource
def load_data(index_path, chunks_path):
    index = faiss.read_index(index_path)
    chunks = np.load(chunks_path, allow_pickle=True).tolist()
    return index, chunks


index_path, chunks_path = get_paper_paths(paper_option)
index, chunks = load_data(index_path, chunks_path)


# -------------------- CHAT STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------- DISPLAY CHAT --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -------------------- INPUT --------------------
user_input = st.chat_input("Ask a question...")


# -------------------- MAIN PIPELINE --------------------
if user_input:

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking... 🤔"):

            # 1️⃣ Embed query
            query_embedding = embedding_model.encode([user_input])

            # 2️⃣ Retrieve
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

        # Info
        st.caption(f"Retrieved {len(final_chunks)} relevant chunks")

        # Sources
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