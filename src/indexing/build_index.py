import numpy as np
import faiss

from src.data.pdf_loader import extract_text_from_pdf
from src.data.text_chunker import chunk_text
from src.embeddings.embedding_generator import generate_embeddings
from src.retrieval.vector_store import build_vector_index
from src.data.text_cleaning import remove_references


def build_and_save_index(pdf_path):

    print("Loading document...")
    text = extract_text_from_pdf(pdf_path)
    text = remove_references(text)

    print("Chunking document...")
    chunks = chunk_text(text)
    # chunks = [c for c in chunks if len(c) > 200]

    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks)

    print("Building FAISS index...")
    index = build_vector_index(embeddings)

    print("Saving index and chunks...")

    faiss.write_index(index, "data/faiss.index")
    np.save("data/chunks.npy", np.array(chunks, dtype=object))

    print("Index saved successfully.")


if __name__ == "__main__":

    pdf_path = "data/raw/sample_paper.pdf"

    build_and_save_index(pdf_path)