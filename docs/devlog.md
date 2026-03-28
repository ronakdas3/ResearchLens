# Development Log

This file records the step-by-step development process of the **AI Research Assistant** project.

---

## Step 1 – Project Initialization

Created the main project directory.

Command used:

mkdir ai-research-assistant
cd ai-research-assistant

Purpose:

Create the root directory for the project repository.

---

## Step 2 – Initialize Git Repository

Initialized version control.

Commands used:

git init
git branch -M main

Purpose:

Track project changes and enable collaboration through GitHub.

---

## Step 3 – Create Python Virtual Environment

Created a dedicated Python environment for dependency isolation.

Command used:

python -m venv venv

Activation:

Windows
venv\Scripts\activate

Purpose:

Ensure that project dependencies are isolated from the global Python installation.

---

## Step 4 – Create Initial Project Files

Created the core repository files.

Files created:

.gitignore
README.md
requirements.txt
main.py

Purpose:

Provide basic project configuration and documentation.

---

## Step 5 – Install Initial Dependencies

Installed the main libraries required for the project.

Command used:

pip install torch transformers sentence-transformers faiss-cpu streamlit pymupdf

Saved dependency versions:

pip freeze > requirements.txt

Purpose:

Prepare the environment for machine learning, document processing, and building the user interface.

---

## Step 6 – Create Project Folder Structure

Created the main directories for the project.

Commands used:

mkdir data
mkdir notebooks
mkdir src
mkdir app
mkdir tests

Created subdirectories inside src:

mkdir src/data
mkdir src/embeddings
mkdir src/retrieval
mkdir src/models
mkdir src/training
mkdir src/evaluation
mkdir src/inference
mkdir src/utils

Purpose:

Organize the project into modular components separating data processing, models, training, evaluation, and inference logic.

---

## Step 7 – Track Folder Structure in Git(didnt actually do this)

Added `.gitkeep` files to ensure empty folders are tracked by Git.

Example:

touch src/data/.gitkeep

Purpose:

Git does not track empty folders, so placeholder files are used.

---

## Step 8 – Initial Commit

Committed the initial project setup.

Commands used:

git add .
git commit -m "Initial project setup with environment and project structure"

Purpose:

Create the first version checkpoint of the repository.

---

## Step 9 – Implement PDF Text Extraction

Created module for extracting text from research papers.

File created:
src/data/pdf_loader.py

Library used:
PyMuPDF

Purpose:
Extract raw text from uploaded research papers for further processing in the pipeline.

Test:
Ran the script using a sample research paper placed in data/raw/.

Command:
python src/data/pdf_loader.py

Result:
extract and print the first portion of the document text.

---

## Step 10 – Implement Text Chunking

Created module for splitting document text into smaller chunks.

File created:
src/data/text_chunker.py

Purpose:
Large documents must be split into smaller pieces before generating embeddings and storing them in a vector database.

Chunking parameters:
chunk_size = 500 characters
overlap = 50 characters

Overlap ensures contextual continuity between adjacent chunks.

Test:
Ran the module on sample research paper.

Command:
python src/data/text_chunker.py

Result:
generate multiple text chunks from the document.

---

## Step 11 – Implement Embedding Generation

Created module for converting text chunks into vector embeddings.

File created:
src/embeddings/embedding_generator.py

Model used:
Sentence-BERT (all-MiniLM-L6-v2)

Purpose:
Convert document chunks into numerical vectors representing semantic meaning.

Pipeline:

PDF → Text → Chunks → Embeddings

Test:
Ran embedding generator on sample research paper.

Command:
python src/embeddings/embedding_generator.py

Result:
generate embeddings for all text chunks.

---

## Step 12 – Fix Python Module Import Structure

Encountered import error:

ModuleNotFoundError: No module named 'src'

Solution:

Converted project folders into Python packages by adding `__init__.py` files.

Files added:
src/__init__.py
src/data/__init__.py
src/embeddings/__init__.py
src/retrieval/__init__.py
src/models/__init__.py
src/training/__init__.py
src/evaluation/__init__.py
src/inference/__init__.py
src/utils/__init__.py

This allows modules to be imported using the project package structure.

command to run as a module(from the root):

python -m src.embeddings.embedding_generator

---

## Step 13 – Implement Vector Database with FAISS

Created module for storing and searching document embeddings.

File created:
src/retrieval/vector_store.py

Library used:
FAISS

Purpose:
Store embeddings in a vector database and enable similarity search.

Pipeline:

PDF → Text → Chunks → Embeddings → Vector Index

Test:
Ran the module to build a FAISS index from generated embeddings.

Command:
python -m src.retrieval.vector_store

Result:
create vector index containing document embeddings.

---

## Step 14 – Implement Query Engine for Semantic Retrieval

Created module for retrieving relevant document chunks based on a user query.

File created:
src/inference/query_engine.py

Purpose:
Convert user questions into embeddings and perform similarity search on the FAISS index to retrieve the most relevant document chunks.

Pipeline:

User Query → Query Embedding → Vector Search → Retrieve Chunks

Test:
Executed the query engine on a sample research paper.

Command:
python -m src.inference.query_engine

Result:
retrieve ost relevant sections of the document for a given question.

note:
the retrieved chunks may not make the most sense they are just relevent chunks and we would need a LLM to generate answers

---

## Step 15 – Integrate Language Model for Answer Generation

Implemented LLM interface for generating answers using retrieved document context.

File created:
src/models/llm_interface.py

Model used:
FLAN-T5 (google/flan-t5-base)
used AutoTokenizer and AutoModelForSeq2SeqLM for now

Purpose:
Generate natural language answers using the relevant document chunks retrieved by the query engine.

Pipeline:

User Query → Query Embedding → Vector Search → Relevant Chunks → LLM → Generated Answer

Test:
Executed the query engine with LLM integration.

Command:
python -m src.inference.query_engine

Result:
generate answers based on retrieved document context.

---

## Step 16 – Add Persistent FAISS Index

Implemented a separate indexing pipeline to avoid rebuilding embeddings and vector indices for every query.

File created:
src/indexing/build_index.py

Functionality:
Processes documents, generates embeddings, builds a FAISS index, and saves the index and text chunks to disk.

Saved files:
data/faiss.index
data/chunks.npy

This separates the expensive indexing step from the query phase, significantly improving system efficiency.

---

## Step 17 – Add Centralized Model Manager

Implemented a centralized model loader to avoid repeated model initialization.

File created:
src/models/model_manager.py

Features:

* Lazy loading for embedding model and LLM
* Ensures models are loaded only once
* Reduces redundant memory usage and startup time

improves efficiency.

---

## Step 18 – Add Retrieval Reranking

Implemented a reranking stage to improve retrieval quality.

File created:
src/retrieval/reranker.py

Model used:
cross-encoder/ms-marco-MiniLM-L-6-v2

Pipeline improvement:

User Query → FAISS retrieval → Candidate chunks → Reranker → Best chunks → LLM answer

reranker evaluates query–chunk pairs and assigns relevance scores, improving context selection before passing data to the LLM.

---

## Step 19 – Add Prompt Templates and Source Citations

Implemented structured prompt templates for the RAG pipeline.

File created:
src/models/prompt_templates.py

Features:

* Context formatting with chunk identifiers
* Instructions restricting the model to provided context
* Citation-style references in generated answers

Pipeline improvement:

Retrieved chunks → Prompt template → LLM answer with citations

improves answer reliability, reduces hallucinations, and adds explainability.

---

## Step 20 – Build Streamlit UI

Created a Streamlit-based web interface for interacting with the RAG pipeline.

Features:

* User input for queries
* Integration with backend retrieval + generation pipeline
* Display of answers and source chunks

convert project to a interactive AI application.

---

## Step 21 – Add Chat Interface

Upgraded the UI to support a conversational chat interface using Streamlit session state.

Features:

* Maintains chat history across interactions
* Displays user and assistant messages in chat format
* Improves usability and user experience

interactive chat(kind of)

---

## Step 22 – Upgrade LLM to TinyLlama

Replaced FLAN-T5 with TinyLlama (TinyLlama-1.1B-Chat).

Reason:

* Better handling of long-form context
* Improved response quality
* More suitable for instruction-based generation

This significantly improved answer accuracy and relevance.

---

## Step 23 – Add Configurable Answer Length

Introduced dynamic control over LLM output length.

Features:

* Added UI slider to control max tokens
* Updated generation function to accept dynamic token limits
* Improved prompt to encourage detailed responses

some answers were too long but llm was limited. this removes limitations for long answers.

---

## Step 24 – Multi-Paper Support

Implemented support for multiple research papers.

Features:

* Separate FAISS index per paper
* Dynamic loading based on user selection
* Paper selection UI in Streamlit
* Chat reset when switching papers

improved usability and make the system scalable to multiple documents.

