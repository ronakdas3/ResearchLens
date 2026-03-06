# System Architecture

## Project Overview

The AI Research Assistant is a system that allows users to upload research papers and interact with them through natural language queries.

The system uses **Retrieval Augmented Generation (RAG)** to answer questions using information extracted from uploaded documents.

---

## High Level Pipeline

PDF Upload
↓
Text Extraction
↓
Text Chunking
↓
Embedding Generation
↓
Vector Database Storage
↓
Semantic Search
↓
Large Language Model
↓
Generated Answer

---

## Component Breakdown

### 1. Data Processing

Located in:

src/data/

Responsibilities:

* Load PDF documents
* Extract raw text
* Clean and preprocess text
* Split text into manageable chunks

---

### 2. Embedding Generation

Located in:

src/embeddings/

Responsibilities:

* Convert text chunks into vector embeddings
* Use pretrained embedding models
* Prepare vectors for similarity search

---

### 3. Vector Database

Located in:

src/retrieval/

Responsibilities:

* Store document embeddings
* Perform similarity search
* Retrieve relevant text chunks for a query

---

### 4. Query Engine

Located in:

src/inference/

Responsibilities:

* Accept user questions
* Retrieve relevant document sections
* Pass context to the language model
* Generate answers

---

### 5. User Interface

Located in:

app/

Responsibilities:

* Upload PDF documents
* Display answers
* Provide an interactive query interface

---

## Future Improvements

* Multi-document querying
* Citation highlighting
* Knowledge graph generation
* Research paper comparison
