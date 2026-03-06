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
Successfully extracted and printed the first portion of the document text.