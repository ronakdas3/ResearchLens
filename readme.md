## Project Modules

### PDF Loader

Location:
src/data/pdf_loader.py

Purpose:
Extract text from research papers in PDF format.

### Text Chunking

Location:
src/data/text_chunker.py

Purpose:
Split extracted document text into overlapping chunks before embedding generation.

python src/embeddings/embedding_generator.py

### Embedding Generation

Location:
src/embeddings/embedding_generator.py

Purpose:
Convert text chunks into vector embeddings using Sentence-BERT.
These embeddings enable semantic search within documents.