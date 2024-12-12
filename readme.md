# Multilingual PDF RAG System

This repository provides a **Multilingual PDF Retrieval-Augmented Generation (RAG) System**. The system allows processing PDFs, extracting their text, creating a vector store, and performing hybrid semantic and keyword-based search. It answers user queries by retrieving relevant information from PDFs and using a language model for response generation.

## Features

- **Text extraction**: Supports both digital and scanned PDFs using `PyMuPDF` and `Tesseract`.
- **Multilingual support**: Processes text in multiple languages (e.g., English, Hindi, Bengali, Simplified Chinese).
- **Chunking**: Splits text into manageable overlapping chunks for efficient processing.
- **Hybrid search**: Combines semantic search (FAISS) with keyword-based search (BM25).
- **Conversational memory**: Tracks user queries and context for coherent responses.
- **Configurable embeddings and LLM models**: Choose your preferred embedding and LLM models.

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/Shorya-Dixit/Multilingual_PDF_RAG_System.git
cd your-repo-name
```

### Step 2: Create a virtual environment

I recommend using a virtual environment to manage dependencies.

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

Install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 4: Install Tesseract OCR (for scanned PDFs)

Tesseract must be installed on your system.

- **Ubuntu**:
  ```bash
  sudo apt update
  sudo apt install tesseract-ocr
  ```
- **Windows**:
  Download and install from [Tesseract's GitHub page](https://github.com/tesseract-ocr/tesseract).
- **MacOS**:
  ```bash
  brew install tesseract
  ```

### Step 5: Set the PDF Directory

Set the `pdf_dir` variable in `main.py` to the path of the folder containing your PDFs:

```python
pdf_dir = " "
```

### Step 6: Log in via Hugging Face CLI

Once you have your Hugging Face token, you can authenticate using the Hugging Face CLI:

1. Install the `huggingface_hub`:

```bash
pip install huggingface_hub
```

2. Use the CLI to log in:

```bash
huggingface-cli login
```

3. Paste your Hugging Face token when prompted.

## Usage

### Running the Pipeline

To start the system, run the `main.py` script:

```bash
python main.py
```

### Interactive Query Mode

Once the script is running:

- Type your queries into the prompt.
- Enter `exit` to stop the program.

### Directory Structure

```
.
|-- main.py
|-- pdf_processing.py
|-- rag_query.py
|-- requirements.txt
|-- pdfs/       # Directory containing PDF files
```

## Project Components

### 1. `main.py`

The main pipeline orchestrates the process:

- Extracts text from PDFs in the specified directory.
- Chunks the text for efficient embedding.
- Creates a vector store using the specified embedding model.
- Sets up a conversational query interface for user interaction.

### 2. `pdf_processing.py`

Contains functions for:

- Extracting text from PDFs (supports digital and scanned PDFs).
- Splitting text into chunks with overlap.
- Creating a FAISS-based vector store with embeddings.

### 3. `rag_query.py`

Handles query processing:

- Combines semantic search (FAISS) with keyword-based search (BM25).
- Uses the retrieved results to generate responses with a conversational LLM.

## Configuration

- **Embedding model**: Default is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **LLM model**: Default is `meta-llama/Llama-2-7b-chat-hf`.

You can modify these in the `main_pipeline` function in `main.py`.

## Requirements

- Python >= 3.8
- `PyMuPDF`, `Pillow`, `pytesseract`
- `langchain`, `sentence-transformers`, `rank-bm25`
- Tesseract OCR (for scanned PDFs)

## Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests.

## Acknowledgments

This project leverages:

- [LangChain](https://github.com/hwchase17/langchain) for text processing and retrieval.
- [FAISS](https://github.com/facebookresearch/faiss) for semantic search.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for scanned PDF text extraction.

---

Happy querying! 🚀

