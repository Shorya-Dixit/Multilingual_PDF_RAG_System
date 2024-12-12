import os
from pdf_processing import extract_text_from_pdf, chunk_text, create_vector_store
from rag_query import rag_query
from langchain.memory import ConversationBufferMemory

def main_pipeline(pdf_directory, embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """Process PDFs, create a RAG pipeline, and handle queries."""
    chunks = []
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, pdf_file)
            extracted_text = extract_text_from_pdf(file_path)
            chunks.extend(chunk_text(extracted_text))

    # Create vector store
    vector_store = create_vector_store(chunks, embedding_model)

    # Setup chat memory
    chat_memory = ConversationBufferMemory()

    # Interact with the RAG system
    print("Enter your queries. Type 'exit' to stop.")
    while True:
        query = input("Query: ")
        if query.lower() == "exit":
            break
        response = rag_query(vector_store, query, chat_memory, chunks)
        print("Response:", response)

if __name__ == "__main__":
    pdf_dir = "pdfs"  # Replace with the actual directory path
    main_pipeline(pdf_dir)
