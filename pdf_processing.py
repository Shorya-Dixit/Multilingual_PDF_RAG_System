import fitz  # PyMuPDF for PDF text extraction
import pytesseract
from PIL import Image
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Text Extraction Function
def extract_text_from_pdf(file_path):
    """Extract text from a PDF (scanned or digital)."""
    extracted_text = ""
    with fitz.open(file_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            extracted_text += page.get_text()

            # Extract text from images in case of scanned PDFs
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                extracted_text += pytesseract.image_to_string(image, lang="eng+hin+ben+chi_sim")
    return extracted_text

# Step 2: Chunking Function
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for better processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return text_splitter.split_text(text)

# Step 3: Embedding and Vector Store Creation
def create_vector_store(chunks, embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    # Initialize the HuggingFace Embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Ensure chunks are not empty
    if len(chunks) == 0:
        print("Error: No chunks available to generate embeddings.")
        return None
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    # Generate embeddings for the text chunks
    embeddings_list = embeddings.embed_documents(chunks)
    
    if len(embeddings_list) == 0 or len(embeddings_list[0]) == 0:
        print("Error: Empty embeddings generated. Check embedding model.")
        return None
    
    print(f"First embedding vector: {embeddings_list[0][:5]}")  # Print first 5 values of the first embedding for debug
    
    # Use FAISS from_texts instead of from_embeddings for proper initialization
    vector_store = FAISS.from_texts(chunks, embeddings)  # Pass chunks and embeddings directly
    
    return vector_store 
