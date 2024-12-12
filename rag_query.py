from rank_bm25 import BM25Okapi
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline
import os

# Step 4: Hybrid Search (Semantic + Keyword)
def hybrid_search(vector_store, query, chunks):
    """Combine semantic search with keyword-based BM25 search."""
    # Semantic search using FAISS
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    semantic_results = retriever.get_relevant_documents(query)
    semantic_texts = [doc.page_content for doc in semantic_results]

    # Keyword search using BM25
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    keyword_scores = bm25.get_scores(query.split())
    keyword_results = [chunk for _, chunk in sorted(zip(keyword_scores, chunks), reverse=True)[:10]]

    # Combine results, giving preference to semantic results
    combined_results = list(dict.fromkeys(semantic_texts + keyword_results))
    return combined_results

# Step 5: RAG Query Handling
def rag_query(vector_store, query, chat_memory, chunks):
    """Perform a RAG query on the vector store with hybrid search."""
    ranked_texts = hybrid_search(vector_store, query, chunks)

    # Initialize the HuggingFace model using transformers pipeline
    llm_model = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

    # Use the top-ranked documents for LLM response
    chain = ConversationalRetrievalChain.from_llm(
        llm_model,  # Use the HuggingFace model directly from transformers
        retriever=vector_store.as_retriever(),
        memory=chat_memory
    )
    
    # Prepare inputs for the chain
    inputs = {
        "input_documents": ranked_texts[:5],
        "question": query,
        "chat_history": chat_memory.messages  # Use the messages attribute directly
    }
    
    response = chain.invoke(inputs)  
    return response
