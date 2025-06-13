import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
from example import SAMPLE_GUIDELINES
import shutil
from sentence_transformers import SentenceTransformer
import torch

def setup_models(force_reinit: bool = False):
    """Initialize and download all required models"""
    print("Downloading and caching models...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    try:
        # This will download and cache the model
        print(f"Downloading {model_name}...")
        _ = SentenceTransformer(model_name)
        print("Model downloaded and cached successfully!")
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

def setup_vector_store(force_reinit: bool = False):
    """Initialize and populate the vector store with guidelines"""
    guideline_store_path = "guideline_store"
    
    # If force reinit, remove existing store
    if force_reinit and os.path.exists(guideline_store_path):
        print("Removing existing vector store...")
        shutil.rmtree(guideline_store_path)
    
    # Check if store already exists
    if os.path.exists(guideline_store_path):
        print(f"Vector store already exists at {guideline_store_path}")
        return True
    
    try:
        print("Initializing HuggingFace Embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        print("Creating new vector store...")
        texts = []
        metadatas = []
        
        # Prepare guidelines for ingestion
        for guideline_id, content in tqdm(SAMPLE_GUIDELINES.items(), desc="Processing guidelines"):
            texts.append(content)
            metadatas.append({"guideline_id": guideline_id})
        
        # Create and persist vector store
        print("Creating Chroma vector store...")
        Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=guideline_store_path
        )
        print("Vector store created and persisted successfully!")
        return True
        
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        if os.path.exists(guideline_store_path):
            print("Cleaning up partial setup...")
            shutil.rmtree(guideline_store_path)
        return False

if __name__ == "__main__":
    print("Starting model and database setup...")
    
    if not setup_models():
        print("Failed to download and cache models. Setup incomplete.")
        exit(1)
        
    if not setup_vector_store():
        print("Failed to create vector store. Setup incomplete.")
        exit(1)
        
    print("Setup complete! You can now run the main application.") 