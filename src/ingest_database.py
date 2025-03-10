import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from typing import Dict, List

# Load environment variables
load_dotenv()

# Initialize OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Initialize OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name="text-embedding-3-small"
)

def process_and_upload_file(file_path: str, collection_name: str):
    """Process a JSON file and upload embeddings to ChromaDB"""
    try:
        # Create or get collection
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef
        )
        
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            qa_pairs = data.get('qa_pairs', [])
            
            # Prepare data for batch upload
            ids = []
            documents = []
            metadatas = []
            
            for i, qa_pair in enumerate(qa_pairs):
                # Create unique ID
                unique_id = f"qa_{i}"
                
                # Combine question and answer for document
                document = f"Câu hỏi: {qa_pair['question'].strip()} Trả lời: {qa_pair['answer'].strip()}"
                
                # Prepare metadata
                metadata = {
                    'question': qa_pair['question'],
                    'answer': qa_pair['answer'],
                    'category': qa_pair.get('category', ''),
                    'subcategory': qa_pair.get('subcategory', ''),
                    'source': file_path
                }
                
                ids.append(unique_id)
                documents.append(document)
                metadatas.append(metadata)
                
                # Upload in batches of 100
                if len(ids) >= 100:
                    collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                    print(f"Processed {i + 1}/{len(qa_pairs)} QA pairs")
                    ids, documents, metadatas = [], [], []
            
            # Upload any remaining items
            if ids:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                print(f"Processed all {len(qa_pairs)} QA pairs")
        
        print(f"Total items in collection: {collection.count()}")
        
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        raise

def search_qa_pairs(query: str, n_results: int = 3):
    """Search for similar QA pairs"""
    collection = chroma_client.get_collection(
        name="scholarship-qa",
        embedding_function=openai_ef
    )
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return results

if __name__ == "__main__":
    file_path = "data/NĐ 84_QĐ ve HB.json"
    collection_name = "scholarship-qa"
    
    try:
        # Process and upload data
        process_and_upload_file(file_path, collection_name)
        print("Processing completed successfully!")
        
        # Example search (uncomment to test)
        # test_query = "Cho tôi biết về học bổng"
        # results = search_qa_pairs(test_query)
        # print("\nSearch Results:")
        # print(results)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
