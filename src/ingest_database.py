import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from typing import Dict, List
import glob

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
            
            if not qa_pairs:
                print(f"Warning: No QA pairs found in {file_path}")
                return 0
            
            # Prepare data for batch upload
            ids = []
            documents = []
            metadatas = []
            
            for i, qa_pair in enumerate(qa_pairs):
                # Create unique ID using filename and index for better traceability
                file_basename = os.path.basename(file_path).replace('.json', '')
                unique_id = f"{file_basename}_qa_{i}"
                
                # Combine question and answer for document
                document = f"Câu hỏi: {qa_pair['question'].strip()} Trả lời: {qa_pair['answer'].strip()}"
                
                # Prepare metadata
                metadata = {
                    'question': qa_pair['question'],
                    'answer': qa_pair['answer'],
                    'category': qa_pair.get('category', ''),
                    'subcategory': qa_pair.get('subcategory', ''),
                    'source': file_path,
                    'version': data.get('version', '1.0')  # Assume a version field in JSON
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
                    print(f"Processed {i + 1}/{len(qa_pairs)} QA pairs from {file_path}")
                    ids, documents, metadatas = [], [], []
            
            # Upload any remaining items
            if ids:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                print(f"Processed all {len(qa_pairs)} QA pairs from {file_path}")
            
            return len(qa_pairs)
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 0

def process_all_files(data_folder: str, collection_name: str):
    """Process all JSON files in the data folder"""
    # Find all JSON files in the data folder
    json_files = glob.glob(os.path.join(data_folder, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {data_folder}")
        return
    
    print(f"Found {len(json_files)} JSON files in {data_folder}")
    
    total_processed = 0
    for file_path in json_files:
        print(f"\nProcessing: {file_path}")
        count = process_and_upload_file(file_path, collection_name)
        total_processed += count
    
    # Get final collection count
    collection = chroma_client.get_collection(
        name=collection_name,
        embedding_function=openai_ef
    )
    
    print(f"\nEmbedding complete!")
    print(f"Total QA pairs processed: {total_processed}")
    print(f"Total items in collection: {collection.count()}")

if __name__ == "__main__":
    data_folder = "data"
    collection_name = "scholarship-qa"
    
    try:
        # Process and upload all JSON files in the data folder
        process_all_files(data_folder, collection_name)
        print("Processing completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
