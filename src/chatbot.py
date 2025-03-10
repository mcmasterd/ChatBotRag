import os
import asyncio
import chromadb
import gradio as gr
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional, Set, Tuple
from functools import lru_cache
import time
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
async_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name="text-embedding-3-small",
    dimensions=1536
)
collection = chroma_client.get_collection(
    name="scholarship-qa",
    embedding_function=embedding_function
)

# Replace the current cache implementation with this:
_response_cache = {}  # Simple dictionary to store cached responses

@lru_cache(maxsize=100)
def get_cached_response(query_key: str) -> str:
    """Return cached response if available"""
    return _response_cache.get(query_key)

def normalize_query(query: str) -> str:
    """Normalize query for better cache hits"""
    return ' '.join(query.lower().split())

# Category keyword mapping
CATEGORY_KEYWORDS = {
    "Scholarship": ["điều kiện", "yêu cầu", "tiêu chuẩn", "tiêu chí", "đủ điều kiện", "đáp ứng","học bổng"],
    "Decree": ["quy trình", "thủ tục", "các bước", "quá trình", "thực hiện", "đăng ký", "nộp hồ sơ"]
}

# Inverse index for faster keyword lookup
KEYWORD_TO_CATEGORY = {}
for category, keywords in CATEGORY_KEYWORDS.items():
    for keyword in keywords:
        KEYWORD_TO_CATEGORY[keyword] = category

def detect_categories(query: str) -> Set[str]:
    """
    Detect relevant categories from user query using keyword matching
    Returns a set of detected categories
    """
    query_lower = query.lower()
    detected_categories = set()
    
    # Direct keyword matching
    for keyword, category in KEYWORD_TO_CATEGORY.items():
        if keyword in query_lower:
            detected_categories.add(category)
    
    return detected_categories

def get_relevant_content(query: str, use_categories: bool = True) -> List[Dict]:
    """Get relevant content with hybrid retrieval approach"""
    start_time = time.time()
    
    # Step 1: Detect categories from query (only if we want to use categories)
    filter_dict = None
    if use_categories:
        categories = detect_categories(query)
        print(f"Detected categories: {categories}")
        
        # Step 2: Build filter based on detected categories
        if categories:
            # Handle filter construction differently based on number of categories
            if len(categories) == 1:
                # For a single category, no need for outer $or operator
                category = next(iter(categories))
                filter_dict = {"$or": [
                    {"category": category},
                    {"subcategory": category}
                ]}
            elif len(categories) > 1:
                # For multiple categories, build a more complex filter
                category_conditions = []
                for category in categories:
                    category_conditions.append({"category": category})
                    category_conditions.append({"subcategory": category})
                
                filter_dict = {"$or": category_conditions}
    
    # Step 3: Query with category filter + embedding similarity
    results = collection.query(
        query_texts=[query],
        n_results=3,  # Get more results when using filters for better coverage
        where=filter_dict
    )
    
    # Process results
    relevant_content = []
    for i in range(len(results['documents'][0])):
        relevant_content.append({
            'question': results['metadatas'][0][i].get('question', ''),
            'answer': results['metadatas'][0][i].get('answer', ''),
            'category': results['metadatas'][0][i].get('category', ''),
            'subcategory': results['metadatas'][0][i].get('subcategory', '')
        })
    
    print(f"Retrieval time: {time.time() - start_time:.4f}s, Results: {len(relevant_content)}")
    return relevant_content

def create_prompt(query: str, content: List[Dict]) -> str:
    """Create an improved prompt with category context"""
    # Extract categories for context
    categories = ", ".join(set([
        item.get('category', '') for item in content if item.get('category')
    ] + [
        item.get('subcategory', '') for item in content if item.get('subcategory')
    ]))
    
    # Include full context with category information
    context_items = []
    for item in content:
        context_str = f"Q: {item['question']}\nA: {item['answer']}"
        if item.get('category'):
            context_str += f"\nDanh mục: {item['category']}"
        if item.get('subcategory'):
            context_str += f"\nDanh mục con: {item['subcategory']}"
        context_items.append(context_str)
    
    context = "\n\n".join(context_items)
    
    prompt = f"""Trả lời câu hỏi dựa trên thông tin sau:
    
    {context}
    
    DANH MỤC LIÊN QUAN: {categories}
    
    CÂU HỎI: {query}
    
    TRẢ LỜI:"""
    
    return prompt

async def get_llm_response_async(prompt: str) -> str:
    """Asynchronous version of LLM response for better performance"""
    response = await async_client.chat.completions.create(
        model="gpt-3.5-turbo",  # Using a faster model for most queries
        messages=[
            {"role": "system", "content": "Bạn là trợ lý tư vấn học bổng. Trả lời ngắn gọn nhưng đầy đủ thông tin."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,  # Lower temperature for more consistent responses
        max_tokens=500,   # Reduce max tokens to speed up response
        presence_penalty=0.3
    )
    return response.choices[0].message.content

def get_llm_response(prompt: str) -> str:
    """Get LLM response with optimized parameters"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Using a faster model for most queries
        messages=[
            {"role": "system", "content": "Bạn là trợ lý tư vấn học bổng. Trả lời ngắn gọn nhưng đầy đủ thông tin."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,  # Lower temperature for more consistent responses 
        max_tokens=500,   # Reduce max tokens to speed up response
        presence_penalty=0.3
    )
    return response.choices[0].message.content

# Optional: fallback to more powerful model if needed
def get_advanced_llm_response(prompt: str, query_complexity: float) -> str:
    """Use more powerful model for complex queries"""
    if query_complexity > 0.8:  # Threshold for complex queries
        model = "gpt-4-0125-preview"
    else:
        model = "gpt-3.5-turbo"
        
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Bạn là trợ lý tư vấn học bổng. Trả lời ngắn gọn nhưng đầy đủ thông tin."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=600,
        presence_penalty=0.3
    )
    return response.choices[0].message.content

def process_user_query(query: str) -> str:
    """Process user query with hybrid search optimizations"""
    try:
        start_time = time.time()
        
        # Check cache for similar queries
        normalized_query = normalize_query(query)
        cached_result = get_cached_response(normalized_query)
        
        if cached_result is not None:
            print(f"Cache hit! Response time: {time.time() - start_time:.2f}s")
            return cached_result
        
        # Get relevant content using hybrid search with categories
        relevant_content = get_relevant_content(query, use_categories=True)
        
        # If no results found, try without category filtering
        if not relevant_content:
            print("No results with category filter, falling back to embedding-only search")
            relevant_content = get_relevant_content(query, use_categories=False)
        
        # Create and get response
        prompt = create_prompt(query, relevant_content)
        response = get_llm_response(prompt)
        
        # Update cache properly
        _response_cache[normalized_query] = response
        # Clear the function cache
        get_cached_response.cache_clear()
        
        print(f"Total response time: {time.time() - start_time:.2f}s")
        return response
        
    except Exception as e:
        return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"

# Gradio interface with streaming for better perceived performance
def chatbot_interface(message, history):
    return process_user_query(message)

# Configure Gradio
demo = gr.ChatInterface(
    fn=chatbot_interface,
    title="Trợ lý tư vấn Học bổng 🎓",
    description="Hãy đặt câu hỏi về học bổng, mình sẽ trả lời đầy đủ và nhanh chóng!",
    theme=gr.themes.Soft(),
    examples=[
        "Điều kiện để nhận học bổng là gì?",
        "Quy trình xét học bổng như thế nào?",
        "Thời gian xét học bổng là khi nào?"
    ]
)

if __name__ == "__main__":
    demo.queue()  # Removed unsupported parameter
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True
    )