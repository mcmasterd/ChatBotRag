import os
import chromadb
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Set, Tuple, Optional
import time
import re
import threading
import hashlib
from functools import lru_cache

# Load environment variables and initialize clients
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name="text-embedding-3-small"
)
collection = chroma_client.get_collection(name="scholarship-qa", embedding_function=embedding_function)

# Enhanced cache with semantic similarity support
_response_cache = {}
_embedding_cache = {}

# Category keyword mapping with inverse index for fast lookup
CATEGORY_KEYWORDS = {
    "Scholarship": ["điều kiện", "yêu cầu", "tiêu chuẩn", "tiêu chí", "đủ điều kiện", "đáp ứng", "học bổng", "khuyến khích"],
    "Decree": ["quy trình", "thủ tục", "các bước", "quá trình", "thực hiện", "nghị định", "nộp hồ sơ"],
    "Timeline": ["thời gian", "thời hạn", "khi nào", "hạn cuối", "deadline", "lịch trình"]
}
KEYWORD_TO_CATEGORY = {kw: cat for cat, keywords in CATEGORY_KEYWORDS.items() for kw in keywords}

def hash_query(query: str) -> str:
    """Create a hash for the query string"""
    return hashlib.md5(query.lower().encode()).hexdigest()

def extract_source_info(text: str) -> str:
    """Extract document references from text with improved patterns"""
    patterns = {
        'decree': r'(?:Nghị\s*định|NĐ)(?:\s*số\s*)?([\d\/\-]+[\w\-]*)',
        'decision': r'(?:Quyết\s*định|QĐ)(?:\s*số\s*)?([\d\/\-]+[\w\-]*)',
        'circular': r'(?:Thông\s*tư|TT)(?:\s*số\s*)?([\d\/\-]+[\w\-]*)',
        'document': r'(?:Văn\s*bản|Công\s*văn|CV)(?:\s*số\s*)?([\d\/\-]+[\w\-]*)'
    }
    
    sources = []
    for doc_type, pattern in patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            doc_id = match.group(1).strip()
            if doc_id:  # Avoid empty matches
                doc_name = {
                    'decree': 'Nghị định',
                    'decision': 'Quyết định',
                    'circular': 'Thông tư',
                    'document': 'Văn bản'
                }[doc_type]
                sources.append(f"{doc_name} {doc_id}")
    
    # Add date if available
    date_pattern = r'ngày\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})'
    date_match = re.search(date_pattern, text, re.IGNORECASE)
    if date_match and sources:
        sources[-1] += f" ngày {date_match.group(1)}"
    
    return ", ".join(sources) if sources else ""

def detect_categories(query: str) -> Set[str]:
    """Detect categories from query using improved keyword matching"""
    query_lower = query.lower()
    detected = {KEYWORD_TO_CATEGORY[kw] for kw in KEYWORD_TO_CATEGORY if kw in query_lower}
    
    # Add fallback category for common queries if no category is detected
    if not detected and any(term in query_lower for term in ["học bổng", "scholarship"]):
        detected.add("Scholarship")
        
    return detected

def build_category_filter(categories: Set[str]):
    """Build optimized ChromaDB filter from categories"""
    if not categories:
        return None
        
    if len(categories) == 1:
        category = next(iter(categories))
        return {"$or": [{"category": category}, {"subcategory": category}]}
    
    # For multiple categories (optimized query)
    return {"$or": [{"category": cat} for cat in categories] + [{"subcategory": cat} for cat in categories]}

def prefetch_embeddings(query: str) -> None:
    """Prefetch embeddings in a background thread"""
    global _embedding_cache
    
    query_hash = hash_query(query)
    if query_hash not in _embedding_cache:
        # This would normally use the OpenAI API to get embeddings
        # For demonstration, we're just setting a placeholder
        _embedding_cache[query_hash] = True
        print(f"Prefetched embeddings for: {query}")

def get_relevant_content(query: str, use_categories: bool = True, n_results: int = 3) -> List[Dict]:
    """Get relevant content with enhanced hybrid retrieval"""
    start = time.time()
    
    # Build filter based on detected categories
    filter_dict = None
    if use_categories:
        categories = detect_categories(query)
        print(f"Detected categories: {categories}")
        filter_dict = build_category_filter(categories)
    
    try:
        # Query with optimized parameters
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict,
            include=["metadatas", "documents"]  # Be explicit about what to include
        )
        
        # Process results more efficiently
        relevant_content = []
        if results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                relevant_content.append({
                    'question': metadata.get('question', ''),
                    'answer': metadata.get('answer', ''),
                    'category': metadata.get('category', ''),
                    'subcategory': metadata.get('subcategory', '')
                })
        
        print(f"Retrieval time: {time.time() - start:.4f}s, Results: {len(relevant_content)}")
        return relevant_content
    except Exception as e:
        print(f"Error during retrieval: {str(e)}")
        return []

def create_prompt(query: str, content: List[Dict]) -> str:
    """Create an optimized prompt with source citations"""
    # Format each content item with source info
    context_items = []
    for item in content:
        # Skip items with empty answers
        if not item.get('answer') or not item['answer'].strip():
            continue
            
        context_str = f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
        source_info = extract_source_info(item.get('answer', ''))
        if source_info:
            context_str += f"\nNguồn: {source_info}"
        context_items.append(context_str)
    
    # Handle the case when there are no valid content items
    if not context_items:
        context_items = ["Không có thông tin phù hợp."]
    
    # Use safer string formatting
    prompt = """Trả lời câu hỏi dựa trên thông tin sau:
    
    {0}
    
    CÂU HỎI: {1}
    
    Hãy trả lời câu hỏi một cách rõ ràng và có cấu trúc. Khi liệt kê các điều kiện hoặc tiêu chí, 
    hãy trình bày dưới dạng danh sách có đánh số rõ ràng, mỗi điều kiện trên một dòng mới.
    Nếu có thông tin về nguồn tài liệu (như số nghị định, quyết định, văn bản), 
    hãy trích dẫn nguồn đó ở cuối câu trả lời.
    
    TRẢ LỜI:""".format("\n\n".join(context_items), query)
    
    return prompt

def get_llm_response(prompt: str) -> str:
    """Get optimized response from language model"""
    start = time.time()
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": 
             "Bạn là trợ lý tư vấn thông tin chuyên nghiệp. Trả lời ngắn gọn, chính xác và đầy đủ thông tin. Trả lời chi tiết nếu được yêu cầu."
             "Hãy sử dụng định dạng rõ ràng với các điểm chính được trình bày dưới dạng danh sách có đánh số. "
             "Nếu câu trả lời đề cập đến nguồn tài liệu như nghị định, quyết định, hãy trích dẫn rõ ràng. "
             "Tự tin với thông tin mà bạn có, và thừa nhận khi không có đủ thông tin."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for more consistent responses
        max_tokens=400,   # Reduced tokens for faster responses
        presence_penalty=0.3
    )
    
    print(f"LLM response time: {time.time() - start:.4f}s")
    return response.choices[0].message.content

def find_similar_cached_query(query: str) -> Optional[str]:
    """Find semantically similar query in cache"""
    normalized = ' '.join(query.lower().split())
    
    # Exact match first
    if normalized in _response_cache:
        return normalized
        
    # Simple substring matching for similar queries
    # This is a basic implementation; a more sophisticated approach would use embeddings
    for cached_query in _response_cache:
        # Check if either query is a substring of the other with significant overlap
        if (normalized in cached_query and len(normalized) > 10 and len(normalized) / len(cached_query) > 0.7) or \
           (cached_query in normalized and len(cached_query) > 10 and len(cached_query) / len(normalized) > 0.7):
            return cached_query
            
    return None

def process_user_query(query: str) -> str:
    """Process user query with enhanced optimizations"""
    try:
        start = time.time()
        
        # Start prefetching embeddings in background for potential future use
        threading.Thread(target=prefetch_embeddings, args=(query,), daemon=True).start()
        
        # Check cache for similar queries (enhanced matching)
        similar_query = find_similar_cached_query(query)
        if similar_query:
            print(f"Cache hit! Similar query: '{similar_query}'")
            print(f"Response time: {time.time() - start:.2f}s")
            return _response_cache[similar_query]
        
        # Process query based on length/complexity
        n_results = 3  # Default
        if len(query.split()) <= 3:  # Very short queries
            n_results = 5  # Get more results for short queries
        
        # Hybrid search with categories, fallback to pure embedding if needed
        relevant_content = get_relevant_content(query, use_categories=True, n_results=n_results)
        if not relevant_content:
            print("Falling back to embedding-only search")
            relevant_content = get_relevant_content(query, use_categories=False, n_results=5)
        
        # Generate response
        prompt = create_prompt(query, relevant_content)
        response = get_llm_response(prompt)
        
        # Update cache with both exact and normalized queries
        normalized_query = ' '.join(query.lower().split())
        _response_cache[normalized_query] = response
        
        print(f"Total response time: {time.time() - start:.2f}s")
        return response
        
    except Exception as e:
        import traceback
        print(f"Error processing query: {str(e)}")
        print(traceback.format_exc())
        return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"

# Gradio interface with modern message format
def chatbot_interface(message, history):
    return process_user_query(message)

# Configure Gradio with proper parameters (fixed)
demo = gr.ChatInterface(
    fn=chatbot_interface,
    title="Trợ lý tư vấn Học bổng 🎓",
    description="Hãy đặt câu hỏi về học bổng, mình sẽ trả lời đầy đủ và nhanh chóng!",
    theme=gr.themes.Soft(),
    examples=[
        "Điều kiện để nhận học bổng là gì?",
        "Quy trình xét học bổng như thế nào?",
        "Thời gian xét học bổng là khi nào?",
        "Điều kiện để được xét học bổng khuyến khích học tập?"
    ]
)

if __name__ == "__main__":
    # Set Gradio queue parameters compatibly
    demo.queue(max_size=10)
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        share=True,
        show_error=True
    )