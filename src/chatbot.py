import os
import chromadb
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import time
import re
from typing import List, Dict, Set, Optional

# Initialize clients
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(
    name="scholarship-qa",
    embedding_function=chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name="text-embedding-3-small"
    )
)

# Simple cache and category mapping
_response_cache = {}
CATEGORIES = {
    "Scholarship": ["điều kiện", "yêu cầu", "tiêu chuẩn", "học bổng", "khuyến khích"],
    "Decree": ["quy trình", "thủ tục", "các bước", "nghị định", "nộp hồ sơ"],
    "Timeline": ["thời gian", "thời hạn", "khi nào", "deadline", "lịch trình"]
}
KEYWORD_TO_CATEGORY = {kw: cat for cat, keywords in CATEGORIES.items() for kw in keywords}

# Document source extraction patterns
SOURCE_PATTERNS = {
    r'(?:Nghị\s*định|NĐ)(?:\s*số\s*)?([\d\/\-]+[\w\-]*)': 'Nghị định',
    r'(?:Quyết\s*định|QĐ)(?:\s*số\s*)?([\d\/\-]+[\w\-]*)': 'Quyết định',
    r'(?:Thông\s*tư|TT)(?:\s*số\s*)?([\d\/\-]+[\w\-]*)': 'Thông tư',
    r'(?:Văn\s*bản|Công\s*văn|CV)(?:\s*số\s*)?([\d\/\-]+[\w\-]*)': 'Văn bản'
}

def extract_sources(text: str) -> str:
    """Extract document references from text"""
    sources = []
    for pattern, doc_type in SOURCE_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            doc_id = match.group(1).strip()
            if doc_id:
                sources.append(f"{doc_type} {doc_id}")
    
    # Add date if available
    date_match = re.search(r'ngày\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text, re.IGNORECASE)
    if date_match and sources:
        sources[-1] += f" ngày {date_match.group(1)}"
    
    return ", ".join(sources) if sources else ""

def detect_categories(query: str) -> Set[str]:
    """Identify query categories for faster retrieval"""
    query_lower = query.lower()
    
    # Direct keyword matching
    detected = {KEYWORD_TO_CATEGORY[kw] for kw in KEYWORD_TO_CATEGORY if kw in query_lower}
    
    # Add fallback category
    if not detected and any(term in query_lower for term in ["học bổng", "scholarship"]):
        detected.add("Scholarship")
        
    return detected

def get_relevant_content(query: str, use_categories: bool = True) -> List[Dict]:
    """Retrieve relevant content using hybrid search"""
    start = time.time()
    
    # Build category filter
    filter_dict = None
    if use_categories:
        categories = detect_categories(query)
        print(f"Categories: {categories}")
        
        if categories:
            if len(categories) == 1:
                cat = next(iter(categories))
                filter_dict = {"$or": [{"category": cat}, {"subcategory": cat}]}
            else:
                filter_dict = {"$or": [{"category": cat} for cat in categories] + 
                                      [{"subcategory": cat} for cat in categories]}
    
    # Query database
    try:
        results = collection.query(
            query_texts=[query],
            n_results=5 if len(query.split()) <= 3 else 3,  # More results for short queries
            where=filter_dict
        )
        
        # Process results
        content = []
        if results['metadatas'] and results['metadatas'][0]:
            for metadata in results['metadatas'][0]:
                content.append({k: metadata.get(k, '') for k in 
                               ['question', 'answer', 'category', 'subcategory']})
        
        print(f"Retrieval: {time.time() - start:.2f}s, Results: {len(content)}")
        return content
    except Exception as e:
        print(f"Retrieval error: {str(e)}")
        return []

def find_similar_query(query: str) -> Optional[str]:
    """Find cached similar query"""
    normalized = ' '.join(query.lower().split())
    
    # Exact match
    if normalized in _response_cache:
        return normalized
    
    # Substring match with significant overlap
    for cached in _response_cache:
        if ((normalized in cached and len(normalized) > 10 and len(normalized) / len(cached) > 0.7) or
            (cached in normalized and len(cached) > 10 and len(cached) / len(normalized) > 0.7)):
            return cached
            
    return None

def create_prompt(query: str, content: List[Dict]) -> str:
    """Create optimized prompt with source citations"""
    context_items = []
    
    for item in content:
        if not item.get('answer', '').strip():
            continue
            
        context = f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
        sources = extract_sources(item.get('answer', ''))
        if sources:
            context += f"\nNguồn: {sources}"
        context_items.append(context)
    
    if not context_items:
        context_items = ["Không có thông tin phù hợp."]
    
    return """Trả lời câu hỏi dựa trên thông tin sau:
    
    {0}
    
    CÂU HỎI: {1}
    
    Hãy trả lời ngắn gọn, đầy đủ dựa trên thông tin đã cung cấp. Trích dẫn nguồn tài liệu 
    (nghị định, quyết định, văn bản) nếu có. Nếu không đủ thông tin, hãy nói rõ.
    
    TRẢ LỜI:""".format("\n\n".join(context_items), query)

def get_response(prompt: str) -> str:
    """Get optimized LLM response"""
    start = time.time()
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": 
             "Bạn là trợ lý tư vấn học bổng chuyên nghiệp. Trả lời ngắn gọn, chính xác và đầy đủ. "
             "Trích dẫn nguồn tài liệu rõ ràng khi có thông tin. Tự tin với thông tin đã có."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=400,
        presence_penalty=0.3
    )
    
    print(f"LLM time: {time.time() - start:.2f}s")
    return response.choices[0].message.content

def process_query(query: str) -> str:
    """Main query processing pipeline"""
    try:
        start = time.time()
        
        # Check cache
        similar = find_similar_query(query)
        if similar:
            print(f"Cache hit: '{similar}'")
            return _response_cache[similar]
        
        # Retrieve content with categories, fallback if needed
        content = get_relevant_content(query, use_categories=True)
        if not content:
            content = get_relevant_content(query, use_categories=False)
        
        # Generate response
        prompt = create_prompt(query, content)
        response = get_response(prompt)
        
        # Update cache
        _response_cache[' '.join(query.lower().split())] = response
        
        print(f"Total time: {time.time() - start:.2f}s")
        return response
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"

# Gradio interface
demo = gr.ChatInterface(
    fn=lambda message, history: process_query(message),
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
    demo.queue(max_size=10)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)