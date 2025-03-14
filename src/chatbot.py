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
import numpy as np
from collections import Counter
import math

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

def get_relevant_content(query: str, use_categories: bool = True, final_results: int = 4) -> List[Dict]:
    """Get relevant content with enhanced two-stage retrieval and reranking"""
    start = time.time()
    
    # Build filter based on detected categories
    filter_dict = None
    if use_categories:
        categories = detect_categories(query)
        print(f"Detected categories: {categories}")
        filter_dict = build_category_filter(categories)
    
    try:
        # First stage: retrieve 10 initial results for reranking
        initial_results = 10  # Always retrieve 10 initial results
        results = collection.query(
            query_texts=[query],
            n_results=initial_results,
            where=filter_dict,
            include=["metadatas", "documents"]
        )
        
        # Process results and prepare for reranking
        candidate_content = []
        if results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                document_text = results['documents'][0][i] if results['documents'] and results['documents'][0] else ''
                candidate_content.append({
                    'question': metadata.get('question', ''),
                    'answer': metadata.get('answer', ''),
                    'category': metadata.get('category', ''),
                    'subcategory': metadata.get('subcategory', ''),
                    'version': metadata.get('version', '1.0'),
                    'source': metadata.get('source', ''),
                    'document': document_text,
                    'relevance_score': 0.0  # Will be updated during reranking
                })
        
        # Second stage: Apply BM25 reranking
        if candidate_content:
            print(f"Reranking {len(candidate_content)} initial results...")
            
            # Extract documents for BM25
            documents = [
                f"{item['question']} {item['answer']}" 
                for item in candidate_content
            ]
            
            # Initialize BM25 with the retrieved documents
            bm25 = BM25(documents)
            
            # Get BM25 scores
            bm25_scores = bm25.get_scores(query)
            
            # Add BM25 scores to the content
            for i, score in enumerate(bm25_scores):
                candidate_content[i]['relevance_score'] = score
            
            # Rerank based on BM25 scores
            candidate_content.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Limit to the requested number of final results
            final_content = candidate_content[:final_results]
            
            # Group related information by topics
            grouped_content = group_related_information(final_content)
            
            print(f"Retrieval time: {time.time() - start:.4f}s, Final results: {len(final_content)}")
            return grouped_content
        else:
            print("No initial results found.")
            return []
    except Exception as e:
        print(f"Error during retrieval: {str(e)}")
        return []

def group_related_information(content_items: List[Dict]) -> List[Dict]:
    """Group related information to ensure coherent answers"""
    # If there are only a few items, no need to group
    if len(content_items) <= 2:
        return content_items
    
    # Simple topic grouping based on shared keywords in questions
    grouped = []
    processed = set()
    
    for i, item in enumerate(content_items):
        if i in processed:
            continue
            
        # Extract main keywords from the question
        question_keywords = set(re.findall(r'\b\w{4,}\b', item['question'].lower()))
        if not question_keywords:
            grouped.append(item)
            processed.add(i)
            continue
        
        # Find related items
        related_items = [item]
        processed.add(i)
        
        for j, other_item in enumerate(content_items):
            if j == i or j in processed:
                continue
                
            other_keywords = set(re.findall(r'\b\w{4,}\b', other_item['question'].lower()))
            # If significant keyword overlap or one is subset of the other
            if (question_keywords & other_keywords) and (
                len(question_keywords & other_keywords) / len(question_keywords) > 0.3 or
                len(question_keywords & other_keywords) / len(other_keywords) > 0.3
            ):
                related_items.append(other_item)
                processed.add(j)
        
        if len(related_items) > 1:
            # Create a merged item for the group
            merged_item = {
                'question': related_items[0]['question'],
                'answer': "\n\n".join([f"{item['answer']}" for item in related_items]),
                'category': related_items[0]['category'],
                'subcategory': related_items[0]['subcategory'],
                'source': ", ".join(set(item['source'] for item in related_items if item['source'])),
                'is_merged': True,
                'merged_count': len(related_items)
            }
            grouped.append(merged_item)
        else:
            grouped.append(item)
    
    return grouped

def create_prompt(query: str, content: List[Dict]) -> str:
    """Create an optimized prompt with synthesized information from multiple sources"""
    # Format each content item with source info
    context_items = []
    
    for item in content:
        # Skip items with empty answers
        if not item.get('answer') or not item['answer'].strip():
            continue
        
        # Special handling for merged items
        if item.get('is_merged'):
            context_str = f"Thông tin tổng hợp (từ {item.get('merged_count', 0)} nguồn):\n{item['answer']}"
        else:
            context_str = f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
        
        # Extract detailed source information if available
        source_info = extract_source_info(item.get('answer', ''))
        
        # Use the file name as the source if no detailed source info is available
        if not source_info:
            source_info = item.get('source', '').split('/')[-1]  # Extract file name from path
        
        if source_info:
            context_str += f"\nNguồn: {source_info}"
        
        context_items.append(context_str)
    
    # Handle the case when there are no valid content items
    if not context_items:
        context_items = ["Không có thông tin phù hợp."]
    
    # Combine all context items into a single prompt with clear section markers
    combined_context = "\n\n---\n\n".join(context_items)
    
    # Enhanced prompt that explicitly instructs how to handle multiple sources
    prompt = """Trả lời câu hỏi dựa trên thông tin sau:
    
    {0}
    
    CÂU HỎI: {1}
    
    Hướng dẫn:
    1. Tổng hợp thông tin từ tất cả các nguồn liên quan được cung cấp.
    2. Nếu có thông tin mâu thuẫn, ưu tiên thông tin từ nguồn mới nhất hoặc nguồn có thẩm quyền cao hơn.
    3. Bỏ qua những thông tin không liên quan đến câu hỏi.
    4. Trả lời một cách rõ ràng và có cấu trúc, dễ hiểu cho người đọc.
    5. Khi liệt kê các điều kiện, tiêu chí, hoặc các bước, hãy trình bày dưới dạng danh sách có đánh số rõ ràng.
    6. Trích dẫn nguồn tài liệu cụ thể (nghị định, quyết định, văn bản) ở cuối câu trả lời.
    
    TRẢ LỜI:""".format(combined_context, query)
    
    return prompt

def get_llm_response(prompt: str) -> str:
    """Get optimized response from language model"""
    start = time.time()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
        max_tokens=500,   # Reduced tokens for faster responses
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
        n_results = 4  # Default
        if len(query.split()) <= 4:  # Very short queries
            n_results = 5  # Get more results for short queries
        
        # Hybrid search with categories, fallback to pure embedding if needed
        relevant_content = get_relevant_content(query, use_categories=True, final_results=n_results)
        if not relevant_content:
            print("Falling back to embedding-only search")
            relevant_content = get_relevant_content(query, use_categories=False, final_results=5)
        
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
        "Sinh viên tại Trường Đại học Công nghệ Thông tin và Truyền thông có thể phản ánh, thắc mắc về danh sách học bổng khuyến khích học tập trong thời gian bao lâu và theo cách nào?",
        "Điều kiện để được xét học bổng khuyến khích học tập?"
    ]
)

# Add BM25 implementation
class BM25:
    """BM25 scoring algorithm for reranking search results"""
    
    def __init__(self, documents, k1=1.5, b=0.75):
        """Initialize BM25 with documents and parameters"""
        self.k1 = k1
        self.b = b
        
        # Tokenize documents
        self.tokenized_docs = [self.tokenize(doc) for doc in documents]
        
        # Calculate document lengths and average document length
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Calculate term frequencies and document frequencies
        self.term_frequencies = []
        for doc in self.tokenized_docs:
            self.term_frequencies.append(Counter(doc))
        
        self.doc_freqs = Counter()
        for doc in self.tokenized_docs:
            for term in set(doc):
                self.doc_freqs[term] += 1
        
        # Calculate IDF values
        self.idfs = {term: self._idf(term) for term in self.doc_freqs}
    
    def tokenize(self, text):
        """Simple tokenization function (can be improved with Vietnamese-specific tokenizers)"""
        # Remove special characters and lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Split by whitespace
        return text.split()
    
    def _idf(self, term):
        """Calculate IDF for a term"""
        # Add 1 to avoid division by zero
        return math.log((len(self.tokenized_docs) - self.doc_freqs[term] + 0.5) / 
                        (self.doc_freqs[term] + 0.5) + 1.0)
    
    def get_scores(self, query):
        """Calculate BM25 scores for a query across all documents"""
        query_terms = self.tokenize(query)
        scores = [0.0] * len(self.tokenized_docs)
        
        for term in query_terms:
            if term not in self.idfs:
                continue
                
            for i, doc in enumerate(self.tokenized_docs):
                if term not in self.term_frequencies[i]:
                    continue
                    
                # Calculate BM25 score for this term-document pair
                freq = self.term_frequencies[i][term]
                numerator = self.idfs[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * self.doc_lengths[i] / self.avgdl)
                scores[i] += numerator / denominator
        
        return scores

if __name__ == "__main__":
    # Set Gradio queue parameters compatibly
    demo.queue(max_size=10)
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        share=True,
        show_error=True
    )