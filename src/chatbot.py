import os
import chromadb
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Set
import time
import re
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

# Simplified response cache with size limit
_response_cache = {}
MAX_CACHE_SIZE = 30

# Category keyword mapping with inverse index for fast lookup
CATEGORY_KEYWORDS = {
    "Scholarship": ["điều kiện", "yêu cầu", "tiêu chuẩn", "tiêu chí", "đủ điều kiện", "đáp ứng", "học bổng", "khuyến khích"],
    "Decree": ["quy trình", "thủ tục", "các bước", "quá trình", "thực hiện", "nghị định", "nộp hồ sơ"],
    "Timeline": ["thời gian", "thời hạn", "khi nào", "hạn cuối", "deadline", "lịch trình"]
}
KEYWORD_TO_CATEGORY = {kw: cat for cat, keywords in CATEGORY_KEYWORDS.items() for kw in keywords}

def detect_categories(query: str) -> Set[str]:
    """Detect categories from query using keyword matching with improved fallback"""
    query_lower = query.lower()
    detected = {KEYWORD_TO_CATEGORY[kw] for kw in KEYWORD_TO_CATEGORY if kw in query_lower}
    
    # More flexible detection - if no categories detected, we'll use pure vector search
    return detected

def build_category_filter(categories: Set[str]):
    """Build ChromaDB filter from categories"""
    if not categories:
        return None
        
    if len(categories) == 1:
        category = next(iter(categories))
        return {"$or": [{"category": category}, {"subcategory": category}]}
    
    return {"$or": [{"category": cat} for cat in categories] + [{"subcategory": cat} for cat in categories]}

def get_relevant_content(query: str, use_categories: bool = False, final_results: int = 4) -> List[Dict]:
    """Get relevant content with hybrid retrieval and reranking"""
    start = time.time()
    
    # Build filter based on detected categories
    filter_dict = None
    categories = set()
    
    if use_categories:
        categories = detect_categories(query)
        if categories:
            print(f"Detected categories: {categories}")
            filter_dict = build_category_filter(categories)
    
    try:
        # First stage: retrieve initial results
        initial_results = 10
        
        # If no categories detected, use pure vector search with more results
        if not categories and use_categories:
            print("No categories detected, using pure vector similarity search")
        
        results = collection.query(
            query_texts=[query],
            n_results=initial_results,
            where=filter_dict,
            include=["metadatas", "documents"]
        )
        
        # Process results
        candidate_content = []
        if results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                document_text = results['documents'][0][i] if results['documents'] and results['documents'][0] else ''
                candidate_content.append({
                    'question': metadata.get('question', ''),
                    'answer': metadata.get('answer', ''),
                    'category': metadata.get('category', ''),
                    'subcategory': metadata.get('subcategory', ''),
                    'source': metadata.get('source', ''),
                    'document': document_text,
                    'relevance_score': 0.0
                })
        
        # Apply BM25 reranking if we have candidates
        if candidate_content:
            print(f"Reranking {len(candidate_content)} initial results...")
            
            # Extract documents for BM25
            documents = [f"{item['question']} {item['answer']}" for item in candidate_content]
            
            # Initialize BM25 with the retrieved documents
            bm25 = BM25(documents)
            
            # Get BM25 scores
            bm25_scores = bm25.get_scores(query)
            
            # Add BM25 scores and sort
            for i, score in enumerate(bm25_scores):
                candidate_content[i]['relevance_score'] = score
            
            # Rerank based on BM25 scores
            candidate_content.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Limit to final results
            final_content = candidate_content[:final_results]
            
            print(f"Retrieval time: {time.time() - start:.4f}s, Final results: {len(final_content)}")
            return final_content
        else:
            print("No initial results found.")
            return []
    except Exception as e:
        print(f"Error during retrieval: {str(e)}")
        return []

def extract_source_info(text: str) -> str:
    """Extract document references from text with simplified pattern"""
    pattern = r'(?:Nghị định|Quyết định|Thông tư|Văn bản|NĐ|QĐ|TT|CV)(?:\s*số\s*)?([\d\/\-]+[\w\-]*)'
    matches = set(re.findall(pattern, text, re.IGNORECASE))
    return ", ".join(f"Văn bản {m}" for m in matches) if matches else ""

def create_prompt(query: str, content: List[Dict]) -> str:
    """Create optimized prompt from retrieved content"""
    context_items = []
    
    for item in content:
        if not item.get('answer') or not item['answer'].strip():
            continue
        
        context_str = f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
        source_info = extract_source_info(item.get('answer', ''))

        if not source_info and item.get('source'):
            source_info = item.get('source', '').split('/')[-1]
            
        if source_info:
            context_str += f"\nNguồn: {source_info}"
        
        context_items.append(context_str)
    
    if not context_items:
        context_items = ["Không có thông tin phù hợp."]
    
    combined_context = "\n\n---\n\n".join(context_items)
    prompt = """Trả lời câu hỏi dựa trên thông tin sau:
    
    {0}
    
    CÂU HỎI: {1}
    
    Hướng dẫn:
    1. Bỏ qua những thông tin không liên quan đến câu hỏi.
    2. Tổng hợp các điểm chung giữa các nguồn thông tin để đưa ra câu trả lời tổng hợp, đáp ứng với yêu cầu của câu hỏi.
    3. Nếu có thông tin mâu thuẫn, ưu tiên thông tin từ nguồn mới nhất.
    4. Trả lời một cách rõ ràng và có cấu trúc.
    5. Khi liệt kê các điều kiện hoặc bước, trình bày dưới dạng danh sách có đánh số.
    6. Trích dẫn nguồn tài liệu cụ thể ở cuối câu trả lời.
    
    TRẢ LỜI:""".format(combined_context, query)
    
    return prompt

def get_llm_response(prompt: str) -> str:
    """Get response from language model"""
    start = time.time()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             "Bạn là trợ lý tư vấn thông tin chuyên nghiệp. Trả lời ngắn gọn, chính xác và đầy đủ. "
             "Sử dụng định dạng rõ ràng với các điểm chính được trình bày dưới dạng danh sách. "
             "Trích dẫn nguồn tài liệu rõ ràng. Thừa nhận khi không có đủ thông tin."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=600,
        presence_penalty=0.3
    )
    
    print(f"LLM response time: {time.time() - start:.4f}s")
    return response.choices[0].message.content

def process_user_query(query: str) -> str:
    try:
        start = time.time()
        
        # Simple cache check (exact match only)
        normalized_query = ' '.join(query.lower().split())
        if normalized_query in _response_cache:
            print(f"Cache hit for query: '{normalized_query}'")
            print(f"Response time: {time.time() - start:.2f}s")
            return _response_cache[normalized_query]
        
        # Adjust number of results based on query complexity
        n_results = 4 if len(query.split()) > 6 else 5
        
        # First try with category filtering
        relevant_content = get_relevant_content(query, use_categories=False, final_results=n_results)
        
        # If no results or very few results with categories, fall back to pure vector search
        if len(relevant_content) < 2:
            print("Insufficient results with categories, falling back to pure vector search")
            relevant_content = get_relevant_content(query, use_categories=False, final_results=5)
        
        # Generate response
        prompt = create_prompt(query, relevant_content)
        response = get_llm_response(prompt)
        
        # Update cache (limit size)
        if len(_response_cache) >= MAX_CACHE_SIZE:
            _response_cache.pop(next(iter(_response_cache)))
        _response_cache[normalized_query] = response
        
        print(f"Total response time: {time.time() - start:.2f}s\n")
        return response
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"

class BM25:
    """BM25 scoring algorithm for reranking search results"""
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [self.tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.term_frequencies = [Counter(doc) for doc in self.tokenized_docs]
        self.doc_freqs = Counter()
        for doc in self.tokenized_docs:
            for term in set(doc):
                self.doc_freqs[term] += 1
        self.idfs = {term: self._idf(term) for term in self.doc_freqs}
    
    def tokenize(self, text):
        """Simple tokenization function"""
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def _idf(self, term):
        """Calculate IDF for a term"""
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
                freq = self.term_frequencies[i][term]
                numerator = self.idfs[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * self.doc_lengths[i] / self.avgdl)
                scores[i] += numerator / denominator
        return scores

# Gradio interface
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
        "Sinh viên tại Trường Đại học Công nghệ Thông tin và Truyền thông có thể phản ánh, thắc mắc về danh sách học bổng khuyến khích học tập trong thời gian bao lâu và theo cách nào?",
        "Điều kiện để được xét học bổng khuyến khích học tập?"
    ]
)

if __name__ == "__main__":
    demo.queue(max_size=10)
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        share=True,
        show_error=True
    )