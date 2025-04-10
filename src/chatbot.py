from flask import Flask, request, jsonify
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Set
import time
import re
from collections import Counter
import math
from flask_cors import CORS
import redis
import json
import uuid

# Load environment variables and initialize clients
load_dotenv()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True, password='terminator')
embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name="text-embedding-3-small"
)

try:
    collection = chroma_client.get_collection(name="scholarship-qa", embedding_function=embedding_function)
except chromadb.errors.InvalidCollectionException:
    print("Collection 'scholarship-qa' does not exist. Creating new collection...")
    collection = chroma_client.create_collection(name="scholarship-qa", embedding_function=embedding_function)
    collection.add(
        documents=["Đây là tài liệu mẫu về học bổng ICTU"],
        metadatas=[{"source": "ictu.edu.vn"}],
        ids=["doc1"]
    )
    print("Collection 'scholarship-qa' created successfully.")

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
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=initial_results,
                where=filter_dict,
                include=["metadatas", "documents"]
            )
        except Exception as e:
            print(f"Lỗi khi truy vấn collection: {str(e)}")
            return []
        
        # Process results
        candidate_content = []
        if results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                document_text = results['documents'][0][i] if results['documents'] and results['documents'][0] else ''
                candidate_content.append({
                    'document': document_text,
                    'metadata': metadata,
                    'relevance_score': 0.0
                })
        
        # Apply BM25 reranking if we have candidates
        if candidate_content:
            print(f"Reranking {len(candidate_content)} initial results...")
            
            # Extract documents for BM25
            documents = [item['document'] for item in candidate_content]
            
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
        if not item.get('document') or not item['document'].strip():
            continue
        
        context_str = item['document']
        metadata = item.get('metadata', {})
        
        # Build source information from markdown metadata
        source_parts = []
        if metadata.get('source'):
            source_parts.append(metadata['source'])
        if metadata.get('doc_id'):
            source_parts.append(metadata['doc_id'])
        
            
        source_info = " tài liệu số ".join(source_parts)
        if source_info:
            context_str += f"\nNguồn: {source_info}"
        
        context_items.append(context_str)
    
    if not context_items:
        context_items = ["Không có thông tin phù hợp."]
    
    combined_context = "\n\n---\n\n".join(context_items)
    prompt = """Trả lời câu hỏi dựa trên thông tin sau:
    {0}
    CÂU HỎI: {1}
    TRẢ LỜI:""".format(combined_context, query)
    
    return prompt

def get_llm_response(prompt: str) -> str:
    """Get response from language model"""
    start = time.time()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             "Bạn là trợ lý tư vấn thông tin chuyên nghiệp. "
             "Chỉ sử dụng thông tin được cung cấp làm tri thức để trả lời"
             "Trả lời đúng trọng tâm câu hỏi. Hãy ngắn gọn và chính xác, có thể thay đổi theo yêu cầu câu hỏi nếu có "
             "Không đưa ra các thông tin câu hỏi không yêu cầu. "
             "Sử dụng định dạng rõ ràng với các điểm chính dưới dạng danh sách. Loại bỏ các ký hiệu markdown. "
             "Nếu có thông tin mâu thuẫn, ưu tiên nguồn mới nhất. "
             "Thừa nhận khi không có đủ thông tin."             
             "Trích dẫn nguồn cụ thể (điều, khoản, số văn bản nếu có). "
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=600,
        presence_penalty=0.3
    )
    
    print(f"LLM response time: {time.time() - start:.4f}s")
    return response.choices[0].message.content

def process_user_query(query: str, user_id: str) -> str:
    try:
        start = time.time()
        normalized_query = ' '.join(query.lower().split())
        
        if normalized_query in _response_cache:
            print(f"Cache hit for query: '{normalized_query}'")
            print(f"Response time: {time.time() - start:.2f}s")
            return _response_cache[normalized_query]
        
        # Lấy session từ Redis
        session_key = f"session:{user_id}"
        session_data = redis_client.get(session_key)
        if session_data:
            session_data = json.loads(session_data)
        else:
            session_data = []
        
        # Giới hạn session: chỉ lưu 5 tương tác cuối
        session_data = session_data[-5:]
        context = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" 
                           for item in session_data]) if session_data else ""
        
        # Tạo prompt với ngữ cảnh
        n_results = 4 if len(query.split()) > 6 else 5
        relevant_content = get_relevant_content(query, use_categories=False, final_results=n_results)
        if len(relevant_content) < 2:
            print("Insufficient results with categories, falling back to pure vector search")
            relevant_content = get_relevant_content(query, use_categories=False, final_results=5)
        
        base_prompt = create_prompt(query, relevant_content)
        prompt_with_context = f"Lịch sử trò chuyện:\n{context}\n\n{base_prompt}" if context else base_prompt
        
        response = get_llm_response(prompt_with_context)
        
        # Cập nhật session với TTL (hết hạn sau 1 giờ)
        session_data.append({"question": query, "answer": response})
        redis_client.setex(session_key, 900, json.dumps(session_data))  # TTL = 900 giây (15 phút)
        
        # Cập nhật cache
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

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

@app.route('/get_user_id', methods=['GET'])
def get_user_id():
    user_id = f"user_{uuid.uuid4()}"  # Tạo user_id duy nhất
    return jsonify({'user_id': user_id})

# Tạo endpoint /ask để nhận câu hỏi và trả lời
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    print('Received data:', data)
    query = data.get('query', '')
    user_id = data.get('user_id', '')
    if not query or not user_id:
        print('Error: Missing query or user_id', {'query': query, 'user_id': user_id})
        return jsonify({'error': 'No query or user_id provided'}), 400
    
    # Kiểm tra cache
    if query in _response_cache:
        response = _response_cache[query]
        print('Response from cache:', response)
    else:
        # Nếu không có trong cache, xử lý và lưu vào cache
        response = process_user_query(query, user_id)
        _response_cache[query] = response
        print('Response from process_user_query:', response)
    
    # Luôn lưu lịch sử vào Redis, bất kể phản hồi từ cache hay không
    session_key = f"session_history:{user_id}"  # Sửa key ở đây
    session_data = redis_client.get(session_key)
    if session_data:
        session_data = json.loads(session_data)
    else:
        session_data = []
    
    # Thêm câu hỏi và phản hồi vào lịch sử
    session_data.append({'question': query, 'answer': response})
    # Giới hạn số lượng tương tác (nếu cần, ví dụ: 5 tương tác tối đa)
    if len(session_data) > 5:
        session_data = session_data[-5:]
    redis_client.setex(session_key, 3600, json.dumps(session_data))  # TTL 1 giờ
    
    print('Updated session history:', session_data)
    return jsonify({'response': response})

@app.route('/get_session_history', methods=['GET'])
def get_session_history():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        history = redis_client.get(f"session_history:{user_id}")
        if history:
            return jsonify({'history': json.loads(history)})
        return jsonify({'history': []})
    except Exception as e:
        app.logger.error(f"Error in /get_session_history: {e}")
        return jsonify({'error': 'Internal server error'}), 500
    
@app.route('/set_session_name', methods=['POST'])
def set_session_name():
    data = request.json
    user_id = data.get('user_id')
    name = data.get('name')
    if not user_id or not name:
        return jsonify({'error': 'Missing user_id or name'}), 400
    redis_client.set(f"session_name:{user_id}", name)
    return jsonify({'status': 'success'})

@app.route('/get_session_name', methods=['GET'])
def get_session_name():
    try:
        user_id = request.args.get('user_id')
        app.logger.info(f"Received request for /get_session_name with user_id: {user_id}")
        if not user_id:
            app.logger.error("Missing user_id in /get_session_name")
            return jsonify({'error': 'Missing user_id'}), 400
        app.logger.info(f"Attempting to get session_name:{user_id} from Redis")
        name = redis_client.get(f"session_name:{user_id}")
        app.logger.info(f"Redis response for session_name:{user_id}: {name}")
        if name:
            app.logger.info(f"Found session name for {user_id}: {name}")
            return jsonify({'name': name})  # Sửa ở đây
        app.logger.info(f"No session name found for {user_id}, returning user_id as name")
        return jsonify({'name': user_id})
    except redis.exceptions.AuthenticationError as e:
        app.logger.error(f"Redis authentication error in /get_session_name: {e}")
        return jsonify({'error': 'Redis authentication failed'}), 500
    except redis.exceptions.ConnectionError as e:
        app.logger.error(f"Redis connection error in /get_session_name: {e}")
        return jsonify({'error': 'Redis connection failed'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error in /get_session_name: {e}")
        return jsonify({'error': 'Internal server error'}), 500
          
@app.route('/clear_session', methods=['POST'])
def clear_session():
    try:
        data = request.json
        user_id = data.get('user_id')
        app.logger.info(f"Received request for /clear_session with user_id: {user_id}")
        if not user_id:
            app.logger.error("Missing user_id in /clear_session")
            return jsonify({'error': 'Missing user_id'}), 400
        # Xóa dữ liệu phiên trong Redis
        redis_client.delete(f"session_history:{user_id}")  # Sửa key ở đây
        redis_client.delete(f"session_name:{user_id}")
        app.logger.info(f"Cleared session for user_id: {user_id}")
        return jsonify({'status': 'success'})
    except redis.exceptions.AuthenticationError as e:
        app.logger.error(f"Redis authentication error in /clear_session: {e}")
        return jsonify({'error': 'Redis authentication failed'}), 500
    except redis.exceptions.ConnectionError as e:
        app.logger.error(f"Redis connection error in /clear_session: {e}")
        return jsonify({'error': 'Redis connection failed'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error in /clear_session: {e}")
        return jsonify({'error': 'Internal server error'}), 500
    
# Chạy server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1508)