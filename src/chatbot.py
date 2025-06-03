from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Set
from collections import Counter
from flask_cors import CORS
from pathlib import Path
from zoneinfo import ZoneInfo
import time
import os
import chromadb
import re
import math
import redis
import json
import uuid
import requests
import csv
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables and initialize clients
load_dotenv()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Đảm bảo thư mục logs tồn tại
Path(LOGS_DIR).mkdir(exist_ok=True)

# Tệp log cho câu hỏi và câu trả lời
QA_LOG_FILE = os.path.join(LOGS_DIR, "qa_log.csv")

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True, password='terminator')

# Simplified response cache with size limit
_response_cache = {}
MAX_CACHE_SIZE = 30

# Danh sách collection và mô tả chủ đề 
COLLECTIONS_INFO = [
    {"name": "scholarship_documents", "desc": "học bổng"},
    {"name": "ictu_intro_documents", "desc": "tầm nhìn, sứ mệnh, giá trị cốt lõi, giá trị giáo dục"},
    {"name": "training_and_regulations", "desc": "đào tạo, quy chế học tập"},
    {"name": "tuition_and_support", "desc": "học phí, hỗ trợ học tập"},
]
COLLECTION_NAME_TO_DESC = {c["name"]: c["desc"] for c in COLLECTIONS_INFO}
COLLECTION_DESC_TO_NAME = {c["desc"]: c["name"] for c in COLLECTIONS_INFO}

# Định nghĩa Local Embedding Function
class LocalEmbeddingFunction:
    def __init__(self, timeout=30):
        self.timeout = timeout
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        try:
            response = requests.post(
                "http://192.168.1.131:8000/embed", 
                json={"texts": input},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Kiểm tra định dạng đặc biệt: API có thể trả về dạng phẳng (768 số)
            # thay vì dạng mảng 2 chiều (mảng của các mảng 768 phần tử)
            if isinstance(result, list) and len(result) > 0:
                # Trường hợp đặc biệt: nếu số phần tử là bội số của 768 và tất cả đều là số
                # thì đây có thể là dạng phẳng của vector
                if (len(result) % 768 == 0 and 
                    all(isinstance(x, (int, float)) for x in result) and
                    len(input) == len(result) // 768):
                    
                    print(f"Phát hiện định dạng phẳng của vector! Chuyển đổi...")
                    # Chuyển đổi từ dạng phẳng sang dạng mảng 2 chiều
                    reshaped_result = []
                    for i in range(0, len(result), 768):
                        reshaped_result.append(result[i:i+768])
                    result = reshaped_result
                    print(f"Đã chuyển đổi thành {len(result)} vector, mỗi vector có {len(result[0]) if result else 0} chiều")
                
                # Kiểm tra từng embedding
                for i, embedding in enumerate(result):
                    # Nếu là float (lỗi), chuyển đổi thành list
                    if isinstance(embedding, (int, float)):
                        print(f"Lỗi: embedding thứ {i} là số, không phải list")
                        result[i] = [0.0] * 768
                    # Nếu là list rỗng, thay thế bằng vector 0
                    elif not embedding or len(embedding) == 0:
                        print(f"Lỗi: embedding thứ {i} là list rỗng")
                        result[i] = [0.0] * 768
            else:
                print(f"Lỗi định dạng embedding: Kết quả không phải danh sách hoặc rỗng")
                return [[0.0] * 768 for _ in input]  # Vector 768 chiều
            
            return result
        except requests.exceptions.RequestException as e:
            print(f"Lỗi khi gọi API embedding: {str(e)}")
            return [[0.0] * 768 for _ in input]  # Vector 768 chiều
        except Exception as e:
            print(f"Lỗi không xác định khi xử lý embedding: {str(e)}")
            return [[0.0] * 768 for _ in input]  # Vector 768 chiều

# Sử dụng Local Embedding Function
embedding_function = LocalEmbeddingFunction()

# Tạo file CSV log nếu chưa tồn tại
def init_qa_log_file():
    if not os.path.exists(QA_LOG_FILE):
        with open(QA_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Conversation_ID',
                'User_ID', 
                'Session_Name', 
                'Question', 
                'Answer', 
                'Timestamp',
                'Processing_Time',
                'Rating',
                'Feedback'
            ])
# Khởi tạo file log
init_qa_log_file()

# Hàm log câu hỏi và câu trả lời
def log_qa(user_id, question, answer, processing_time=None, sources=None, rating=None, comment=None):
    try:
        # Lấy tên phiên nếu có
        session_name = ""
        try:
            session_name_data = redis_client.get(f"session_name:{user_id}")
            if session_name_data:
                session_name = session_name_data
        except:
            pass
        
        # Lấy timestamp hiện tại
        # Lấy múi giờ Hà Nội
        tz = ZoneInfo("Asia/Ho_Chi_Minh")

        # Lấy thời gian hiện tại theo múi giờ
        now = datetime.datetime.now(tz)

        # Format thời gian để lưu
        timestamp = now.strftime("%d-%m-%Y %H:%M:%S")

        # Tạo conversation_id dùng cùng thời điểm
        conversation_id = f"{user_id}_{now.strftime('%d%m%Y%H%M%S')}"
        
        # Chuẩn bị dữ liệu để ghi
        row = [
            conversation_id,
            user_id,
            session_name,
            question,
            answer,
            timestamp,
            f"{processing_time:.2f}" if processing_time else "",
            rating or "",
            comment or ""
        ]
        
        # Ghi vào file CSV
        with open(QA_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        print(f"Logged Q&A for user {user_id}")
        return True
    except Exception as e:
        print(f"Error logging Q&A: {str(e)}")
        return False

# Hàm cập nhật đánh giá cho câu trả lời trước đó
def update_rating(user_id, question, answer, rating, comment=None):
    try:
        # Đọc file log hiện tại
        rows = []
        found = False
        
        # Chuẩn hóa dữ liệu đầu vào
        question_cleaned = question.strip() if question else ""
        answer_cleaned = answer.strip() if answer else ""
        
        # In thông tin để debug
        print(f"Looking for rating match with user_id={user_id}, question={question_cleaned[:30]}...")
        
        if os.path.exists(QA_LOG_FILE):
            with open(QA_LOG_FILE, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Lấy header
                
                for row in reader:
                    if len(row) < 9:  # Đảm bảo có đủ cột
                        rows.append(row)
                        continue
                    
                    # Lấy dữ liệu từ bản ghi
                    row_user_id = row[1].strip() if len(row) > 1 else ""
                    row_question = row[3].strip() if len(row) > 3 else ""
                    row_answer = row[4].strip() if len(row) > 4 else ""
                    
                    # Debug để xem các giá trị
                    if row_user_id == user_id and row_question == question_cleaned:
                        print(f"  Found user+question match. Checking answer...")
                        print(f"  Expected answer: {answer_cleaned[:30]}...")
                        print(f"  Found answer: {row_answer[:30]}...")
                    
                    # So sánh chính xác cả ba giá trị
                    if (row_user_id == user_id and 
                        row_question == question_cleaned and 
                        row_answer == answer_cleaned):
                        # Nếu tìm thấy, cập nhật đánh giá và bình luận
                        row[7] = rating  # Rating
                        row[8] = comment or ""  # Feedback
                        found = True
                        print(f"Found matching Q&A for rating update!")
                    
                    rows.append(row)
        
        if found:
            # Ghi lại toàn bộ file với dữ liệu đã cập nhật
            with open(QA_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
            print(f"Updated rating for Q&A from user {user_id}")
            return True
        else:
            # Nếu không tìm thấy, log một bản ghi mới với đánh giá
            print(f"No matching record found, creating new entry")
            log_qa(user_id, question_cleaned, answer_cleaned, None, None, rating, comment)
            return True
            
    except Exception as e:
        print(f"Error updating rating: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def get_relevant_content(query: str, final_results: int = 4) -> List[Dict]:
    """Get relevant content with hybrid retrieval and reranking"""
    start = time.time()
    
    try:
        # First stage: retrieve initial results
        initial_results = 10
        
        results = collection.query(
            query_texts=[query],
            n_results=initial_results,
            include=["metadatas", "documents"]
        )
        
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
             "Bạn là trợ lý tư vấn thông tin chuyên nghiệp."
             "Chỉ sử dụng thông tin đã được cung cấp để trả lời."
             "Nếu không có đủ thông tin, hãy thừa nhận một cách lịch sự và đề nghị người dùng hỏi lại."
             "Nếu có thông tin mâu thuẫn, ưu tiên nguồn mới nhất."
             "Trả lời đúng đầy đủ, trọng tâm câu hỏi với giọng điệu thân thiện và gần gũi."
             "Sử dụng định dạng dễ đọc, có thể trình bày dưới dạng danh sách nếu phù hợp."
             "Hãy ngắn gọn và chính xác, nhưng có thể cung cấp thêm chi tiết nếu người dùng yêu cầu."
             "Sử dụng ngôn ngữ gần gũi, phù hợp với văn hóa Việt Nam."
             "Không dùng các ký hiệu markdown(Ví dụ: #, *, **, _) để trả lời." 
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=600,
        presence_penalty=0.3
    )
    
    print(f"LLM response time: {time.time() - start:.4f}s")
    return response.choices[0].message.content

def normalize_query(query: str) -> str:
    """Chuẩn hóa câu hỏi: loại bỏ khoảng trắng thừa, chuyển thành chữ thường."""
    return ' '.join(query.lower().strip().split())

def get_cached_response(query: str) -> str | None:
    """Lấy phản hồi từ Redis cache."""
    try:
        normalized_query = normalize_query(query)
        cache_key = f"cache:response:{normalized_query}"
        cached_response = redis_client.get(cache_key)
        if cached_response:
            print(f"Cache hit for query: '{normalized_query}'")
            # Cập nhật TTL để gia hạn cache
            redis_client.expire(cache_key, 3600)  # TTL 1 giờ
            return cached_response
    except redis.exceptions.RedisError as e:
        print(f"Redis error when checking cache: {str(e)}")
    return None

def set_cached_response(query: str, response: str):
    """Lưu phản hồi vào Redis cache với TTL (chỉ exact match, không lưu embedding)."""
    try:
        normalized_query = normalize_query(query)
        cache_key = f"cache:response:{normalized_query}"
        redis_client.setex(cache_key, 1800, response)
        # Cập nhật index
        cache_index_key = "cache:index"
        current_time = time.time()
        redis_client.zadd(cache_index_key, {normalized_query: current_time})
        cache_count = redis_client.zcard(cache_index_key)
        if cache_count > MAX_CACHE_SIZE:
            old_keys = redis_client.zrange(cache_index_key, 0, cache_count - MAX_CACHE_SIZE)
            for old_query in old_keys:
                redis_client.delete(f"cache:response:{old_query}")
                redis_client.zrem(cache_index_key, old_query)
        print(f"Cached response for query: '{normalized_query}'")
    except redis.exceptions.RedisError as e:
        print(f"Redis error when setting cache: {str(e)}")

# Hàm dùng LLM để phân loại collection phù hợp cho câu hỏi

def classify_collections_llm(question, context=None):
    desc_list = "\n".join([f"- {c['desc']} ({c['name']})" for c in COLLECTIONS_INFO])
    prompt = (
        "Bạn là hệ thống phân loại chủ đề cho chatbot.\n"
        "Các collection hiện có:\n"
        f"{desc_list}\n"
        "Dựa vào câu hỏi (và ngữ cảnh hội thoại nếu có), hãy trả về tên collection phù hợp nhất.\n"
        "Nếu liên quan nhiều collection, hãy trả về danh sách tên collection, phân tách bằng dấu phẩy.\n"
        "Chỉ trả về tên collection, không giải thích.\n"
        "Câu hỏi: " + question
    )
    if context:
        prompt += f"\nNgữ cảnh hội thoại: {context}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Bạn là hệ thống phân loại collection cho chatbot."},
                 {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=32
    )
    raw = resp.choices[0].message.content.strip()
    # Chuẩn hóa kết quả: tách tên collection, loại bỏ thừa
    names = [x.strip() for x in raw.split(",") if x.strip() in COLLECTION_NAME_TO_DESC]
    return names

# Truy vấn nhiều collection, hợp nhất kết quả, rerank

def multi_collection_search(query, collection_names, final_results=4):
    all_results = []
    def query_collection(cname):
        try:
            col = chroma_client.get_collection(name=cname, embedding_function=embedding_function)
            results = col.query(query_texts=[query], n_results=7, include=["metadatas", "documents"])
            local_results = []
            if results['metadatas'] and results['metadatas'][0]:
                for i, metadata in enumerate(results['metadatas'][0]):
                    document_text = results['documents'][0][i] if results['documents'] and results['documents'][0] else ''
                    local_results.append({
                        'document': document_text,
                        'metadata': metadata,
                        'collection': cname,
                        'relevance_score': 0.0
                    })
            return local_results
        except Exception as e:
            print(f"Không thể truy vấn collection {cname}: {e}")
            return []
    # Truy vấn song song các collection
    with ThreadPoolExecutor(max_workers=min(8, len(collection_names))) as executor:
        future_to_cname = {executor.submit(query_collection, cname): cname for cname in collection_names}
        for future in as_completed(future_to_cname):
            result = future.result()
            if result:
                all_results.extend(result)
    # Rerank toàn bộ kết quả bằng BM25
    if all_results:
        documents = [item['document'] for item in all_results]
        bm25 = BM25(documents)
        bm25_scores = bm25.get_scores(query)
        for i, score in enumerate(bm25_scores):
            all_results[i]['relevance_score'] = score
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return all_results[:final_results]
    return []

# Sửa process_user_query để dùng LLM phân loại collection

def process_user_query(query: str, user_id: str) -> str:
    try:
        start = time.time()
        normalized_query = normalize_query(query)

        # Kiểm tra cache 
        cached_response = get_cached_response(query)
        if cached_response:
            processing_time = time.time() - start
            log_qa(user_id, query, cached_response, processing_time)
            print(f"Response time: {processing_time:.2f}s")
            return cached_response

        # Lấy session từ Redis
        session_key = f"session:{user_id}"
        session_data = redis_client.get(session_key)
        if session_data:
            session_data = json.loads(session_data)
        else:
            session_data = []
        # Lấy context hội thoại gần nhất
        context = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in session_data]) if session_data else ""

        # Dùng LLM phân loại collection
        collection_names = classify_collections_llm(query, context)
        print(f"LLM phân loại collection: {collection_names}")
        if not collection_names:
            # Fallback: tìm trên tất cả collection
            collection_names = [c['name'] for c in COLLECTIONS_INFO]
        # Truy vấn các collection được xác định
        relevant_content = multi_collection_search(query, collection_names, final_results=4)
        # Nếu không có kết quả, fallback tìm tất cả collection
        if not relevant_content:
            print("Không tìm thấy kết quả, fallback tìm trên tất cả collection")
            collection_names = [c['name'] for c in COLLECTIONS_INFO]
            relevant_content = multi_collection_search(query, collection_names, final_results=4)
        # Trích xuất nguồn tài liệu
        sources = []
        for item in relevant_content:
            metadata = item.get('metadata', {})
            if metadata.get('source'):
                sources.append(metadata['source'])
            elif metadata.get('doc_id'):
                sources.append(metadata['doc_id'])
        sources_str = ", ".join(set(sources))
        print(f"Sources: {sources_str}")
        base_prompt = create_prompt(query, relevant_content)
        prompt_with_context = f"Lịch sử trò chuyện:\n{context}\n\n{base_prompt}" if context else base_prompt
        response = get_llm_response(prompt_with_context)
        # Cập nhật session với TTL
        session_data.append({"question": query, "answer": response})
        redis_client.setex(session_key, 3600, json.dumps(session_data))
        # Lưu vào cache Redis
        set_cached_response(query, response)
        processing_time = time.time() - start
        log_qa(user_id, query, response, processing_time)
        print(f"Total response time: {processing_time:.2f}s\n")
        return response
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        error_msg = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
        log_qa(user_id, query, error_msg)
        return error_msg

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
CORS(app, resources={r"/*": {"origins": "*"}})

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
    
    # Kiểm tra cache Redis
    cached_response = get_cached_response(query)
    if cached_response:
        # Lưu lịch sử vào Redis ngay cả khi dùng cache
        session_key = f"session_history:{user_id}"
        session_data = redis_client.get(session_key)
        if session_data:
            session_data = json.loads(session_data)
        else:
            session_data = []
        session_data.append({'question': query, 'answer': cached_response})
        if len(session_data) > 5:
            session_data = session_data[-5:]
        redis_client.setex(session_key, 1800, json.dumps(session_data))
        print('Response from cache:', cached_response)
        return jsonify({'response': cached_response})
    
    # Nếu không có trong cache, xử lý truy vấn
    response = process_user_query(query, user_id)
    print('Response from process_user_query:', response)
    
    # Lưu lịch sử vào Redis
    session_key = f"session_history:{user_id}"
    session_data = redis_client.get(session_key)
    if session_data:
        session_data = json.loads(session_data)
    else:
        session_data = []
    session_data.append({'question': query, 'answer': response})
    if len(session_data) > 5:
        session_data = session_data[-5:]
    redis_client.setex(session_key, 3600, json.dumps(session_data))
    
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
        print(f"Received request for /get_session_name with user_id: {user_id}")
        
        if not user_id:
            print("Missing user_id in /get_session_name")
            return jsonify({'error': 'Missing user_id'}), 400
        
        print(f"Attempting to get session_name:{user_id} from Redis")
        
        try:
            # Test Redis connection first
            redis_client.ping()
            print("Redis connection successful")
            
            # Get session name with timeout
            name = redis_client.get(f"session_name:{user_id}")
            print(f"Redis response for session_name:{user_id}: {name}")
            
            if name:
                print(f"Found session name for {user_id}: {name}")
                return jsonify({'name': name})
                
            print(f"No session name found for {user_id}, returning user_id as name")
            return jsonify({'name': user_id})
            
        except redis.exceptions.TimeoutError as e:
            print(f"Redis timeout error in /get_session_name: {e}")
            # Return user_id as fallback
            return jsonify({'name': user_id, 'note': 'Using fallback due to Redis timeout'})
            
        except redis.exceptions.AuthenticationError as e:
            print(f"Redis authentication error in /get_session_name: {e}")
            return jsonify({'name': user_id, 'note': 'Using fallback due to Redis auth error'})
            
        except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error in /get_session_name: {e}")
            return jsonify({'name': user_id, 'note': 'Using fallback due to Redis connection error'})
            
    except Exception as e:
        print(f"Unexpected error in /get_session_name: {e}")
        # Always return a valid response even on error
        return jsonify({'name': user_id if user_id else 'unknown', 'error': str(e)})
          
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
    
# Endpoint để xử lý đánh giá từ người dùng
@app.route('/rate_response', methods=['POST', 'OPTIONS'])
def rate_response():
    try:
        # Xử lý OPTIONS request cho CORS
        if request.method == 'OPTIONS':
            response = jsonify({'status': 'success'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            return response
            
        data = request.json
        user_id = data.get('user_id')
        rating = data.get('rating')
        comment = data.get('comment', '')
        question = data.get('question', '')
        answer = data.get('answer', '')
        
        print(f"Received rating request: {rating} from user {user_id}")
        print(f"Question: {question[:30]}... Answer: {answer[:30]}...")
        
        if not user_id or not rating:
            return jsonify({'error': 'Missing required data'}), 400
            
        # Cập nhật đánh giá trong file log
        success = update_rating(user_id, question, answer, rating, comment)
        
        if success:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Failed to save rating'}), 500
            
    except Exception as e:
        print(f"Error in /rate_response: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
# Chạy server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1508)