from flask import Flask, request, jsonify, Response, stream_with_context
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Set, Generator
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

# Khởi tạo collection duy nhất sau khi đã có embedding_function
collection = chroma_client.get_collection(name="all_documents", embedding_function=embedding_function)

# Simplified response cache with size limit
_response_cache = {}
MAX_CACHE_SIZE = 30

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

def get_relevant_content(query: str, categories: List[str] = None, final_results: int = 4) -> List[Dict]:
    """Get relevant content with category filtering FIRST"""
    start = time.time()
    try:
        # LOGIC MỚI: Filter category TRƯỚC khi semantic search
        if categories:
            # Tạo where clause cho ChromaDB
            where_clause = {"category": {"$in": categories}}
            
            # Tăng số lượng kết quả khi search trong category cụ thể
            initial_results = 20  # Tăng từ 10 lên 20
            
            print(f"Searching within categories: {categories}")
            results = collection.query(
                query_texts=[query],
                n_results=initial_results,
                where=where_clause,  # Filter TRỰC TIẾP trong ChromaDB
                include=["metadatas", "documents"]
            )
        else:
            # Không có category filter, search toàn bộ
            initial_results = 10
            results = collection.query(
                query_texts=[query],
                n_results=initial_results,
                include=["metadatas", "documents"]
            )
        
        # Prepare candidates (không cần filter thêm vì đã filter trong ChromaDB)
        candidate_content = []
        if results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                document_text = results['documents'][0][i] if results['documents'] and results['documents'][0] else ''
                candidate_content.append({
                    'document': document_text,
                    'metadata': metadata,
                    'relevance_score': 0.0
                })
        
        # BM25 rerank
        if candidate_content:
            documents = [item['document'] for item in candidate_content]
            bm25 = BM25(documents)
            bm25_scores = bm25.get_scores(query)
            
            for i, score in enumerate(bm25_scores):
                candidate_content[i]['relevance_score'] = score
                
            candidate_content.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            final_content = candidate_content[:final_results]
            
            print(f"Retrieval time: {time.time() - start:.4f}s, Final results: {len(final_content)}")
            return final_content
        else:
            print(f"Retrieval time: {time.time() - start:.4f}s, Final results: 0")
            return []
            
    except Exception as e:
        print(f"Error in get_relevant_content: {str(e)}")
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

def get_llm_response(prompt: str, stream: bool = False) -> str | Generator:
    """Get response from language model with optional streaming support"""
    start = time.time()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             "Bạn là trợ lý tư vấn thông tin chuyên nghiệp của trường ICTU."
             "Không dùng các ký hiệu (#, *, **, _) trong phản hồi." 
             "Chỉ sử dụng thông tin đã được cung cấp để trả lời."
             "Nếu không có đủ thông tin, hãy thừa nhận một cách lịch sự và đề nghị người dùng hỏi lại."
             "Nếu có thông tin mâu thuẫn, ưu tiên nguồn mới nhất."
             "Chú ý trả lời câu hỏi người dùng cho phù hợp với mạch hội thoại"
             "Trả lời đúng đầy đủ, trọng tâm câu hỏi với giọng điệu thân thiện và gần gũi."
             "Sử dụng định dạng dễ đọc, có thể trình bày dưới dạng danh sách nếu phù hợp."
             "Hãy ngắn gọn và chính xác, nhưng có thể cung cấp thêm chi tiết nếu người dùng yêu cầu."
             "Sử dụng ngôn ngữ gần gũi, phù hợp với văn hóa Việt Nam."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=600,
        presence_penalty=0.3,
        stream=stream
    )
    
    if not stream:
        print(f"LLM response time: {time.time() - start:.4f}s")
        return response.choices[0].message.content
    
    def generate():
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield f"data: {json.dumps({'content': content})}\n\n"
        yield f"event: done\ndata: {json.dumps({'full_response': full_response})}\n\n"
        return full_response
    
    return generate()

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
            redis_client.expire(cache_key, 900)  # TTL 1 giờ
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

# Sửa process_user_query để dùng LLM phân loại collection

def process_user_query(query: str, user_id: str, categories: List[str] = None) -> str:
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

        # Nếu categories chỉ là ['small_talk'] thì bỏ qua vector search
        if categories == ["small_talk"]:
            print("Detected small talk category. Skipping vector search.")
            session_key = f"session_history:{user_id}"
            session_data = redis_client.get(session_key)
            if session_data:
                session_data = json.loads(session_data)
                session_data = session_data[-1:]
            else:
                session_data = []
            context = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in session_data]) if session_data else ""
            base_prompt = f"Lịch sử trò chuyện:\n{context}\n\nCÂU HỎI: {query}\nTRẢ LỜI:" if context else f"CÂU HỎI: {query}\nTRẢ LỜI:"
            
            response = get_llm_response(base_prompt)

            session_data.append({"question": query, "answer": response})
            redis_client.setex(session_key, 900, json.dumps(session_data))
            set_cached_response(query, response) # lưu cache
            processing_time = time.time() - start
            log_qa(user_id, query, response, processing_time)
            print(f"Total response time (small talk): {processing_time:.2f}s\n")
            return response

        # Lấy session từ Redis
        session_key = f"session_history:{user_id}"
        session_data = redis_client.get(session_key)
        if session_data:
            session_data = json.loads(session_data)
            session_data = session_data[-2:]
        else:
            session_data = []
        context = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in session_data]) if session_data else ""
        
        # Truy vấn collection duy nhất, lọc theo nhiều category nếu có
        relevant_content = get_relevant_content(query, categories=categories, final_results=4)
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
        session_data.append({"question": query, "answer": response})
        redis_client.setex(session_key, 900, json.dumps(session_data))
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
@app.route('/ask', methods=['GET', 'POST'])
def ask():
    if request.method == 'POST':
        data = request.json
        query = data.get('query', '')
        user_id = data.get('user_id', '')
        stream = data.get('stream', False)
    else:  # GET request for streaming
        query = request.args.get('query', '')
        user_id = request.args.get('user_id', '')
        stream = request.args.get('stream', 'false').lower() == 'true'

    # category có thể là 1 string hoặc list, chuẩn hóa thành list nếu có
    categories = data.get('category', None) if request.method == 'POST' else None
    if isinstance(categories, str):
        categories = [categories]
    # Nếu không có category từ client, tự động phân loại bằng LLM
    if not categories:
        categories = classify_categories_llm(query)
        print(f"LLM classified categories: {categories}")
    if not query or not user_id:
        print('Error: Missing query or user_id', {'query': query, 'user_id': user_id})
        return jsonify({'error': 'No query or user_id provided'}), 400

    # Kiểm tra cache nếu không phải streaming
    if not stream:
        cached_response = get_cached_response(query)
        if cached_response:
            session_key = f"session_history:{user_id}"
            session_data = redis_client.get(session_key)
            if session_data:
                session_data = json.loads(session_data)
            else:
                session_data = []
            session_data.append({'question': query, 'answer': cached_response})
            if len(session_data) > 5:
                session_data = session_data[-4:]
            redis_client.setex(session_key, 1800, json.dumps(session_data))
            print('Response from cache:', cached_response)
            return jsonify({'response': cached_response})

    # Xử lý truy vấn với categories
    if stream:
        def generate():
            try:
                start = time.time()
                normalized_query = normalize_query(query)
                cached = get_cached_response(query)
                if cached:
                    # phát lại qua SSE để frontend nhận tức thì
                    yield f'data: {json.dumps({"content": cached, "cached": True})}\n\n'
                    yield f'event: done\ndata: {json.dumps({"full_response": cached})}\n\n'
                    return

                # Nếu categories chỉ là ['small_talk'] thì bỏ qua vector search
                if categories == ["small_talk"]:
                    print("Detected small talk category. Skipping vector search.")
                    session_key = f"session_history:{user_id}"
                    session_data = redis_client.get(session_key)
                    if session_data:
                        session_data = json.loads(session_data)
                        session_data = session_data[-1:]
                    else:
                        session_data = []
                    context = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in session_data]) if session_data else ""
                    base_prompt = f"Lịch sử trò chuyện:\n{context}\n\nCÂU HỎI: {query}\nTRẢ LỜI:" if context else f"CÂU HỎI: {query}\nTRẢ LỜI:"
                    
                    full_response = ""
                    for chunk in get_llm_response(base_prompt, stream=True):
                        if chunk.startswith("data: "):
                            data = json.loads(chunk[6:])
                            if "content" in data:
                                yield chunk
                                full_response += data["content"]
                        elif chunk.startswith("event: done"):
                            # Update session and cache after completion
                            session_data.append({"question": query, "answer": full_response})
                            redis_client.setex(session_key, 900, json.dumps(session_data))
                            set_cached_response(query, full_response)
                            processing_time = time.time() - start
                            log_qa(user_id, query, full_response, processing_time)
                            yield chunk
                    return

                # Lấy session từ Redis
                session_key = f"session_history:{user_id}"
                session_data = redis_client.get(session_key)
                if session_data:
                    session_data = json.loads(session_data)
                    session_data = session_data[-2:]
                else:
                    session_data = []
                context = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in session_data]) if session_data else ""
                
                # Truy vấn collection duy nhất, lọc theo nhiều category nếu có
                relevant_content = get_relevant_content(query, categories=categories, final_results=4)
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
                
                full_response = ""
                for chunk in get_llm_response(prompt_with_context, stream=True):
                    if chunk.startswith("data: "):
                        data = json.loads(chunk[6:])
                        if "content" in data:
                            yield chunk
                            full_response += data["content"]
                    elif chunk.startswith("event: done"):
                        # Update session and cache after completion
                        session_data.append({"question": query, "answer": full_response})
                        redis_client.setex(session_key, 900, json.dumps(session_data))
                        set_cached_response(query, full_response)
                        processing_time = time.time() - start
                        log_qa(user_id, query, full_response, processing_time)
                        yield chunk
                return
                
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                error_msg = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
                log_qa(user_id, query, error_msg)
                yield f"data: {json.dumps({'content': error_msg})}\n\n"
                yield f"event: done\ndata: {json.dumps({'full_response': error_msg})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*'
            }
        )
    else:
        response = process_user_query(query, user_id, categories=categories)
        print('Response from process_user_query:', response)
        session_key = f"session_history:{user_id}"
        session_data = redis_client.get(session_key)
        if session_data:
            session_data = json.loads(session_data)
        else:
            session_data = []
        session_data.append({'question': query, 'answer': response})
        if len(session_data) > 5:
            session_data = session_data[-4:]
        redis_client.setex(session_key, 900, json.dumps(session_data))
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

CATEGORIES = [
    "scholarship",
    "ictu_slogan",
    "training_and_regulations",
    "tuition_and_support",
    "student_affairs",
    "small_talk"
]

CATEGORY_DESCRIPTIONS = {
    "scholarship": (
        "Câu hỏi về học bổng, điều kiện, quy trình, mức học bổng, xét duyệt học bổng."
    ),
    "ictu_slogan": (
        "Chỉ bao gồm TẦM NHÌN, SỨ MỆNH, GIÁ TRỊ CỐT LÕI, TRIẾT LÝ GIÁO DỤC của ICTU. "
        "Không bao gồm quyền, nghĩa vụ, môi trường học tập hay đời sống sinh viên."
    ),
    "training_and_regulations": (
        "Chương trình đào tạo, chương trình học, đăng ký học phần, tín chỉ, quy chế – quy định học vụ."
    ),
    "tuition_and_support": (
        "Học phí, các khoản thu, miễn giảm, hỗ trợ tài chính."
    ),
    "student_affairs": (
        "Quản lý, sinh viên, hồ sơ, thẻ SV, nội trú, ĐỜI SỐNG sinh viên,  "
        "QUYỀN & NGHĨA VỤ & NHIỆM VỤ người học, môi trường học tập an toàn, hoạt động ngoại khóa."
        "Công tác, quản lý sinh viên, đoàn hội, giáo dục, khen thưởng, kỷ luật"
    ),
    "small_talk": (
        "Chào hỏi, xã giao, cảm ơn, xin lỗi, hỏi bot là ai, hỏi thời tiết… "
        "Các câu hỏi KHÔNG liên quan tới thông tin trường ICTU."
    )
}

# Hàm phân loại
def classify_categories_llm(query: str) -> List[str]:
    system_prompt = (
        "Bạn là hệ thống phân loại câu hỏi vào đúng category.\n"
        "Mô tả category:\n" +
        "\n".join(f"- {c}: {CATEGORY_DESCRIPTIONS[c]}" for c in CATEGORIES) +
        "\nChỉ chọn category khi câu hỏi phù hợp nhất với mô tả; nếu quá mơ hồ, trả về []."
    )

    cat_list = "\n".join(f"- {c}" for c in CATEGORIES)
    user_prompt = (
        f"Danh sách category hợp lệ:\n{cat_list}\n"
        "Trả về JSON array, tối đa 2 phần tử, ví dụ: [\"scholarship\"].\n"
        "Nếu câu hỏi là small talk (chào hỏi, xã giao…), trả về [\"small_talk\"].\n"
        "Nếu KHÔNG chắc chắn, trả về [].\n"
        f"CÂU HỎI: {query}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=120
        )
        raw = resp.choices[0].message.content

        # Làm sạch & parse JSON
        raw = raw.replace("```json", "").replace("```", "").strip()
        import json as _json
        try:
            cats = _json.loads(raw)
        except _json.JSONDecodeError:
            cats = _json.loads(f"[{raw.strip('[]')}]")  # Fallback tối giản

        # (e) Lọc hợp lệ
        if isinstance(cats, list):
            valid = [c for c in cats if c in CATEGORIES]
            return valid[:2] if valid else []
        return []

    except Exception as e:
        print("LLM classify error:", e)
        return []

# Chạy server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1508)