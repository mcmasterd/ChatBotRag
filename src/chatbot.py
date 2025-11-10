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
import tiktoken

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
TOKEN_LOG_FILE = os.path.join(LOGS_DIR, "token_usage_log.csv")

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True, password='terminator')

# Định nghĩa Local Embedding Function
class LocalEmbeddingFunction:
    def __init__(self, timeout=30):
        self.timeout = timeout
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        start_time = time.time()
        try:
            response = requests.post(
                "http://192.168.1.131:8000/embed", 
                json={"texts": input},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Log embedding usage
            processing_time = time.time() - start_time
            input_text = " | ".join(input)
            log_token_usage(
                user_id="system",
                operation_type="embedding_generation",
                model="local_embedding",
                input_text=input_text,
                output_text=f"Generated {len(input)} embeddings",
                processing_time=processing_time,
                additional_info=f"texts_count={len(input)}, avg_length={sum(len(t) for t in input)/len(input):.1f}"
            )
            
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

# Token logging setup
TOKEN_PRICING = {
    "gpt-4o": {"input": 0.0025 / 1000, "output": 0.01 / 1000},
    "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
    "embedding": {"input": 0.0001 / 1000, "output": 0}  # Local embedding free
}

def init_token_log_file():
    """Khởi tạo file log token usage"""
    try:
        if not os.path.exists(TOKEN_LOG_FILE):
            with open(TOKEN_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp',
                    'User_ID',
                    'Operation_Type',
                    'Model',
                    'Input_Tokens',
                    'Output_Tokens',
                    'Total_Tokens',
                    'Input_Text_Preview',
                    'Output_Text_Preview',
                    'Cost_USD',
                    'Processing_Time_Seconds',
                    'Additional_Info'
                ])
    except Exception as e:
        print(f"Error initializing token log file: {e}")

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in text using tiktoken"""
    try:
        if model.startswith("gpt-4"):
            encoding = tiktoken.encoding_for_model("gpt-4")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(str(text)))
    except Exception as e:
        print(f"Token counting error: {e}")
        # Fallback estimation
        return int(len(str(text).split()) * 1.3)

def log_token_usage(
    user_id: str,
    operation_type: str,
    model: str,
    input_text: str,
    output_text: str = "",
    processing_time: float = 0.0,
    additional_info: str = ""
):
    """Log detailed token usage"""
    try:
        input_tokens = count_tokens(input_text, model)
        output_tokens = count_tokens(output_text, model) if output_text else 0
        total_tokens = input_tokens + output_tokens
        
        # Calculate cost
        pricing = TOKEN_PRICING.get(model, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
        
        # Create previews (first 100 chars)
        input_preview = str(input_text)[:100].replace('\n', ' ').replace(',', ';')
        output_preview = str(output_text)[:100].replace('\n', ' ').replace(',', ';')
        
        timestamp = datetime.datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%d-%m-%Y %H:%M:%S")
        
        row = [
            timestamp,
            user_id or "system",
            operation_type,
            model,
            input_tokens,
            output_tokens,
            total_tokens,
            input_preview,
            output_preview,
            f"{cost:.6f}",
            f"{processing_time:.3f}",
            additional_info
        ]
        
        with open(TOKEN_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        print(f"Token log: {operation_type} | {model} | {input_tokens}+{output_tokens}={total_tokens} tokens | ${cost:.6f}")
        
    except Exception as e:
        print(f"Error logging tokens: {e}")

# Khởi tạo token log
init_token_log_file()

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

def get_relevant_content(query: str, categories: List[str] = None, final_results: int = 3) -> List[Dict]:
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
            initial_results = 20
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
            context_str += f"\n[{metadata['source']}]"
        
        context_items.append(context_str)
    
    if not context_items:
        context_items = ["Không có thông tin phù hợp."]
    
    combined_context = "\n\n".join(context_items)  # Bỏ separator "---"
    prompt = f"Thông tin:\n{combined_context}\n\nCâu hỏi: {query}\nTrả lời:"
    
    return prompt

system_msg = """
    Bạn là trợ lý tư vấn thông tin của trường ICTU.
    Hãy chuyển toàn bộ ký tự đặc biệt/markdown về văn bản thuần.    
    Chỉ sử dụng thông tin đã được cung cấp để trả lời, không suy đoán không căn cứ.
    Nếu không có đủ thông tin, hãy thừa nhận một cách lịch sự và đề nghị người dùng hỏi lại.
    Nếu có thông tin mâu thuẫn, ưu tiên nguồn mới nhất.
    Trả lời chắc chắn, trọng tâm, đầy đủ, câu hỏi với giọng điệu thân thiện và gần gũi.
    Không dài dòng, nhưng  nếu người dùng yêu cầu có thể cung cấp thêm chi tiết.
"""

def get_llm_response(prompt: str, stream: bool = False, user_id: str = None, 
                    operation_type: str = "llm_response") -> str | Generator:
    """Get response from language model with optional streaming support"""
    start = time.time()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=400,  # Giảm từ 600 xuống 400
        presence_penalty=0.3,
        stream=stream
    )
    
    if not stream:
        processing_time = time.time() - start
        content = response.choices[0].message.content
        
        # Log token usage for non-streaming
        if hasattr(response, 'usage') and response.usage:
            log_token_usage(
                user_id=user_id or "system",
                operation_type=operation_type,
                model="gpt-4o-mini",
                input_text=f"System: {system_msg[:50]}...\nUser: {prompt}",
                output_text=content,
                processing_time=processing_time,
                additional_info=f"prompt_tokens={response.usage.prompt_tokens}, completion_tokens={response.usage.completion_tokens}"
            )
        
        print(f"LLM response time: {processing_time:.4f}s")
        return content
    
    def generate():
        full_response = ""
        total_tokens_estimate = 0
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield f"data: {json.dumps({'content': content})}\n\n"
            
            # Lấy usage từ chunk cuối cùng nếu có
            if hasattr(chunk, 'usage') and chunk.usage:
                total_tokens_estimate = chunk.usage.total_tokens
        
        # Log token usage cho streaming sau khi hoàn thành
        processing_time = time.time() - start
        log_token_usage(
            user_id=user_id or "system",
            operation_type=f"{operation_type}_streaming",
            model="gpt-4o-mini",
            input_text=f"System: {system_msg[:50]}...\nUser: {prompt}",
            output_text=full_response,
            processing_time=processing_time,
            additional_info=f"estimated_total_tokens={total_tokens_estimate}, response_length={len(full_response)}"
        )
        
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

def process_user_query(query: str, user_id: str, categories: List[str] = None, rephrased_query: str = None) -> str:
    try:
        start = time.time()
        normalized_query = normalize_query(query)

        # Kiểm tra cache 
        cached_response = get_cached_response(query)
        if cached_response:
            processing_time = time.time() - start
            log_qa(user_id, query, cached_response, processing_time)
            
            # Log cache hit
            log_token_usage(
                user_id=user_id,
                operation_type="cache_hit",
                model="cache",
                input_text=query,
                output_text=cached_response,
                processing_time=processing_time,
                additional_info="Retrieved from cache"
            )
            
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
            
            response = get_llm_response(base_prompt, user_id=user_id, operation_type="small_talk_response")

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
        
        # Log session history tokens
        if context:
            log_token_usage(
                user_id=user_id,
                operation_type="session_history",
                model="gpt-4o-mini",
                input_text=context,
                output_text="",
                processing_time=0,
                additional_info=f"history_items={len(session_data)}"
            )
        
        # Sử dụng rephrased_query nếu có, không thì dùng query gốc
        vector_query = rephrased_query if rephrased_query else query
        if rephrased_query:
            print(f"Using rephrased query for vector search: '{rephrased_query}' (original: '{query}')")
        
        # Truy vấn collection duy nhất, lọc theo nhiều category nếu có
        vector_start = time.time()
        relevant_content = get_relevant_content(vector_query, categories=categories, final_results=3)  # Giảm từ 5 xuống 3
        vector_time = time.time() - vector_start
        
        # Log vector search results
        if relevant_content:
            context_text = "\n---\n".join([item.get('document', '')[:200] + "..." for item in relevant_content])
            log_token_usage(
                user_id=user_id,
                operation_type="vector_search_results",
                model="retrieval",
                input_text=vector_query,
                output_text=context_text,
                processing_time=vector_time,
                additional_info=f"results_count={len(relevant_content)}, categories={categories}"
            )
        
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
        
        # Log final prompt
        log_token_usage(
            user_id=user_id,
            operation_type="final_prompt",
            model="gpt-4o-mini",
            input_text=prompt_with_context,
            output_text="",
            processing_time=0,
            additional_info=f"sources={sources_str}"
        )
        
        response = get_llm_response(prompt_with_context, user_id=user_id, operation_type="main_response")
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

CATEGORIES = [
    "student_affairs",                  # Công tác sinh viên, quyền, nghĩa vụ, hồ sơ/thẻ, ngoại khóa, nội trú, rèn luyện, đời sống
    "training_and_regulations",         # Chương trình đào tạo, quy chế đào tạo, quản lý học vụ, văn bằng, chứng chỉ
    "tuition_and_support",              # Học phí, miễn giảm, hỗ trợ tài chính, phí dịch vụ
    "scholarship",                      # Học bổng, xét duyệt, điều kiện học bổng
    "ictu_slogan",                      # Tầm nhìn, sứ mệnh, giá trị cốt lõi, triết lý giáo dục của ICTU
    "information_security_and_conduct", # An ninh mạng, quy tắc ứng xử, bảo mật, đạo đức số
    "small_talk"                        # Chào hỏi, xã giao, các câu không liên quan trường
]

CATEGORY_DESCRIPTIONS = {
    "student_affairs": (
        "TẤT CẢ các vấn đề về quản lý sinh viên/người học, lưu học sinh: quyền, nghĩa vụ, nhiệm vụ, hồ sơ, thẻ sinh viên, nội trú, ngoại trú, các vấn đề về hoạt động, điểm ngoại khóa, khen thưởng, kỷ luật, tổ chức lớp, đoàn hội, đời sống, sức khỏe, tư vấn, các chính sách liên quan sinh viên và cán bộ/giảng viên trong nhà trường.\n"
        "Bao gồm cả quy định về đánh giá rèn luyện, hồ sơ cá nhân, thẻ SV, nội trú, quản lý hồ sơ – thủ tục của người học, quy tắc ứng xử, đạo đức, thái độ, tác phong, ứng xử của cán bộ, giảng viên, sinh viên trong nhà trường.\n"
    ),
    "training_and_regulations": (
        "TẤT CẢ các vấn đề về chương trình đào tạo, quy chế – quy định học vụ, tín chỉ, tổ chức đào tạo, nội quy giảng đường, quản lý học phần, đăng ký học, xét tốt nghiệp, quản lý/ cấp phát văn bằng, chứng chỉ, phân loại văn bằng, hệ thống văn bằng quốc gia, sửa đổi văn bằng, chuyển trường, ngành, quy định về hành vi, trang phục, thái độ học tập, vi phạm quy chế học tập, xử lý vi phạm trên lớp, kỷ luật lớp học."
        "Bao gồm cả các câu hỏi về khái niệm, định nghĩa, giải thích về hệ thống văn bằng, các loại văn bằng, chuẩn văn bằng, quy định hệ văn bằng quốc gia."
    ),
    "tuition_and_support": (
        "Câu hỏi về học phí, các khoản thu, miễn giảm, hỗ trợ tài chính, các dịch vụ hỗ trợ tài chính cho sinh viên."
    ),
    "scholarship": (
        "Mọi vấn đề về học bổng: điều kiện, xét duyệt, quy trình, mức học bổng, chế độ học bổng, các loại học bổng trong trường. Nếu câu hỏi về nghĩa vụ, chính sách, phân công sau học bổng thì chọn student_affairs."
    ),
    "ictu_slogan": (
        "Chỉ bao gồm các nội dung về TẦM NHÌN, SỨ MỆNH, GIÁ TRỊ CỐT LÕI, TRIẾT LÝ GIÁO DỤC của ICTU. Không chọn nếu hỏi về môi trường, quyền nghĩa vụ, hay thông tin ngoài slogan."
    ),
    "information_security_and_conduct": (
        "Quy định về an ninh mạng, bảo mật dữ liệu cá nhân, luật, quy tắc ứng xử trong môi trường số, quy tắc ứng xử trên mạng xã hội, đạo đức công nghệ thông tin, phòng chống vi phạm pháp luật trên mạng, các khái niệm, định nghĩa, giải thích về không gian, an ninh mạng, hệ thống thông tin trọng yếu quốc gia."
    ),
    "small_talk": (
        "Chào hỏi, xã giao, cảm ơn, xin lỗi, hỏi bot là ai, hỏi thời tiết, các câu hỏi KHÔNG liên quan đến thông tin/quy định trường ICTU."
    )
}

# Hàm phân loại
def classify_categories_llm(query: str, history: list = None, user_id: str = None) -> dict:
    """
    Phân loại category và rephrase câu hỏi nếu cần
    Luôn trả về dict: {"categories": [...], "rephrased_query": "..." hoặc None}
    """
    start_time = time.time()
    system_prompt = (
        "Bạn là hệ thống phân loại câu hỏi vào đúng category của trường ICTU. "
        "Chỉ rephrase nếu câu hỏi có đại từ hoặc phụ thuộc vào hội thoại trước, còn lại chỉ phân loại. "
        "Luôn trả về JSON: {categories: [...], rephrased_query: ...}. Nếu không cần rephrase, rephrased_query là null."
        "\nMô tả category:\n" + "\n".join(f"- {c}: {CATEGORY_DESCRIPTIONS[c][:80]}" for c in CATEGORIES)  # rút gọn mô tả
    )
    cat_list = ", ".join(CATEGORIES)
    # Tạo phần lịch sử hội thoại nếu có
    history_str = ""
    if history:
        for item in history[-1:]:
            history_str += f"Q: {item.get('question','')}\nA: {item.get('answer','')}\n"
    user_prompt = (
        f"Các category hợp lệ: {cat_list}\n"
        f"Lịch sử hội thoại gần nhất:\n{history_str}"
        f"Câu hỏi hiện tại: {query}\n"
        "Trả về JSON: {\"categories\": [...], \"rephrased_query\": ...}. Nếu không cần rephrase, rephrased_query là null."
    )
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        processing_time = time.time() - start_time
        raw = resp.choices[0].message.content
        
        # Log classification tokens
        if hasattr(resp, 'usage') and resp.usage:
            log_token_usage(
                user_id=user_id or "system",
                operation_type="category_classification",
                model="gpt-4o",
                input_text=f"System: {system_prompt[:100]}...\nUser: {user_prompt}",
                output_text=raw,
                processing_time=processing_time,
                additional_info=f"prompt_tokens={resp.usage.prompt_tokens}, completion_tokens={resp.usage.completion_tokens}"
            )
        
        raw = raw.replace("```json", "").replace("```", "").strip()
        
        import json as _json
        try:
            result = _json.loads(raw)
        except _json.JSONDecodeError:
            # Fallback: trả về cấu trúc mặc định
            print(f"JSON decode error, raw response: {raw}")
            result = {"categories": [], "rephrased_query": None}
        
        # Đảm bảo kết quả có cấu trúc đúng
        if not isinstance(result, dict):
            result = {"categories": [], "rephrased_query": None}
        
        # Lọc categories hợp lệ
        cats = result.get("categories", [])
        if isinstance(cats, list):
            valid = [c for c in cats if c in CATEGORIES]
            result["categories"] = valid[:2] if valid else []
        else:
            result["categories"] = []
        
        # Xử lý rephrased_query
        rephrased = result.get("rephrased_query")
        if rephrased and isinstance(rephrased, str) and rephrased.strip():
            result["rephrased_query"] = rephrased.strip()
        else:
            result["rephrased_query"] = None
            
        return result
        
    except Exception as e:
        print(f"LLM classify error: {e}")
        return {"categories": [], "rephrased_query": None}

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
    if not query or not user_id:
        print('Error: Missing query or user_id', {'query': query, 'user_id': user_id})
        return jsonify({'error': 'No query or user_id provided'}), 400

    # ĐƯA BƯỚC KIỂM TRA CACHE LÊN TRƯỚC PHÂN LOẠI CATEGORY
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

    # Nếu không có category từ client, tự động phân loại bằng LLM
    if not categories:
        # Lấy lịch sử hội thoại gần nhất
        session_key = f"session_history:{user_id}"
        session_data = redis_client.get(session_key)
        if session_data:
            session_data = json.loads(session_data)
        else:
            session_data = []
        
        try:
            classify_result = classify_categories_llm(query, history=session_data, user_id=user_id)
            # Đảm bảo kết quả luôn là dict với cấu trúc đúng
            if isinstance(classify_result, dict):
                categories = classify_result.get("categories", [])
                rephrased_query = classify_result.get("rephrased_query")
            else:
                # Fallback nếu kết quả không đúng định dạng
                categories = classify_result if isinstance(classify_result, list) else []
                rephrased_query = None
            print(f"LLM classified categories: {categories}, rephrased_query: {rephrased_query}")
        except Exception as e:
            print(f"Error in category classification: {e}")
            categories = []
            rephrased_query = None
    else:
        rephrased_query = None
    if not query or not user_id:
        print('Error: Missing query or user_id', {'query': query, 'user_id': user_id})
        return jsonify({'error': 'No query or user_id provided'}), 400

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
                    for chunk in get_llm_response(base_prompt, stream=True, user_id=user_id, operation_type="small_talk_streaming"):
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
                
                # Sử dụng rephrased_query nếu có, không thì dùng query gốc
                vector_query = rephrased_query if rephrased_query else query
                if rephrased_query:
                    print(f"Using rephrased query for vector search: '{rephrased_query}' (original: '{query}')")
                
                # Truy vấn collection duy nhất, lọc theo nhiều category nếu có
                relevant_content = get_relevant_content(vector_query, categories=categories, final_results=5)
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
                for chunk in get_llm_response(prompt_with_context, stream=True, user_id=user_id, operation_type="main_streaming"):
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
        response = process_user_query(query, user_id, categories=categories, rephrased_query=rephrased_query)
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

# Chạy server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1508)