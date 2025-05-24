import os
import re
import chromadb
from typing import List, Dict, Any
import logging
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor
from ratelimit import limits, sleep_and_retry
from sentence_transformers import SentenceTransformer

# Cấu hình logging với encoding UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Khởi tạo ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Lớp xử lý embedding bằng sentence-transformers
class LocalEmbeddingFunction:
    def __init__(self):
        try:
            # Khởi tạo mô hình với device CPU
            self.model = SentenceTransformer('NghiemAbe/Vi-Legal-Bi-Encoder-v2')
            self.model.to('cpu')  # Đảm bảo sử dụng CPU
        except Exception as e:
            logging.error(f"Không thể tải mô hình NghiemAbe/Vi-Legal-Bi-Encoder-v2: {str(e)}")
            sys.exit(1)

    def __call__(self, input: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(input, convert_to_numpy=True).tolist()
            if len(embeddings) != len(input):
                logging.error(f"Số embedding không khớp: kỳ vọng {len(input)}, nhận được {len(embeddings)}")
                return [[0.0] * 768 for _ in input]  # Giả định 768 chiều, điều chỉnh nếu khác
            return embeddings
        except Exception as e:
            logging.error(f"Lỗi khi tạo embedding: {str(e)}")
            return [[0.0] * 768 for _ in input]

# Khởi tạo mô hình SentenceTransformer một lần duy nhất
try:
    embedding_function = LocalEmbeddingFunction()
except Exception as e:
    logging.error(f"Không thể khởi tạo LocalEmbeddingFunction: {str(e)}")
    sys.exit(1)

# Hàm embedding với giới hạn request
@sleep_and_retry
@limits(calls=60, period=60)
def rate_limited_embedding(texts: List[str]) -> List[List[float]]:
    try:
        embeddings = embedding_function(texts)
        if len(embeddings) != len(texts):
            logging.error(f"Số embedding không khớp: kỳ vọng {len(texts)}, nhận được {len(embeddings)}")
            return [[0.0] * 768 for _ in texts]  # Giả định 768 chiều
        return embeddings
    except Exception as e:
        logging.error(f"Lỗi khi tạo embedding: {str(e)}")
        return [[0.0] * 768 for _ in texts]

# Đường dẫn đến file trạng thái
PROCESSED_FILES_PATH = "processed_files.json"

def load_processed_files() -> Dict[str, Dict[str, Any]]:
    if os.path.exists(PROCESSED_FILES_PATH):
        try:
            with open(PROCESSED_FILES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    logging.error(f"File processed_files.json có định dạng không hợp lệ: {data}")
                    return {}
                return data
        except Exception as e:
            logging.error(f"Lỗi khi đọc file processed_files.json: {str(e)}")
            return {}
    return {}

def save_processed_files(processed_files: Dict[str, Dict[str, Any]]):
    try:
        with open(PROCESSED_FILES_PATH, 'w', encoding='utf-8') as f:
            json.dump(processed_files, f, indent=4)
    except Exception as e:
        logging.error(f"Lỗi khi lưu file processed_files.json: {str(e)}")

# Đường dẫn đến thư mục data
data_folder = 'data'

def get_data_files(data_folder: str) -> List[Dict[str, str]]:
    data_files = []
    try:
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.lower().endswith('.md'):
                    data_files.append({"path": os.path.join(root, file), "type": "markdown"})
                elif file.lower().endswith('.json'):
                    data_files.append({"path": os.path.join(root, file), "type": "json"})
        logging.info(f"Tìm thấy {len(data_files)} file (.md và .json) trong thư mục {data_folder}")
    except Exception as e:
        logging.error(f"Lỗi khi đọc thư mục {data_folder}: {str(e)}")
    return data_files

def parse_markdown(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Lỗi khi đọc file {file_path}: {str(e)}")
        return []

    sections = content.split('---')
    parsed_sections = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        metadata = {
            "document_type": "markdown",
            "category": "Scholarship",
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "source": "docs"
        }
        content = ''
        in_metadata = False

        lines = section.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('## Metadata'):
                in_metadata = True
                continue
            elif line.startswith('## Nội dung'):
                in_metadata = False
                continue
            if in_metadata and line.startswith('- **'):
                remaining = line[4:]
                if ':**' in remaining:
                    key_value = remaining.split(':**', 1)
                    key = key_value[0].strip()
                    value = key_value[1].strip() if len(key_value) > 1 else ""
                    metadata[key] = value
            elif not in_metadata:
                content += line + '\n'

        # Trích xuất tiêu đề và nội dung chính
        title_match = re.search(r'### (.*?)\n', content)
        if not title_match:
            logging.warning(f"Section bị bỏ qua trong {file_path}: {section[:100]}... (Không tìm thấy tiêu đề dạng ###)")
            continue

        title = title_match.group(1).strip()
        # Lấy toàn bộ nội dung sau tiêu đề, không yêu cầu "- Nội dung:"
        content_start_index = content.find(title_match.group(0)) + len(title_match.group(0))
        content_text = content[content_start_index:].strip()
        if not content_text:
            logging.warning(f"Section bị bỏ qua trong {file_path}: {section[:100]}... (Nội dung trống)")
            continue

        document = f"{title}: {content_text}"

        # Trích xuất thêm metadata nếu có
        article_match = re.search(r'Điều (\d+)', title)
        if article_match:
            metadata["article"] = article_match.group(1)

        clause_match = re.search(r'Khoản (\d+)', title)
        if clause_match:
            metadata["clause"] = clause_match.group(1)

        parsed_sections.append({
            'document': document,
            'metadata': metadata
        })

    logging.info(f"Tìm thấy {len(parsed_sections)} section trong file {file_path}")
    return parsed_sections

def parse_json(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Lỗi cú pháp JSON trong file {file_path}: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Lỗi khi đọc file JSON {file_path}: {str(e)}")
        return []

    # Kiểm tra xem dữ liệu có phải là danh sách trực tiếp không
    if isinstance(data, list):
        qa_pairs = data
    elif isinstance(data, dict) and "qa_pairs" in data:
        qa_pairs = data["qa_pairs"]
    else:
        logging.error(f"File JSON {file_path} không có key 'qa_pairs' và không phải danh sách trực tiếp. Nội dung file: {json.dumps(data, ensure_ascii=False)[:200]}...")
        return []

    parsed_sections = []
    for i, item in enumerate(qa_pairs):
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            logging.warning(f"Bỏ qua mục không hợp lệ trong {file_path}: {item}")
            continue

        question = item["question"].strip()
        answer = item["answer"].strip()
        document = f"{question}: {answer}"

        metadata = {
            "document_type": "json",
            "category": item.get("category", "FAQs"),
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "source": "FAQs",
            "qa_source": item.get("qa_source", ""),
            "date": item.get("date", ""),
            "amend": item.get("amend", ""),
            "data_type": item.get("data_type", "faqs"),
            "doc_id": item.get("doc_id", "")
        }

        parsed_sections.append({
            'document': document,
            'metadata': metadata
        })

    logging.info(f"Tìm thấy {len(parsed_sections)} cặp Q&A trong file {file_path}")
    return parsed_sections

def process_file(file_info, collection, processed_files):
    start_time = time.time()
    file_path = file_info["path"]
    file_type = file_info["type"]
    try:
        logging.info(f"Xử lý file {file_path}")
        if file_type == "markdown":
            sections = parse_markdown(file_path)
        else:
            sections = parse_json(file_path)

        if not sections:
            logging.warning(f"Không tìm thấy section nào trong {file_path}")
            return 0

        documents = [section['document'] for section in sections]
        metadatas = [section['metadata'] for section in sections]
        ids = [f"{file_type}_{os.path.basename(file_path).replace('.md', '').replace('.json', '')}_section_{i}" 
               for i in range(len(sections))]

        # Kiểm tra các section đã xử lý
        processed_sections = processed_files.get(file_path, {}).get("sections", [])
        sections_to_process = [(doc, meta, id_) for doc, meta, id_ in zip(documents, metadatas, ids) 
                             if id_ not in processed_sections]

        if not sections_to_process:
            logging.info(f"Tất cả section trong {file_path} đã được xử lý")
            return 0

        documents_to_process = [item[0] for item in sections_to_process]
        metadatas_to_process = [item[1] for item in sections_to_process]
        ids_to_process = [item[2] for item in sections_to_process]

        # Làm sạch dữ liệu đầu vào
        documents_to_process = [
            ''.join(c for c in doc if c.isprintable()) if doc and len(doc.strip()) > 0 else "Nội dung trống"
            for doc in documents_to_process
        ]
        if any(len(doc) > 10000 for doc in documents_to_process):
            logging.warning(f"Có document quá dài trong {file_path}, có thể gây lỗi embedding")

        # Xử lý theo batch
        batch_size = 10
        processed_count = 0
        for i in range(0, len(documents_to_process), batch_size):
            batch_end = min(i + batch_size, len(documents_to_process))
            batch_docs = documents_to_process[i:batch_end]
            batch_meta = metadatas_to_process[i:batch_end]
            batch_ids = ids_to_process[i:batch_end]
            
            retry_count = 5
            for attempt in range(retry_count):
                try:
                    embeddings = rate_limited_embedding(batch_docs)
                    if not embeddings or all(not emb for emb in embeddings):
                        logging.error(f"Embedding rỗng cho batch {i//batch_size + 1}. Documents: {batch_docs[:2]}...")
                        if attempt == retry_count - 1:
                            logging.error(f"Không thể xử lý batch {i//batch_size + 1} sau {retry_count} lần thử")
                            continue
                        continue
                
                    collection.add(
                        ids=batch_ids,
                        embeddings=embeddings,
                        metadatas=batch_meta,
                        documents=batch_docs
                    )
                    logging.info(f"Đã xử lý batch {i//batch_size + 1} với {len(batch_docs)} section")
                    processed_count += len(batch_docs)
                    
                    # Cập nhật trạng thái ngay sau mỗi batch thành công
                    if file_path not in processed_files:
                        processed_files[file_path] = {"mtime": os.path.getmtime(file_path), "sections": []}
                    processed_files[file_path]["sections"].extend(batch_ids)
                    save_processed_files(processed_files)
                    break
                except Exception as e:
                    logging.error(f"Lỗi khi tạo embedding cho batch {i//batch_size + 1}, lần thử {attempt + 1}/{retry_count}: {str(e)}")
                    if attempt == retry_count - 1:
                        logging.error(f"Không thể xử lý batch {i//batch_size + 1} sau {retry_count} lần thử")
                        continue
        logging.info(f"Thời gian xử lý file {file_path}: {time.time() - start_time:.4f}s")
        return processed_count
    except Exception as e:
        logging.error(f"Lỗi khi xử lý {file_path}: {str(e)}")
        return 0

def embed_and_store_data_files(data_folder: str, collection_name: str):
    files = get_data_files(data_folder)
    processed_files = load_processed_files()
    files_to_process = files

    if not files_to_process:
        logging.info("Không có file nào để xử lý")
        return

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function  # Sử dụng instance toàn cục
    )

    total_files = len(files_to_process)
    total_sections = 0
    failed_files = 0
    skipped_files = 0

    # Giảm số workers để tránh xung đột tài nguyên
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures_with_info = []
        for file_info in files_to_process:
            future = executor.submit(process_file, file_info, collection, processed_files)
            futures_with_info.append((future, file_info))

        for future, file_info in futures_with_info:
            sections_processed = future.result()
            file_path = file_info["path"]
            file_type = file_info["type"]
            if file_type == "markdown":
                sections = parse_markdown(file_path)
            else:
                sections = parse_json(file_path)
            total_sections_in_file = len(sections)
            processed_sections = processed_files.get(file_path, {}).get("sections", [])
            sections_to_process = total_sections_in_file - len(processed_sections)

            if sections_to_process == 0:
                skipped_files += 1
            elif sections_processed == 0:
                failed_files += 1
            total_sections += sections_processed

    logging.info(f"""
    Tổng kết quá trình embedding:
    - Tổng số file cần xử lý: {total_files}
    - Số file đã xử lý từ trước: {skipped_files}
    - Số file thất bại: {failed_files}
    - Tổng số section mới được xử lý: {total_sections}
    """)

if __name__ == "__main__":
    data_folder = "data"
    collection_name = "scholarship_documents"
    
    logging.info("Bắt đầu quá trình embedding...")
    embed_and_store_data_files(data_folder, collection_name)
    logging.info("Hoàn tất quá trình embedding!")