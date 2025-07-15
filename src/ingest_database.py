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
import argparse

# Cấu hình logging
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

# Danh sách category hợp lệ
valid_categories = [
    "scholarship",
    "ictu_slogan",
    "training_and_regulations",
    "tuition_and_support",
    "student_affairs",
    "career_and_startup_support"
]

# Lớp xử lý embedding
class LocalEmbeddingFunction:
    def __init__(self):
        try:
            self.model = SentenceTransformer('NghiemAbe/Vi-Legal-Bi-Encoder-v2')
            self.model.to('cpu')
        except Exception as e:
            logging.error(f"Không thể tải mô hình NghiemAbe/Vi-Legal-Bi-Encoder-v2: {str(e)}")
            sys.exit(1)

    def __call__(self, input: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(input, convert_to_numpy=True).tolist()
            if len(embeddings) != len(input):
                logging.error(f"Số embedding không khớp: kỳ vọng {len(input)}, nhận được {len(embeddings)}")
                return [[0.0] * 768 for _ in input]
            return embeddings
        except Exception as e:
            logging.error(f"Lỗi khi tạo embedding: {str(e)}")
            return [[0.0] * 768 for _ in input]

try:
    embedding_function = LocalEmbeddingFunction()
except Exception as e:
    logging.error(f"Không thể khởi tạo LocalEmbeddingFunction: {str(e)}")
    sys.exit(1)

@sleep_and_retry
@limits(calls=60, period=60)
def rate_limited_embedding(texts: List[str]) -> List[List[float]]:
    try:
        embeddings = embedding_function(texts)
        if len(embeddings) != len(texts):
            logging.error(f"Số embedding không khớp: kỳ vọng {len(texts)}, nhận được {len(embeddings)}")
            return [[0.0] * 768 for _ in texts]
        return embeddings
    except Exception as e:
        logging.error(f"Lỗi khi tạo embedding: {str(e)}")
        return [[0.0] * 768 for _ in texts]

PROCESSED_FILES_PATH = "processed_files.json"

def load_processed_files() -> Dict[str, Dict[str, Any]]:
    if os.path.exists(PROCESSED_FILES_PATH):
        try:
            with open(PROCESSED_FILES_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Lỗi khi đọc file processed_files.json: {str(e)}")
            return {}
    return {}

def calculate_content_hash(content: str) -> str:
    """Tính hash của nội dung để phát hiện thay đổi"""
    import hashlib
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def save_processed_files(processed_files: Dict[str, Dict[str, Any]]):
    try:
        with open(PROCESSED_FILES_PATH, 'w', encoding='utf-8') as f:
            json.dump(processed_files, f, indent=4)
    except Exception as e:
        logging.error(f"Lỗi khi lưu file processed_files.json: {str(e)}")

def get_data_files(data_folder: str) -> List[Dict[str, str]]:
    data_files = []
    try:
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.lower().endswith(('.md', '.json')):
                    data_files.append({"path": os.path.join(root, file), "type": "markdown" if file.endswith('.md') else "json"})
        logging.info(f"Tìm thấy {len(data_files)} file trong thư mục {data_folder}")
    except Exception as e:
        logging.error(f"Lỗi khi đọc thư mục {data_folder}: {str(e)}")
    return data_files

REQUIRED_FIELDS_MD = ["data_type", "category"]
REQUIRED_FIELDS_JSON = ["data_type", "category"]
OPTIONAL_FIELDS = ["doc_id", "source", "date", "partial_mod", "modify", "amend"]
ALL_FIELDS_MD = REQUIRED_FIELDS_MD + OPTIONAL_FIELDS
ALL_FIELDS_JSON = REQUIRED_FIELDS_JSON + OPTIONAL_FIELDS

def parse_markdown(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Lỗi đọc file {file_path}: {str(e)}")
        return []

    sections = content.split('---')
    parsed_sections = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        metadata = {}
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
                    if key in ALL_FIELDS_MD:
                        metadata[key] = value
            elif not in_metadata:
                content += line + '\n'

        title_match = re.search(r'###\s+(.*?)\s*(?:\r?\n|$)', content)
        if not title_match:
            logging.warning(f"Bỏ qua section trong {file_path}: Không tìm thấy tiêu đề dạng ###")
            continue

        title = title_match.group(1).strip()
        content_start_index = content.find(title_match.group(0)) + len(title_match.group(0))
        content_text = content[content_start_index:].strip()
        if not content_text:
            logging.warning(f"Bỏ qua section trong {file_path}: Nội dung trống")
            continue

        document = f"{title}: {content_text}"

        missing_required = [f for f in REQUIRED_FIELDS_MD if f not in metadata or not metadata[f]]
        if missing_required:
            logging.warning(f"Bỏ qua section trong {file_path}: Thiếu trường bắt buộc {missing_required}")
            continue

        if metadata['category'] not in valid_categories:
            logging.warning(f"Bỏ qua section trong {file_path}: Category không hợp lệ '{metadata['category']}'")
            continue

        for f in OPTIONAL_FIELDS:
            if f not in metadata:
                metadata[f] = '' if f != 'partial_mod' else False
        if 'partial_mod' in metadata:
            metadata['partial_mod'] = metadata['partial_mod'].lower() in ['true', '1', 'yes'] if isinstance(metadata['partial_mod'], str) else bool(metadata['partial_mod'])

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
        logging.error(f"Lỗi cú pháp JSON trong {file_path}: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Lỗi đọc file JSON {file_path}: {str(e)}")
        return []

    if isinstance(data, list):
        qa_pairs = data
    elif isinstance(data, dict) and "qa_pairs" in data:
        qa_pairs = data["qa_pairs"]
    else:
        logging.error(f"Bỏ qua {file_path}: Không có key 'qa_pairs' hoặc không phải danh sách")
        return []

    parsed_sections = []
    for i, item in enumerate(qa_pairs):
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            logging.warning(f"Bỏ qua mục {i} trong {file_path}: Không hợp lệ")
            continue

        question = item["question"].strip()
        answer = item["answer"].strip()
        document = f"{question}: {answer}"

        metadata = {f: item.get(f, '' if f != 'partial_mod' else False) for f in ALL_FIELDS_JSON}
        if 'partial_mod' in metadata:
            metadata['partial_mod'] = metadata['partial_mod'].lower() in ['true', '1', 'yes'] if isinstance(metadata['partial_mod'], str) else bool(metadata['partial_mod'])

        missing_required = [f for f in REQUIRED_FIELDS_JSON if f not in metadata or not metadata[f]]
        if missing_required:
            logging.warning(f"Bỏ qua mục {i} trong {file_path}: Thiếu trường bắt buộc {missing_required}")
            continue

        if metadata['category'] not in valid_categories:
            logging.warning(f"Bỏ qua mục {i} trong {file_path}: Category không hợp lệ '{metadata['category']}'")
            continue

        parsed_sections.append({
            'document': document,
            'metadata': metadata
        })

    logging.info(f"Tìm thấy {len(parsed_sections)} cặp Q&A trong file {file_path}")
    return parsed_sections

def check_file_needs_update(file_path: str, processed_files: Dict[str, Dict[str, Any]], force_update: bool = False) -> bool:
    """Kiểm tra xem file có cần cập nhật không dựa trên modification time"""
    if force_update:
        return True
        
    if file_path not in processed_files:
        return True
    
    try:
        current_mtime = os.path.getmtime(file_path)
        stored_mtime = processed_files[file_path].get('mtime', 0)
        return current_mtime > stored_mtime
    except Exception as e:
        logging.warning(f"Lỗi khi kiểm tra modification time của {file_path}: {str(e)}")
        return True

def remove_existing_sections_from_collection(collection, section_ids: List[str]):
    """Xóa các section đã tồn tại khỏi collection"""
    for section_id in section_ids:
        try:
            # Kiểm tra xem ID có tồn tại không
            result = collection.get(ids=[section_id])
            if result['ids']:
                collection.delete(ids=[section_id])
                logging.info(f"Đã xóa section cũ: {section_id}")
        except Exception as e:
            logging.warning(f"Lỗi khi xóa section {section_id}: {str(e)}")

def clear_collection_if_needed(collection, force_clear: bool = False):
    """Xóa toàn bộ dữ liệu trong collection nếu cần"""
    if force_clear:
        try:
            # Lấy tất cả IDs trong collection
            result = collection.get()
            if result['ids']:
                logging.warning(f"CẢNH BÁO: Sẽ xóa {len(result['ids'])} documents từ TOÀN BỘ collection!")
                logging.warning("Điều này sẽ xóa tất cả dữ liệu từ các folders khác!")
                collection.delete(ids=result['ids'])
                logging.info("Đã xóa toàn bộ collection")
        except Exception as e:
            logging.error(f"Lỗi khi xóa collection: {str(e)}")

def clear_folder_data_only(collection, data_folder: str, processed_files: Dict[str, Dict[str, Any]]):
    """Chỉ xóa dữ liệu từ folder cụ thể"""
    try:
        # Normalize đường dẫn để so sánh
        normalized_folder = os.path.normpath(data_folder).replace('\\', '/')
        
        # Tìm tất cả files trong folder này từ processed_files
        folder_files = []
        for fp in processed_files.keys():
            normalized_fp = os.path.normpath(fp).replace('\\', '/')
            if normalized_fp.startswith(normalized_folder):
                folder_files.append(fp)
        
        if not folder_files:
            logging.info(f"Không tìm thấy dữ liệu cũ từ folder {data_folder}")
            return
        
        # Lấy tất cả section IDs từ các files trong folder này
        sections_to_delete = []
        for file_path in folder_files:
            sections_to_delete.extend(processed_files[file_path].get("sections", []))
        
        if sections_to_delete:
            logging.info(f"Xóa {len(sections_to_delete)} sections cũ từ {len(folder_files)} files trong folder {data_folder}")
            remove_existing_sections_from_collection(collection, sections_to_delete)
            
            # Xóa thông tin các files trong folder này khỏi processed_files
            for file_path in folder_files:
                if file_path in processed_files:
                    del processed_files[file_path]
            save_processed_files(processed_files)
            logging.info(f"Đã xóa thông tin {len(folder_files)} files khỏi processed_files.json")
        
    except Exception as e:
        logging.error(f"Lỗi khi xóa dữ liệu folder {data_folder}: {str(e)}")

def process_file(file_info, collection, processed_files, force_update: bool = False):
    start_time = time.time()
    file_path = file_info["path"]
    file_type = file_info["type"]
    try:
        logging.info(f"Xử lý file {file_path}")
        
        # Kiểm tra xem file có cần cập nhật không
        if not check_file_needs_update(file_path, processed_files, force_update):
            logging.info(f"File {file_path} không thay đổi, bỏ qua")
            return 0
        
        sections = parse_markdown(file_path) if file_type == "markdown" else parse_json(file_path)
        if not sections:
            logging.warning(f"Không tìm thấy section nào trong {file_path}")
            return 0

        documents = [section['document'] for section in sections]
        metadatas = [section['metadata'] for section in sections]
        ids = [f"{file_type}_{os.path.basename(file_path).replace('.md', '').replace('.json', '')}_section_{i}" 
               for i in range(len(sections))]

        # Nếu file đã thay đổi, xóa tất cả section cũ trước khi thêm mới
        if file_path in processed_files:
            old_section_ids = processed_files[file_path].get("sections", [])
            if old_section_ids:
                logging.info(f"File {file_path} đã thay đổi, xóa {len(old_section_ids)} section cũ")
                remove_existing_sections_from_collection(collection, old_section_ids)
        
        # Xóa thông tin file cũ trong processed_files
        if file_path in processed_files:
            del processed_files[file_path]
        
        documents_to_process = [
            ''.join(c for c in doc if c.isprintable()) if doc and len(doc.strip()) > 0 else "Nội dung trống"
            for doc in documents
        ]
        
        if any(len(doc) > 10000 for doc in documents_to_process):
            logging.warning(f"Có document quá dài trong {file_path}, có thể gây lỗi embedding")

        batch_size = 10
        processed_count = 0
        
        for i in range(0, len(documents_to_process), batch_size):
            batch_end = min(i + batch_size, len(documents_to_process))
            batch_docs = documents_to_process[i:batch_end]
            batch_meta = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            retry_count = 3
            for attempt in range(retry_count):
                try:
                    embeddings = rate_limited_embedding(batch_docs)
                    if not embeddings or all(not emb for emb in embeddings):
                        logging.error(f"Embedding rỗng cho batch {i//batch_size + 1} trong {file_path}")
                        if attempt == retry_count - 1:
                            continue
                    
                    # Kiểm tra xem có ID nào đã tồn tại không trước khi add
                    existing_ids = []
                    for batch_id in batch_ids:
                        try:
                            result = collection.get(ids=[batch_id])
                            if result['ids']:
                                existing_ids.append(batch_id)
                        except:
                            pass
                    
                    if existing_ids:
                        logging.warning(f"Phát hiện {len(existing_ids)} ID đã tồn tại, xóa trước khi thêm mới")
                        remove_existing_sections_from_collection(collection, existing_ids)
                    
                    collection.add(
                        ids=batch_ids,
                        embeddings=embeddings,
                        metadatas=batch_meta,
                        documents=batch_docs
                    )
                    logging.info(f"Đã xử lý batch {i//batch_size + 1} với {len(batch_docs)} section")
                    processed_count += len(batch_docs)
                    
                    # Cập nhật processed_files
                    if file_path not in processed_files:
                        processed_files[file_path] = {"mtime": os.path.getmtime(file_path), "sections": []}
                    processed_files[file_path]["sections"].extend(batch_ids)
                    save_processed_files(processed_files)
                    break
                    
                except Exception as e:
                    logging.error(f"Lỗi batch {i//batch_size + 1} trong {file_path}, thử {attempt + 1}/{retry_count}: {str(e)}")
                    if attempt == retry_count - 1:
                        logging.error(f"Không thể xử lý batch {i//batch_size + 1} sau {retry_count} lần thử")
                        continue
                        
        logging.info(f"Thời gian xử lý file {file_path}: {time.time() - start_time:.4f}s")
        return processed_count
        
    except Exception as e:
        logging.error(f"Lỗi xử lý {file_path}: {str(e)}")
        return 0
        logging.error(f"Lỗi xử lý {file_path}: {str(e)}")
        return 0

def embed_and_process_files(data_folder: str, force_update: bool = False, clear_db: bool = False, clear_folder: bool = False):
    files = get_data_files(data_folder)
    processed_files = load_processed_files()
    if not files:
        logging.info("Không có file nào để xử lý")
        return

    collection = chroma_client.get_or_create_collection(
        name="all_documents",
        embedding_function=embedding_function
    )
    
    # Xóa toàn bộ collection nếu được yêu cầu (NGUY HIỂM!)
    if clear_db:
        clear_collection_if_needed(collection, force_clear=True)
        # Reset processed_files khi clear database
        processed_files = {}
        save_processed_files(processed_files)
    
    # Chỉ xóa dữ liệu của folder hiện tại (AN TOÀN)
    elif clear_folder:
        clear_folder_data_only(collection, data_folder, processed_files)

    total_files = len(files)
    total_sections = 0
    failed_files = 0
    skipped_files = 0
    updated_files = 0

    # Tạo wrapper function để truyền force_update
    def process_file_wrapper(file_info):
        return process_file(file_info, collection, processed_files, force_update)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_file_wrapper, file_info) for file_info in files]
        for future, file_info in zip(futures, files):
            sections_processed = future.result()
            file_path = file_info['path']
            
            if sections_processed == 0:
                # File không thay đổi hoặc không có section
                if check_file_needs_update(file_path, processed_files, force_update):
                    failed_files += 1
                else:
                    skipped_files += 1
            else:
                updated_files += 1
                total_sections += sections_processed

    logging.info(f"""
    Tổng kết:
    - Tổng số file: {total_files}
    - File đã xử lý/cập nhật: {updated_files}
    - File không thay đổi (bỏ qua): {skipped_files}
    - File thất bại: {failed_files}
    - Số section đã thêm/cập nhật: {total_sections}
    """)

def main():
    parser = argparse.ArgumentParser(description="Ingest dữ liệu vào ChromaDB")
    parser.add_argument('--data-folder', type=str, default='data/conduct_and_cybersecurity', help='Thư mục chứa dữ liệu')
    parser.add_argument('--force-update', action='store_true', help='Buộc cập nhật tất cả files bất kể modification time')
    parser.add_argument('--clear-db', action='store_true', help='XÓA TOÀN BỘ database (tất cả folders) trước khi embedding')
    parser.add_argument('--clear-folder', action='store_true', help='Chỉ xóa dữ liệu của folder hiện tại trước khi embedding (AN TOÀN)')
    args = parser.parse_args()

    data_folder = args.data_folder
    force_update = args.force_update
    clear_db = args.clear_db
    clear_folder = args.clear_folder
    
    if clear_db and clear_folder:
        logging.error("Không thể sử dụng cả --clear-db và --clear-folder cùng lúc!")
        return
    
    if clear_db:
        logging.warning("Mode: XÓA TOÀN BỘ DATABASE và embedding lại")
        logging.warning("CẢNH BÁO: Tất cả dữ liệu từ các folders khác sẽ bị mất!")
    elif clear_folder:
        logging.info(f"Mode: Chỉ xóa dữ liệu folder '{data_folder}' và embedding lại")
    elif force_update:
        logging.info("Mode: Buộc cập nhật tất cả files trong folder")
    else:
        logging.info("Mode: Chỉ cập nhật files đã thay đổi")
        
    logging.info(f"Bắt đầu xử lý thư mục: {data_folder}")
    embed_and_process_files(data_folder, force_update, clear_db, clear_folder)
    logging.info("Hoàn thành!")

if __name__ == "__main__":
    main()