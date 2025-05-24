import chromadb
import re
from collections import Counter
import math
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import os
import time

# Đường dẫn ChromaDB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "..", "chroma_db")  # Điều chỉnh để truy cập đúng thư mục

# Khởi tạo ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Hàm embedding
class LocalEmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer('NghiemAbe/Vi-Legal-Bi-Encoder-v2')
        self.model.to('cpu')
        print("Đã khởi tạo mô hình SentenceTransformer cho embedding")

    def __call__(self, input: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(input, convert_to_numpy=True).tolist()
            return embeddings
        except Exception as e:
            print(f"Lỗi khi tạo embedding: {str(e)}")
            return [[0.0] * 768 for _ in input]  # Giả định 768 chiều

embedding_function = LocalEmbeddingFunction()

# Kết nối với collection
collection = None
try:
    collection = chroma_client.get_collection(name="scholarship_documents", embedding_function=embedding_function)
    print("Đã tải thành công collection 'scholarship_documents'.")
except chromadb.errors.InvalidCollectionException:
    print("Lỗi: Collection 'scholarship_documents' không tồn tại. Vui lòng chạy ingest_database.py để khởi tạo.")

# Hàm BM25
class BM25:
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
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def _idf(self, term):
        return math.log((len(self.tokenized_docs) - self.doc_freqs[term] + 0.5) / 
                        (self.doc_freqs[term] + 0.5) + 1.0)
    
    def get_scores(self, query):
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

# Hàm in kết quả
def print_search_results(query: str, initial_results: List[Dict], reranked_results: List[Dict]):
    print("\n=== Kết Quả Tìm Kiếm Vector ===")
    print(f"Câu hỏi: {query}")
    print(f"Tổng số kết quả ban đầu: {len(initial_results)}")
    for i, item in enumerate(initial_results, 1):
        document = item['document'][:100] + "..." if len(item['document']) > 100 else item['document']
        metadata = item['metadata']
        source = metadata.get('source', 'Không rõ nguồn')
        doc_id = metadata.get('doc_id', 'Không rõ ID')
        print(f"\nKết quả {i}:")
        print(f"Nội dung: {document}")
        print(f"Nguồn: {source}")
        print(f"ID tài liệu: {doc_id}")
        print(f"Điểm liên quan (Ban đầu): {item['relevance_score']:.4f}")

    print("\n=== Kết Quả Sau Xếp Hạng Lại (BM25) ===")
    print(f"Tổng số kết quả sau xếp hạng: {len(reranked_results)}")
    for i, item in enumerate(reranked_results, 1):
        document = item['document'][:100] + "..." if len(item['document']) > 100 else item['document']
        metadata = item['metadata']
        source = metadata.get('source', 'Không rõ nguồn')
        doc_id = metadata.get('doc_id', 'Không rõ ID')
        print(f"\nKết quả {i}:")
        print(f"Nội dung: {document}")
        print(f"Nguồn: {source}")
        print(f"ID tài liệu: {doc_id}")
        print(f"Điểm liên quan (BM25): {item['relevance_score']:.4f}")
start = time.time()
# Hàm lấy nội dung liên quan
def get_relevant_content(query: str, final_results: int = 4) -> List[Dict]:
    if collection is None:
        print("Lỗi: Collection 'scholarship_documents' không khả dụng.")
        return []
    
    try:
        initial_results = 10
        results = collection.query(
            query_texts=[query],
            n_results=initial_results,
            include=["metadatas", "documents"]
        )
        
        candidate_content = []
        if results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                document_text = results['documents'][0][i] if results['documents'] and results['documents'][0] else ''
                candidate_content.append({
                    'document': document_text,
                    'metadata': metadata,
                    'relevance_score': 0.0
                })
        
        if candidate_content:
            print(f"Đang xếp hạng lại {len(candidate_content)} kết quả ban đầu...")
            documents = [item['document'] for item in candidate_content]
            bm25 = BM25(documents)
            bm25_scores = bm25.get_scores(query)
            
            print_search_results(query, candidate_content, candidate_content)
            
            for i, score in enumerate(bm25_scores):
                candidate_content[i]['relevance_score'] = score
            
            candidate_content.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            final_content = candidate_content[:final_results]
            
            print(f"Tổng thời gian truy vấn: {time.time() - start:.4f}s, Kết quả cuối cùng: {len(final_content)}")
            return final_content
        else:
            print("Không tìm thấy kết quả ban đầu.")
            return []
    except Exception as e:
        print(f"Lỗi khi truy vấn: {str(e)}")
        return []

# Hàm chính
def main():
    while True:
        query = input("\nNhập câu hỏi (hoặc 'thoát' để dừng): ")
        if query.lower() == 'thoát':
            print("Đã dừng chương trình.")
            break
        if not query.strip():
            print("Vui lòng nhập câu hỏi hợp lệ.")
            continue
        
        
        results = get_relevant_content(query)
        if results:
            print("\n=== Kết Quả Cuối Cùng ===")
            for i, item in enumerate(results, 1):
                document = item['document'][:100] + "..." if len(item['document']) > 100 else item['document']
                print(f"Kết quả {i}: {document}")
        else:
            print("Không tìm thấy kết quả phù hợp.")
        print(f"Tổng thời gian xử lý: {time.time() - start:.4f}s")

if __name__ == "__main__":
    main()