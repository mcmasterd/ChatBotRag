import os
import chromadb
from dotenv import load_dotenv
import re
import math
from collections import Counter
from typing import List, Dict, Set
from chromadb.utils import embedding_functions
from pprint import pprint  # Để in kết quả dễ nhìn

# Load environment variables
load_dotenv()

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Initialize OpenAI embedding function (model phải giống với quá trình upload dữ liệu)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name="text-embedding-3-small"
)

# Get collection với embedding function tương ứng
collection = chroma_client.get_collection(name="scholarship-qa", embedding_function=openai_ef)

# Define category keywords and mapping
CATEGORY_KEYWORDS = {
    "Scholarship": ["điều kiện", "yêu cầu", "tiêu chuẩn", "tiêu chí", "đủ điều kiện", "đáp ứng", "học bổng", "khuyến khích"],
    "Decree": ["quy trình", "thủ tục", "các bước", "quá trình", "thực hiện", "nghị định", "nộp hồ sơ"],
    "Timeline": ["thời gian", "thời hạn", "khi nào", "hạn cuối", "deadline", "lịch trình"]
}
KEYWORD_TO_CATEGORY = {kw: cat for cat, keywords in CATEGORY_KEYWORDS.items() for kw in keywords}

def detect_categories(query: str) -> Set[str]:
    """Detect categories from query using keyword matching."""
    query_lower = query.lower()
    detected = {KEYWORD_TO_CATEGORY[kw] for kw in KEYWORD_TO_CATEGORY if kw in query_lower}
    if not detected and any(term in query_lower for term in ["học bổng", "scholarship"]):
        detected.add("Scholarship")
    return detected

def build_category_filter(categories: Set[str]):
    """Build a ChromaDB filter based on detected categories."""
    if not categories:
        return None
    return {"$or": [{"category": cat} for cat in categories] + [{"subcategory": cat} for cat in categories]}

def get_relevant_content(query: str, use_categories: bool = True, n_results: int = 10) -> List[Dict]:
    """Retrieve relevant content from ChromaDB with category filtering.
       Lấy n_results kết quả (ở đây n_results=10) và trả về metadata cùng nội dung văn bản.
    """
    categories = detect_categories(query) if use_categories else set()
    filter_dict = build_category_filter(categories)
    
    # Lấy nhiều kết quả để phục vụ BM25 rerank (có thể lấy 10 kết quả trực tiếp)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=filter_dict,
        include=["metadatas", "documents"]
    )
    
    relevant_content = []
    if results['metadatas'] and results['metadatas'][0]:
        for i, metadata in enumerate(results['metadatas'][0]):
            relevant_content.append({
                'question': metadata.get('question', ''),
                'answer': metadata.get('answer', ''),
                'category': metadata.get('category', ''),
                'subcategory': metadata.get('subcategory', ''),
                'document': results['documents'][0][i] if results['documents'] and len(results['documents'][0]) > i else ''
            })
    return relevant_content

# BM25 scoring algorithm for reranking search results
class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        # Tokenize documents
        self.tokenized_docs = [self.tokenize(doc) for doc in documents]
        # Document lengths and average document length
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        # Term frequencies for each document
        self.term_frequencies = [Counter(doc) for doc in self.tokenized_docs]
        # Document frequencies for terms
        self.doc_freqs = Counter()
        for doc in self.tokenized_docs:
            for term in set(doc):
                self.doc_freqs[term] += 1
        # Calculate IDF values for each term
        self.idfs = {term: self._idf(term) for term in self.doc_freqs}
    
    def tokenize(self, text):
        """Tokenize text: loại bỏ ký tự đặc biệt, chuyển thành chữ thường, và tách từ theo khoảng trắng."""
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def _idf(self, term):
        """Tính IDF cho term."""
        return math.log((len(self.tokenized_docs) - self.doc_freqs[term] + 0.5) / (self.doc_freqs[term] + 0.5) + 1.0)
    
    def get_scores(self, query):
        """Tính toán BM25 scores cho query trên tất cả các documents."""
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

if __name__ == "__main__":
    query = ("Sinh viên tại Trường Đại học Công nghệ Thông tin và Truyền thông có thể phản ánh "
             "thắc mắc về danh sách học bổng khuyến khích học tập trong thời gian bao lâu và theo cách nào?")
    
    # Lấy 10 kết quả từ DB
    relevant_data = get_relevant_content(query, use_categories=True, n_results=10)
    
    print("==== Dữ liệu tìm được (trước BM25 rerank) ====")
    pprint(relevant_data)
    
    # Nếu có dữ liệu, tính BM25 score và sắp xếp
    if relevant_data:
        documents = [item['document'] for item in relevant_data]
        bm25 = BM25(documents)
        bm25_scores = bm25.get_scores(query)
        for i, score in enumerate(bm25_scores):
            relevant_data[i]['bm25_score'] = score
        
        # Sắp xếp theo BM25 score giảm dần
        relevant_data.sort(key=lambda x: x.get('bm25_score', 0), reverse=True)
        
        # Chọn ra 3 kết quả cao nhất
        top3 = relevant_data[:3]
        
        print("\n==== Top 3 kết quả sau BM25 rerank ====")
        for idx, item in enumerate(top3, 1):
            print(f"\n--- Kết quả {idx} ---")
            print(f"Question : {item.get('question', '')}")
            print(f"Answer   : {item.get('answer', '')}")
            print(f"Category : {item.get('category', '')} | Subcategory: {item.get('subcategory', '')}")
            print(f"BM25 Score: {item.get('bm25_score', 0):.4f}")
            print(f"Document : {item.get('document', '')}")
    else:
        print("Không tìm thấy dữ liệu liên quan.")
