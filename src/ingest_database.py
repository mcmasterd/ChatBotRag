import os
import json
from uuid import uuid4
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Sử dụng đường dẫn tuyệt đối đến file JSON
JSON_FILE_PATH = r"D:\Python\ChatBotICTU\data\3. Thong-tu-08-2021-tt-bgddt-quy-che-dao-tao-trinh-do-dai-hoc.json"

# Đường dẫn thư mục lưu trữ Chroma
CHROMA_PATH = r"D:\Python\ChatBotICTU\chroma_db"

# Khởi tạo mô hình embeddings với OpenAI
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Khởi tạo vector store với Chroma
vector_store = Chroma(
    collection_name="qa_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Đọc file JSON chứa các cặp Q&A
with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

qa_pairs = data.get("qa_pairs", [])

# Chuyển mỗi cặp Q&A thành một đối tượng Document
documents = [
    Document(page_content=f"Question: {item.get('question', '').strip()}\nAnswer: {item.get('answer', '').strip()}")
    for item in qa_pairs
]

# Chia nhỏ văn bản thành các đoạn nhỏ nếu cần
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(documents)

# Tạo UUID cho mỗi đoạn văn bản
uuids = [str(uuid4()) for _ in range(len(chunks))]

# Thêm các đoạn đã tạo embeddings vào vector store
vector_store.add_documents(documents=chunks, ids=uuids)

print("Quá trình ingest dữ liệu hoàn tất!")
