import chromadb

# Khởi tạo ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Lấy danh sách tên collection
collections = chroma_client.list_collections()
if not collections:
    print("Không tìm thấy collection nào trong ChromaDB.")
else:
    print("Các collection hiện có:")
    for name in collections:
        print(f"Collection: {name}")