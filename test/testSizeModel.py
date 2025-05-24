from sentence_transformers import SentenceTransformer
model = SentenceTransformer('NghiemAbe/Vi-Legal-Bi-Encoder-v2')
embedding = model.encode(["test"]).tolist()[0]
print(f"Kích thước embedding: {len(embedding)}")