import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from chatbot import classify_categories_llm, get_relevant_content

if __name__ == "__main__":
    print("=== Kiểm tra nhận diện ý định và tìm kiếm ===")
    while True:
        query = input("Nhập câu hỏi (hoặc 'exit' để thoát): ").strip()
        if query.lower() == 'exit':
            break
        # Phân loại category
        categories = classify_categories_llm(query)
        print(f"\n[Category LLM phân loại]: {categories}")
        # Tìm kiếm vector
        results = get_relevant_content(query, categories=categories, final_results=4)
        print("\n[Kết quả tìm kiếm]:")
        if not results:
            print("Không tìm thấy kết quả phù hợp.")
        else:
            for i, item in enumerate(results, 1):
                doc = item.get('document', '')
                meta = item.get('metadata', {})
                print(f"--- Kết quả {i} ---")
                print(f"Nội dung: {doc[:300]}{'...' if len(doc) > 300 else ''}")
                print(f"Metadata: {meta}")
                print() 