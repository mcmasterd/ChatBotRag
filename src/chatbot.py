import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from dotenv import load_dotenv

# Nạp biến môi trường từ file .env ở PROJECT_ROOT
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Đường dẫn đến thư mục chroma_db (nằm trong PROJECT_ROOT)
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

# Khởi tạo mô hình embeddings với OpenAI
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Khởi tạo mô hình chat (sử dụng model 'gpt-4o-mini' với nhiệt độ 0.5)
llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")

# Kết nối đến cơ sở dữ liệu vector Chroma
vector_store = Chroma(
    collection_name="qa_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Thiết lập retriever để lấy ra 5 tài liệu liên quan nhất
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={"k": num_results})

def stream_response(message, history):
    """
    Xử lý tin nhắn từ người dùng:
      1. Lấy các tài liệu (Document) liên quan dựa trên câu hỏi.
      2. Gom nội dung các tài liệu đó thành chuỗi 'knowledge'.
      3. Tạo prompt dạng RAG và gọi LLM (streaming) trả về kết quả.
    """
    docs = retriever.get_relevant_documents(message)
    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content + "\n\n"
    
    rag_prompt = f"""
You are an assistant that answers questions solely based on the provided knowledge.
Do not use any external information or mention that you are using provided knowledge.

The question: {message}

Conversation history: {history}

The knowledge: {knowledge}
    """
    partial_message = ""
    for response in llm.stream(rag_prompt):
        partial_message += response.content
        yield partial_message

# Khởi tạo giao diện chat của Gradio
chatbot = gr.ChatInterface(
    fn=stream_response,
    textbox=gr.Textbox(
        placeholder="Send to the LLM...",
        container=False,
        autoscroll=True,
        scale=7,
    )
)

if __name__ == "__main__":
    # Khởi chạy ứng dụng Gradio, lắng nghe trên tất cả các địa chỉ IP (cho truy cập từ xa)
    chatbot.launch(server_name="0.0.0.0", server_port=7860)
