# Hướng dẫn cài đặt và triển khai ICTU Scholarship Chatbot

## 1. Chuẩn bị môi trường
- Python >= 3.10
- pip, venv
- Git
- Redis server (mặc định chạy local, port 6379, password: terminator)
- ChromaDB (sẽ tự tạo thư mục chroma_db/ khi ingest)

## 2. Cài đặt mã nguồn
```bash
git clone <repository-url>
cd ChatBotRag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Cấu hình biến môi trường
Tạo file `.env` trong thư mục gốc với nội dung:
```
OPENAI_API_KEY=your_openai_api_key_here 
GOOGLE_SHEET_ID=your_google_sheet_id_here                    # <- Thay bằng ID thật từ URL Google Sheet
GOOGLE_SHEET_WORKSHEET=Sheet1                                # <- Tên worksheet bạn muốn ghi vào
GOOGLE_CREDENTIALS_FILE=k-project-456412-29ad68606a96.json   # <- Tên file credentials JSON của bạn (nằm trong thư mục gốc dự án)
```

## 4. Khởi tạo dữ liệu tri thức (ChromaDB)
- Đảm bảo các file dữ liệu .md, .json nằm trong thư mục `data/`
- Chạy ingest:
```bash
python src/ingest_database.py
```
- Sau khi chạy xong sẽ có thư mục `chroma_db/` chứa dữ liệu vector hóa.

## 5. Chạy backend Flask
```bash
python src/chatbot.py
```
- Mặc định chạy ở port 1508, có thể đổi trong code.

## 6. Chạy giao diện web
- Mở file `src/index2.html` trên trình duyệt (hoặc cấu hình Nginx để phục vụ file này).

## 7. Cấu hình Google Sheets API
- Tạo Google Service Account, tải file credentials JSON về thư mục gốc dự án (ví dụ: `k-project-456412-29ad68606a96.json`).
- Cập nhật các biến SHEET_ID, WORKSHEET_NAME, CREDENTIALS_FILE trong `src/update_google_sheet.py` cho phù hợp.

## 8. Đồng bộ log lên Google Sheets
- Chạy dịch vụ đồng bộ:
```bash
python src/sync_service.py
```
- Có thể cấu hình chạy tự động bằng systemd service.

## 9. Cấu hình Nginx (reverse proxy)
- Tham khảo file cấu hình mẫu trong Readme.md.
- Đảm bảo các endpoint `/ask`, `/get_user_id`, `/get_session_history` được proxy về Flask backend.

## 10. Chạy production (khuyến nghị)
- Sử dụng systemd để quản lý tiến trình Flask backend và sync_service.
- Đảm bảo quyền truy cập file, thư mục logs/, chroma_db/, data/.

## 11. Lưu ý
- Không commit file .env, credentials Google, hoặc dữ liệu nhạy cảm lên git.
- Đảm bảo Redis, ChromaDB, Google Sheets đều hoạt động ổn định trước khi chạy production. 