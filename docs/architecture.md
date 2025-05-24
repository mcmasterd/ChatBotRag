# Kiến trúc tổng thể dự án ICTU Scholarship Chatbot

## Sơ đồ tổng quan

```
+-------------------+         +-------------------+         +-------------------+
|   Trình duyệt     | <-----> |      Flask API    | <-----> |    ChromaDB       |
|  (index2.html)    |         |   (chatbot.py)    |         | (Vector Database) |
+-------------------+         +-------------------+         +-------------------+
         |                            |                             |
         |                            |                             |
         |                            v                             |
         |                  +-------------------+                   |
         |                  |     Redis Cache   |                   |
         |                  +-------------------+                   |
         |                            |                             |
         |                            v                             |
         |                  +-------------------+                   |
         |                  |   Google Sheets   |                   |
         |                  +-------------------+                   |
         |                            ^                             |
         |                            |                             |
         |                  +-------------------+                   |
         |                  |  Logging (CSV)    |                   |
         |                  +-------------------+                   |
         |                            ^                             |
         |                            |                             |
         |                  +-------------------+                   |
         |                  |  Ingest Pipeline  |                   |
         |                  | (ingest_database) |                   |
         |                  +-------------------+                   |
```

## Mô tả các thành phần

### 1. Frontend (src/index2.html)
- Giao diện người dùng hiện đại, responsive, sử dụng HTML/CSS/JS thuần.
- Giao tiếp với backend qua các endpoint RESTful (`/ask`, `/get_user_id`, `/get_session_history`, ...).
- Hiển thị lịch sử, đánh giá, các phiên trò chuyện, và các câu hỏi liên quan.

### 2. Backend (src/chatbot.py)
- Sử dụng Flask để cung cấp API cho frontend.
- Xử lý truy vấn người dùng, truy xuất dữ liệu từ ChromaDB, cache với Redis, và gọi OpenAI API để sinh câu trả lời.
- Lưu lịch sử, đánh giá, và các thông tin liên quan vào file log CSV và Redis.
- Cung cấp các endpoint quản lý phiên, đánh giá, và truy xuất lịch sử.

### 3. Pipeline ingest dữ liệu (src/ingest_database.py)
- Đọc dữ liệu từ thư mục `data/` (file .md, .json), sinh embedding bằng mô hình local (SentenceTransformer), lưu vào ChromaDB.
- Hỗ trợ cập nhật, kiểm tra trạng thái ingest, và xử lý dữ liệu lớn theo batch.

### 4. Lưu trữ
- **ChromaDB**: Lưu trữ vector embedding cho tìm kiếm ngữ nghĩa.
- **Redis**: Cache kết quả truy vấn, lưu session, tên phiên, và các thông tin tạm thời.
- **logs/qa_log.csv**: Lưu lịch sử hỏi đáp, đánh giá, feedback.

### 5. Google Sheets API (src/update_google_sheet.py, src/sync_service.py)
- Tự động đồng bộ log hỏi đáp và đánh giá lên Google Sheets để quản trị viên theo dõi.
- Sử dụng service account và credentials riêng.

### 6. Logging & Monitoring
- Ghi log chi tiết quá trình hỏi đáp, ingest, đồng bộ, và lỗi vào file log và Google Sheets.

### 7. Các script & kiểm thử
- Thư mục `test/`: Chứa các script kiểm thử vector search, API, và pipeline ingest.

### 8. Cấu hình ngoài (không nằm trong repo)
- **Nginx**: Reverse proxy cho Flask app và tĩnh frontend.
- **Systemd service**: Quản lý tiến trình chatbot và sync service.

## Luồng hoạt động chính
1. Người dùng truy cập giao diện web, gửi câu hỏi.
2. Frontend gọi API `/ask` với user_id và câu hỏi.
3. Backend kiểm tra cache, truy vấn ChromaDB, sinh câu trả lời bằng OpenAI, trả về frontend.
4. Lịch sử, đánh giá, feedback được lưu vào Redis, file log, và đồng bộ lên Google Sheets.
5. Pipeline ingest cho phép cập nhật dữ liệu tri thức mới vào ChromaDB. 