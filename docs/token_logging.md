# 📊 Token Logging System

## Tổng quan

Hệ thống log token chi tiết đã được tích hợp vào chatbot ICTU để theo dõi và phân tích việc sử dụng API OpenAI và các thao tác khác.

## 📁 File Log

File log được lưu tại: `/logs/token_usage_log.csv`

### Cấu trúc Columns:

| Column | Mô tả |
|--------|-------|
| `Timestamp` | Thời gian thực hiện (múi giờ Hà Nội) |
| `User_ID` | ID người dùng hoặc "system" |
| `Operation_Type` | Loại thao tác (xem bên dưới) |
| `Model` | Model được sử dụng |
| `Input_Tokens` | Số token đầu vào |
| `Output_Tokens` | Số token đầu ra |
| `Total_Tokens` | Tổng số token |
| `Input_Text_Preview` | Preview 100 ký tự đầu của input |
| `Output_Text_Preview` | Preview 100 ký tự đầu của output |
| `Cost_USD` | Chi phí ước tính (USD) |
| `Processing_Time_Seconds` | Thời gian xử lý (giây) |
| `Additional_Info` | Thông tin bổ sung |

## 🔍 Các Operation Types được Log

### 1. **Category Classification**
- `operation_type`: `category_classification`
- `model`: `gpt-4o`
- **Mô tả**: Phân loại câu hỏi vào categories
- **Input**: System prompt + user prompt
- **Output**: JSON response với categories và rephrased_query

### 2. **Embedding Generation**
- `operation_type`: `embedding_generation`
- `model`: `local_embedding`
- **Mô tả**: Tạo vector embeddings cho text
- **Input**: Các text cần embedding
- **Output**: Thông tin về số embeddings được tạo

### 3. **Vector Search Results**
- `operation_type`: `vector_search_results`
- `model`: `retrieval`
- **Mô tả**: Kết quả tìm kiếm vector từ ChromaDB
- **Input**: Query để search
- **Output**: Context được truy xuất
- **Additional Info**: Số kết quả, categories

### 4. **Session History**
- `operation_type`: `session_history`
- `model`: `gpt-4o-mini`
- **Mô tả**: Lịch sử hội thoại được sử dụng
- **Input**: Context từ Redis session
- **Output**: Rỗng
- **Additional Info**: Số items trong history

### 5. **Final Prompt**
- `operation_type`: `final_prompt`
- `model`: `gpt-4o-mini`
- **Mô tả**: Prompt cuối cùng gửi đến LLM
- **Input**: Prompt hoàn chỉnh (context + query)
- **Output**: Rỗng
- **Additional Info**: Sources được sử dụng

### 6. **Main Response**
- `operation_type`: `main_response` / `main_streaming`
- `model`: `gpt-4o-mini`
- **Mô tả**: Response chính từ LLM
- **Input**: System message + user prompt
- **Output**: Response từ LLM

### 7. **Small Talk**
- `operation_type`: `small_talk_response` / `small_talk_streaming`
- `model`: `gpt-4o-mini`
- **Mô tả**: Response cho câu hỏi xã giao
- **Input**: Prompt đơn giản
- **Output**: Response xã giao

### 8. **Cache Operations**
- `operation_type`: `cache_hit`
- `model`: `cache`
- **Mô tả**: Khi response được lấy từ cache
- **Cost**: $0.000000 (miễn phí)

## 📈 Phân tích Token Usage

### Xem tổng quan token usage:
```bash
# Xem 10 dòng cuối
tail -10 /home/skynet1/ChatBotRag/logs/token_usage_log.csv

# Đếm số lượng operations
cut -d',' -f3 /home/skynet1/ChatBotRag/logs/token_usage_log.csv | sort | uniq -c

# Tính tổng chi phí
awk -F',' 'NR>1 {sum+=$10} END {printf "Total cost: $%.6f\n", sum}' /home/skynet1/ChatBotRag/logs/token_usage_log.csv
```

### Phân tích theo model:
```bash
# Tokens theo model
awk -F',' 'NR>1 {model[$4]+=$7} END {for(m in model) printf "%s: %d tokens\n", m, model[m]}' /home/skynet1/ChatBotRag/logs/token_usage_log.csv
```

### Phân tích theo operation:
```bash
# Tokens theo operation type
awk -F',' 'NR>1 {op[$3]+=$7} END {for(o in op) printf "%s: %d tokens\n", o, op[o]}' /home/skynet1/ChatBotRag/logs/token_usage_log.csv
```

## 💰 Chi phí ước tính

| Model | Input ($/1K tokens) | Output ($/1K tokens) |
|-------|---------------------|----------------------|
| gpt-4o | $0.0025 | $0.01 |
| gpt-4o-mini | $0.00015 | $0.0006 |
| local_embedding | $0.0001 | $0 (free) |

## 🔧 Cấu hình

### Vô hiệu hóa token logging:
Để tắt token logging, comment dòng `log_token_usage()` trong các hàm.

### Thay đổi pricing:
Cập nhật dict `TOKEN_PRICING` trong `chatbot.py`.

## 📊 Dashboard Recommendations

Có thể tích hợp với:
- **Grafana**: Visualize token usage theo thời gian
- **Jupyter Notebook**: Phân tích chi tiết
- **Google Sheets**: Sync log để review

## 🚨 Monitoring Alerts

Thiết lập alerts khi:
- Token usage > ngưỡng mỗi giờ
- Chi phí hàng ngày > budget
- Error rate cao trong token counting

## ✅ Test Đã Thực hiện

```bash
# Chạy test
python3 test_token_simple.py
```

Test bao gồm:
- ✅ Token counting với tiktoken
- ✅ Log các operation types
- ✅ Tính toán chi phí chính xác
- ✅ CSV format hợp lệ
- ✅ Timezone Hà Nội
