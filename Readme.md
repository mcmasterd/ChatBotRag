# ICTU Scholarship Chatbot

## Giới thiệu
Chatbot hỏi đáp học bổng ICTU, sử dụng Flask, ChromaDB, Redis, OpenAI, Google Sheets API.

## Cài đặt nhanh

1. Clone repository:
   ```bash
   git clone <repository-url>
   cd ChatBotRag
   ```
2. Tạo môi trường ảo và cài requirements:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Tạo file `.env` với nội dung:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_SHEET_ID=your_google_sheet_id_here
   GOOGLE_SHEET_WORKSHEET=Sheet1
   GOOGLE_CREDENTIALS_FILE=k-project-456412-29ad68606a96.json
   ```
   > **Lưu ý:** Không commit file `.env` và file credentials `.json` lên git!

4. Các bước cài đặt, ingest dữ liệu, chạy backend, frontend, cấu hình Google Sheets, xem chi tiết trong [docs/setup.md](docs/setup.md)

## Cấu hình Nginx, Systemd, ...
(Phần này giữ nguyên như cũ, xem hướng dẫn chi tiết bên dưới)

4. Tạo file .env: OPENAI_API_KEY=your_openai_api_key_here

5. Cài đặt Nginx và cấu hình:
- sudo apt install nginx
- Sao chép cấu hình vào /etc/nginx/sites-available/default
- Di chuyển index.html vào /var/www/html:
    sudo mv /path/to/ChatBotRag/src/index.html /var/www/html/index.html
    sudo chmod 644 /var/www/html/index.html
    sudo chown www-data:www-data /var/www/html/index.html
- Cấu hình Nginx
server {
    listen 80 default_server;
    listen [::]:80 default_server;

    root /var/www/html;
    index index.html;

    server_name 192.168.1.129 27.72.246.67;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /ask {
        proxy_pass http://127.0.0.1:1508/ask;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 90s;
        proxy_connect_timeout 90s;
        proxy_send_timeout 90s;
    }

    location /get_user_id {
        proxy_pass http://127.0.0.1:1508/get_user_id;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 90s;
        proxy_connect_timeout 90s;
        proxy_send_timeout 90s;
    }

    location /get_session_history {
        proxy_pass http://127.0.0.1:1508/get_session_history;
proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 90s;
        proxy_connect_timeout 90s;
        proxy_send_timeout 90s;
}
}
server {
    listen 7860;
    listen [::]:7860;

    root /var/www/html;
    index index.html;

    server_name 192.168.1.129 27.72.246.67;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /ask {
        proxy_pass http://127.0.0.1:1508/ask;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 90s;
        proxy_connect_timeout 90s;
        proxy_send_timeout 90s;
    }
location /get_user_id {
        proxy_pass http://127.0.0.1:1508/get_user_id;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 90s;
        proxy_connect_timeout 90s;
        proxy_send_timeout 90s;
    }
    location /get_session_history {
        proxy_pass http://127.0.0.1:1508/get_session_history;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 90s;
        proxy_connect_timeout 90s;
        proxy_send_timeout 90s;
    }
}

- Khởi động lại nginx

6. Cấu hình Systemd service
- Tạo file chatbot.service, cập nhật đường dẫn (User, WorkingDirectory, Environment, ExecStart) theo máy mới:
[Unit]
Description=Gradio Chatbot Service
After=network.target

[Service]
User=skynet1
Group=skynet1
WorkingDirectory=/home/skynet1/ChatBotRag/src
Environment="PATH=/home/skynet1/ChatBotRag/venv/bin"
ExecStart=/home/skynet1/ChatBotRag/venv/bin/python3 /home/skynet1/ChatBotRag/src/chatbot.py
Restart=always

[Install]
WantedBy=multi-user.target
- Cài đặt dịch vụ:
    sudo systemctl daemon-reload
    sudo systemctl enable chatbot.service
    sudo systemctl start chatbot.service

## Lưu ý bảo mật
- Không commit file `.env`, file credentials Google, hoặc dữ liệu nhạy cảm lên git.
- Đảm bảo các file nhạy cảm đã được thêm vào `.gitignore`.