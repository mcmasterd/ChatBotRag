# ICTU Scholarship Chatbot

## Cài đặt

1. Clone repository:
   ```bash
   git clone <repository-url>
   cd ChatBotRag

2. Tạo môi trường ảo:
    python3 -m venv venv
    source venv/bin/activate  # Linux/Mac
    # Hoặc: venv\Scripts\activate  # Windows

3. Cài đặt các gói từ requirements.txt(pip freeze > requirements.txt để sửa file)
    pip install -r requirements.txt


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