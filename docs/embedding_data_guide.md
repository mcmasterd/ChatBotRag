# Hướng dẫn quản lý dữ liệu Embedding

## Tổng quan
Dự án này bao gồm dữ liệu đã được embedding trong folder `chroma_db/` để tăng tốc độ truy vấn và giảm thời gian xử lý.

## Cấu trúc dữ liệu
```
chroma_db/
├── chroma.sqlite3          # Database chính chứa metadata và index
└── 921b0ff1-fb1a-4f29-b003-ec25d9ec71dc/
    ├── data_level0.bin     # Dữ liệu vector embedding (3.1MB)
    ├── header.bin          # Header metadata
    ├── length.bin          # Thông tin độ dài
    └── link_lists.bin      # Danh sách liên kết
```

## Lưu ý quan trọng

### ✅ Nên commit lên GitHub khi:
- Dự án có quy mô nhỏ (dữ liệu < 100MB)
- Cần chia sẻ dữ liệu embedding với team
- Muốn giảm thời gian setup cho người dùng mới
- Dữ liệu không chứa thông tin nhạy cảm

### ⚠️ Không nên commit khi:
- Dữ liệu quá lớn (> 100MB)
- Chứa thông tin nhạy cảm
- Cần bảo mật cao
- Repository công khai với dữ liệu riêng tư

## Cách quản lý

### Thêm dữ liệu embedding vào Git:
```bash
# 1. Chỉnh sửa .gitignore (đã thực hiện)
# Comment out dòng: # chroma_db/

# 2. Thêm vào staging
git add chroma_db/

# 3. Commit
git commit -m "Add embedded data"

# 4. Push lên GitHub
git push origin main
```

### Loại bỏ dữ liệu embedding khỏi Git:
```bash
# 1. Xóa khỏi Git tracking (giữ lại file local)
git rm -r --cached chroma_db/

# 2. Thêm lại vào .gitignore
echo "chroma_db/" >> .gitignore

# 3. Commit thay đổi
git commit -m "Remove embedded data from tracking"
```

## Tái tạo dữ liệu embedding
Nếu không có dữ liệu embedding, chạy script để tạo lại:
```bash
python src/ingest_database.py
```

## Kích thước dữ liệu hiện tại
- `chroma.sqlite3`: ~8.3MB
- `data_level0.bin`: ~3.1MB
- Tổng cộng: ~11.4MB

## Lời khuyên
- Định kỳ kiểm tra kích thước repository
- Sử dụng Git LFS nếu dữ liệu > 50MB
- Backup dữ liệu embedding riêng biệt
- Cập nhật dữ liệu khi có thay đổi trong source documents 