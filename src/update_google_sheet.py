import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path
import os
import numpy as np
from dotenv import load_dotenv

# ====== Cấu hình ======
SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
WORKSHEET_NAME = os.environ.get("GOOGLE_SHEET_WORKSHEET")
CREDENTIALS_FILE = os.environ.get("GOOGLE_CREDENTIALS_FILE")

# Kiểm tra biến môi trường
if not SHEET_ID:
    raise ValueError("Chưa cấu hình biến môi trường GOOGLE_SHEET_ID")
if not CREDENTIALS_FILE:
    raise ValueError("Chưa cấu hình biến môi trường GOOGLE_CREDENTIALS_FILE")

# ====== Thiết lập đường dẫn ======
BASE_DIR = Path(__file__).resolve().parent.parent
qa_log_path = BASE_DIR / "logs" / "qa_log.csv"

# Đảm bảo đường dẫn chính xác
print(f"Đang sử dụng đường dẫn: {qa_log_path}")
if not os.path.exists(qa_log_path):
    raise FileNotFoundError(f"File không tồn tại: {qa_log_path}")

creds_path = BASE_DIR / CREDENTIALS_FILE
if not os.path.exists(creds_path):
    raise FileNotFoundError(f"File credentials không tồn tại: {creds_path}")

# ====== Kết nối Google Sheets ======
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(str(creds_path), scope)
client = gspread.authorize(creds)

# ====== Mở Google Sheet ======
sh = client.open_by_key(SHEET_ID)
sheet = sh.worksheet(WORKSHEET_NAME)

# ====== Đọc dữ liệu log ======
try:
    df_local = pd.read_csv(qa_log_path)
    print(f"Đã đọc thành công file log với {len(df_local)} dòng")
except Exception as e:
    print(f"Lỗi khi đọc file CSV: {str(e)}")
    raise

# ====== Làm sạch NaN và Inf để tránh lỗi JSON ======
df_local = df_local.replace([float('inf'), float('-inf')], '')
df_local = df_local.fillna('')

# ====== Lấy dữ liệu hiện tại từ Google Sheet ======
sheet_data = sheet.get_all_values()
original_headers = sheet_data[0] if sheet_data else []

# Chuẩn hóa headers bằng cách bỏ khoảng trắng và chuyển sang lowercase để so sánh dễ dàng hơn
headers = [h.strip().lower().replace(' ', '_') for h in original_headers] if original_headers else []
local_headers = [h.strip().lower().replace(' ', '_') for h in df_local.columns]

# In thông tin debug về headers
print(f"Headers trong CSV local: {list(df_local.columns)}")
print(f"Headers trong Google Sheet (gốc): {original_headers}")
print(f"Headers trong Google Sheet (chuẩn hóa): {headers}")

# Tạo mapping giữa tên cột chuẩn hóa và tên cột gốc trong Google Sheet
header_mapping = {headers[i]: original_headers[i] for i in range(len(headers))} if headers else {}

existing_rows = len(sheet_data)
print(f"Sheet hiện tại có {existing_rows} dòng")

if existing_rows <= 1:  # Chỉ có header hoặc không có dữ liệu
    # ====== Xóa tất cả dữ liệu cũ và tải lại ======
    print("Sheet trống, đang tải toàn bộ dữ liệu lên...")
    if existing_rows > 0:  # Có header, giữ lại
        # Sử dụng tham số có tên để tránh cảnh báo
        sheet.update(values=df_local.values.tolist(), range_name='A2:Z')
    else:  # Không có header, thêm cả header
        all_data = [df_local.columns.tolist()] + df_local.values.tolist()
        # Sử dụng tham số có tên để tránh cảnh báo
        sheet.update(values=all_data, range_name='A1:Z')
    print(f"✅ Đã tải {len(df_local)} dòng lên Google Sheet.")
else:
    # ====== Đọc dữ liệu từ Google Sheet thành DataFrame ======
    try:
        df_sheet = pd.DataFrame(sheet_data[1:], columns=original_headers)
        
        # Xác định các cột ID để so sánh (Conversation_ID, User_ID, Question, Answer)
        id_columns_local = ['Conversation_ID', 'User_ID', 'Question', 'Answer']
        
        # Tìm tên tương ứng của cột trong sheet
        id_columns_sheet = []
        for col in id_columns_local:
            col_lower = col.strip().lower().replace(' ', '_')
            for sheet_col in original_headers:
                if sheet_col.strip().lower().replace(' ', '_') == col_lower:
                    id_columns_sheet.append(sheet_col)
                    break
        
        print(f"Các cột ID trong local: {id_columns_local}")
        print(f"Các cột ID trong sheet: {id_columns_sheet}")
        
        # Tìm cột Rating và Feedback trong sheet
        rating_col = None
        feedback_col = None
        
        for col in original_headers:
            col_lower = col.strip().lower().replace(' ', '_')
            if col_lower == 'rating':
                rating_col = col
            elif col_lower == 'feedback':
                feedback_col = col
        
        print(f"Cột Rating trong sheet: {rating_col}")
        print(f"Cột Feedback trong sheet: {feedback_col}")
        
        # Đánh dấu những dòng cần cập nhật (có trong sheet và trong file local)
        rows_to_update = []
        for i, local_row in df_local.iterrows():
            for j, sheet_row in df_sheet.iterrows():
                # So sánh các cột ID
                match = True
                for local_col, sheet_col in zip(id_columns_local, id_columns_sheet):
                    if local_col not in local_row or sheet_col not in sheet_row:
                        match = False
                        break
                    
                    local_val = str(local_row[local_col]).strip()
                    sheet_val = str(sheet_row[sheet_col]).strip()
                    
                    if local_val != sheet_val:
                        match = False
                        break
                
                if match:
                    # So sánh các trường khác (Rating, Feedback)
                    diff = False
                    
                    # Lấy giá trị Rating từ cả hai nguồn
                    local_rating = str(local_row['Rating']).strip() if 'Rating' in local_row else ""
                    sheet_rating = str(sheet_row[rating_col]).strip() if rating_col and rating_col in sheet_row else ""
                    
                    # Lấy giá trị Feedback từ cả hai nguồn
                    local_feedback = str(local_row['Feedback']).strip() if 'Feedback' in local_row else ""
                    sheet_feedback = str(sheet_row[feedback_col]).strip() if feedback_col and feedback_col in sheet_row else ""
                    
                    # So sánh
                    if local_rating != sheet_rating or local_feedback != sheet_feedback:
                        diff = True
                        print(f"Phát hiện sự khác biệt giữa local và sheet cho dòng {j+2}:")
                        print(f"  Rating: local='{local_rating}', sheet='{sheet_rating}'")
                        print(f"  Feedback: local='{local_feedback}', sheet='{sheet_feedback}'")
                    
                    if diff:
                        # Cập nhật dòng j+2 (header là 1, dữ liệu bắt đầu từ 2)
                        sheet_row_idx = j + 2
                        rows_to_update.append((sheet_row_idx, local_row))
                    break
        
        # Tìm các dòng mới (không có trong sheet)
        ids_in_sheet = set()
        conv_id_col = None
        for col in id_columns_sheet:
            if col.strip().lower().replace(' ', '_') == 'conversation_id':
                conv_id_col = col
                break
                
        if conv_id_col and conv_id_col in df_sheet.columns:
            ids_in_sheet = set(df_sheet[conv_id_col].astype(str))
        
        new_rows = df_local[~df_local['Conversation_ID'].astype(str).isin(ids_in_sheet)]
        
        # Cập nhật các dòng cần sửa
        if rows_to_update:
            print(f"🔄 Đang cập nhật {len(rows_to_update)} dòng có thay đổi...")
            for sheet_row_idx, local_row in rows_to_update:
                # Cập nhật Rating nếu cột tồn tại
                if rating_col:
                    rating_col_idx = original_headers.index(rating_col) + 1
                    rating_value = local_row['Rating'] if 'Rating' in local_row else ""
                    print(f"Cập nhật Rating ở ô ({sheet_row_idx}, {rating_col_idx}) thành '{rating_value}'")
                    sheet.update_cell(sheet_row_idx, rating_col_idx, rating_value)
                
                # Cập nhật Feedback nếu cột tồn tại
                if feedback_col:
                    feedback_col_idx = original_headers.index(feedback_col) + 1
                    feedback_value = local_row['Feedback'] if 'Feedback' in local_row else ""
                    print(f"Cập nhật Feedback ở ô ({sheet_row_idx}, {feedback_col_idx}) thành '{feedback_value}'")
                    sheet.update_cell(sheet_row_idx, feedback_col_idx, feedback_value)
                
            print("✅ Đã cập nhật xong các dòng có thay đổi.")
        
        # Thêm các dòng mới
        if not new_rows.empty:
            print(f"🟢 Đang thêm {len(new_rows)} dòng mới vào Google Sheet...")
            new_rows_list = new_rows.values.tolist()
            sheet.append_rows(new_rows_list, value_input_option="USER_ENTERED")
            print("✅ Đã thêm các dòng mới thành công.")
        
        if not rows_to_update and new_rows.empty:
            print("ℹ️ Không có thay đổi nào cần đồng bộ.")
            
    except Exception as e:
        print(f"❌ Lỗi khi đồng bộ dữ liệu: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
