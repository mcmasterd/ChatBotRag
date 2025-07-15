import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path
import os
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

existing_rows = len(sheet_data)
print(f"Sheet hiện tại có {existing_rows} dòng")

# Hàm chuẩn hóa tên cột
normalize_col = lambda x: x.strip().replace(' ', '_').replace('\u00a0', '').replace('\xa0', '').replace('\t', '').replace('\n', '').replace('\r', '').replace('__', '_')

if existing_rows <= 1:  # Chỉ có header hoặc không có dữ liệu
    # ====== Xóa tất cả dữ liệu cũ và tải lại ======
    print("Sheet trống, đang tải toàn bộ dữ liệu lên...")
    if existing_rows > 0:  # Có header, giữ lại
        sheet.update(values=df_local.values.tolist(), range_name='A2:Z')
    else:  # Không có header, thêm cả header
        all_data = [df_local.columns.tolist()] + df_local.values.tolist()
        sheet.update(values=all_data, range_name='A1:Z')
    print(f"✅ Đã tải {len(df_local)} dòng lên Google Sheet.")
else:
    # ====== Đọc dữ liệu từ Google Sheet thành DataFrame ======
    try:
        # Chuẩn hóa tên cột của sheet về đúng chuẩn
        norm_headers = [normalize_col(h) for h in original_headers]
        df_sheet = pd.DataFrame(sheet_data[1:], columns=norm_headers)
        # Chuẩn hóa tên cột của local
        df_local.columns = [normalize_col(c) for c in df_local.columns]

        # Tìm các cột ID để so sánh
        id_columns = ['Conversation_ID', 'User_ID', 'Question', 'Answer']
        rating_col = 'Rating'
        feedback_col = 'Feedback'

        # Kiểm tra đủ cột ID
        for col in id_columns:
            if col not in df_sheet.columns:
                print(f"❌ Sheet thiếu cột {col}, không thể đồng bộ!")
                raise Exception(f"Sheet thiếu cột {col}")
            if col not in df_local.columns:
                print(f"❌ File log thiếu cột {col}, không thể đồng bộ!")
                raise Exception(f"File log thiếu cột {col}")

        # Chuẩn hóa dữ liệu về string để so sánh
        for col in id_columns + [rating_col, feedback_col]:
            if col in df_local.columns:
                df_local[col] = df_local[col].astype(str).fillna("").str.strip()
            if col in df_sheet.columns:
                df_sheet[col] = df_sheet[col].astype(str).fillna("").str.strip()

        # Tạo index nhanh cho sheet theo tuple ID
        sheet_index = {}
        for idx, row in df_sheet.iterrows():
            try:
                key = tuple(row[col] for col in id_columns)
                sheet_index[key] = (idx, row)
            except Exception as e:
                print(f"⚠️ Bỏ qua dòng sheet lỗi key: {e}")
                continue

        rows_to_update = []
        new_rows = []
        for _, local_row in df_local.iterrows():
            try:
                key = tuple(local_row[col] for col in id_columns)
            except Exception as e:
                print(f"⚠️ Bỏ qua dòng local lỗi key: {e}")
                continue
            if key in sheet_index:
                sheet_idx, sheet_row = sheet_index[key]
                # So sánh toàn bộ trường
                is_same = True
                for col in norm_headers:
                    local_val = str(local_row.get(col, "")).strip()
                    sheet_val = str(sheet_row.get(col, "")).strip()
                    if local_val != sheet_val:
                        is_same = False
                        break
                if is_same:
                    continue  # Bỏ qua nếu giống hệt
                # Nếu chỉ khác Rating hoặc Feedback thì update
                update_rating = (rating_col in df_local.columns and rating_col in df_sheet.columns and local_row[rating_col] != sheet_row[rating_col])
                update_feedback = (feedback_col in df_local.columns and feedback_col in df_sheet.columns and local_row[feedback_col] != sheet_row[feedback_col])
                if update_rating or update_feedback:
                    rows_to_update.append((sheet_idx + 2, local_row))  # +2 vì header là dòng 1
            else:
                new_rows.append(local_row.values.tolist())

        # Cập nhật các dòng cần sửa
        if rows_to_update:
            print(f"🔄 Đang cập nhật {len(rows_to_update)} dòng có thay đổi về Rating/Feedback...")
            for sheet_row_idx, local_row in rows_to_update:
                if rating_col in df_local.columns and rating_col in norm_headers:
                    rating_col_idx = norm_headers.index(rating_col) + 1
                    rating_value = local_row[rating_col]
                    print(f"Cập nhật Rating ở ô ({sheet_row_idx}, {rating_col_idx}) thành '{rating_value}'")
                    sheet.update_cell(sheet_row_idx, rating_col_idx, rating_value)
                if feedback_col in df_local.columns and feedback_col in norm_headers:
                    feedback_col_idx = norm_headers.index(feedback_col) + 1
                    feedback_value = local_row[feedback_col]
                    print(f"Cập nhật Feedback ở ô ({sheet_row_idx}, {feedback_col_idx}) thành '{feedback_value}'")
                    sheet.update_cell(sheet_row_idx, feedback_col_idx, feedback_value)
            print("✅ Đã cập nhật xong các dòng có thay đổi.")

        # Thêm các dòng mới
        if new_rows:
            print(f"🟢 Đang thêm {len(new_rows)} dòng mới vào Google Sheet...")
            sheet.append_rows(new_rows, value_input_option="USER_ENTERED")
            print("✅ Đã thêm các dòng mới thành công.")
        if not rows_to_update and not new_rows:
            print("ℹ️ Không có thay đổi nào cần đồng bộ.")
    except Exception as e:
        print(f"❌ Lỗi khi đồng bộ dữ liệu: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
