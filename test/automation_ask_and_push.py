import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path
import requests
import time

# ====== Màu terminal ======
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# ====== Cấu hình ======
GOOGLE_SHEET_ID = "18MWWJG9S8s9WIr8lN4fh-Amq-zh8aRLQWoB_RoFL7To"
WORKSHEET_NAME = "Testcase"
CREDENTIALS_FILE = "automation-test-ictu-chatbot-092c16c2e27a.json"
CHATBOT_API_URL = "http://192.168.1.129:6002/ask"

# ====== Đường dẫn credentials ======
BASE_DIR = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = BASE_DIR / CREDENTIALS_FILE

# ====== Kết nối Google Sheet ======
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
print(f"{CYAN}Kết nối Google Sheet...{RESET}")
creds = ServiceAccountCredentials.from_json_keyfile_name(str(CREDENTIALS_PATH), scope)
client = gspread.authorize(creds)
sh = client.open_by_key(GOOGLE_SHEET_ID)
sheet = sh.worksheet(WORKSHEET_NAME)

# ====== Xác định số dòng có dữ liệu ở cột "Đầu vào" (cột 3) ======
all_values = sheet.col_values(3)
last_data_row = len(all_values)

# ====== Hỏi chế độ test ======
print(f"{CYAN}Chọn chế độ test:{RESET}")
print(f"1. Test toàn bộ các dòng có dữ liệu ('Đầu vào') (dòng 2 đến dòng {last_data_row})")
print(f"2. Test thủ công một đoạn tự chộn (Nhập số dòng bắt đầu & số dòng test)")

while True:
    choice = input("Nhập lựa chọn (1 hoặc 2): ").strip()
    if choice in ("1", "2"):
        break
    print("Chỉ nhận giá trị 1 hoặc 2. Nhập lại!")

if choice == "1":
    start_row = 2
    end_row = last_data_row
    print(f"{CYAN}Test tất cả từ dòng {start_row} đến dòng {last_data_row}...{RESET}")
else:
    while True:
        try:
            start_row = int(input(f"Nhập số dòng bắt đầu (>=2): ").strip())
            if start_row < 2 or start_row > last_data_row:
                print(f"Chỉ nhận giá trị từ 2 đến {last_data_row}")
                continue
            break
        except Exception:
            print("Nhập số hợp lệ!")
    while True:
        try:
            num_rows = int(input(f"Nhập số dòng muốn test: ").strip())
            if num_rows < 1 or (start_row + num_rows - 1) > last_data_row:
                print(f"Chỉ nhận số phù hợp, không vượt quá dòng {last_data_row}")
                continue
            break
        except Exception:
            print("Nhập số hợp lệ!")
    end_row = start_row + num_rows - 1
    print(f"{CYAN}Test từ dòng {start_row} đến dòng {end_row}...{RESET}")

# ====== Bắt đầu test ======
success_count = 0
error_count = 0
skipped_count = 0

for i in range(start_row, end_row + 1):
    user_input = sheet.cell(i, 3).value  # Cột 3: Đầu vào

    if not user_input or not user_input.strip():
        print(f"{YELLOW}[{i}] Bỏ qua (ô trống hoặc không có dữ liệu){RESET}")
        skipped_count += 1
        continue

    print(f"{CYAN}[{i}] Đang gửi: {user_input}{RESET}")

    # Gửi POST tới bot
    try:
        resp = requests.post(
            CHATBOT_API_URL,
            json={"query": user_input, "user_id": "test_user_auto"},
            timeout=30
        )
        if resp.status_code == 200:
            bot_answer = resp.json().get("response", "")
        else:
            bot_answer = f"Lỗi {resp.status_code}"
    except Exception as e:
        bot_answer = f"Lỗi: {e}"

    # Update kết quả vào cột 5 (Kết quả thực tế)
    try:
        sheet.update_cell(i, 5, bot_answer)
        print(f"{GREEN}    → Đã ghi kết quả ở hàng {i}, cột 5: {bot_answer[:60]}...{RESET}")
        success_count += 1
    except Exception as e:
        print(f"{RED}    ⚠️ Lỗi khi ghi vào sheet hàng {i}: {e}{RESET}")
        error_count += 1

    time.sleep(0.5)

print("\n=======================")
print(f"{GREEN}✅ Hoàn thành test tự động.{RESET}")
print(f"Số dòng thành công: {GREEN}{success_count}{RESET}")
print(f"Số dòng lỗi ghi sheet: {RED}{error_count}{RESET}")
print(f"Số dòng bỏ qua (trống): {YELLOW}{skipped_count}{RESET}")
print("=======================")
