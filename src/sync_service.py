import time
import subprocess
import logging
from pathlib import Path

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sync_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sync_service")

# Đường dẫn đến script cập nhật
BASE_DIR = Path("/home/skynet1/ChatBotRag")
SCRIPT_PATH = BASE_DIR / "src" / "update_google_sheet.py"

def sync_to_sheets():
    """Chạy script cập nhật lên Google Sheets"""
    try:
        logger.info("Bắt đầu đồng bộ dữ liệu lên Google Sheets")
        # Thêm thông tin chi tiết khi lỗi xảy ra
        result = subprocess.run(["python3", str(SCRIPT_PATH)], 
                               capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            logger.info("Đồng bộ thành công")
            return True
        else:
            logger.error(f"Đồng bộ thất bại với mã lỗi {result.returncode}")
            logger.error(f"Lỗi: {result.stderr}")
            return False
    except subprocess.SubprocessError as e:
        logger.error(f"Lỗi khi đồng bộ: {str(e)}")
        return False

def main():
    """Hàm chính của dịch vụ đồng bộ"""
    logger.info("Dịch vụ đồng bộ Google Sheets đã khởi động")
    
    # Khoảng thời gian đồng bộ (đơn vị: giây)
    SYNC_INTERVAL = 15  # 5 giây (khoảng thời gian rất ngắn)
    
    # Ghi cảnh báo nếu thời gian đồng bộ quá ngắn
    if SYNC_INTERVAL < 60:
        logger.warning(
            f"CẢNH BÁO: Thời gian đồng bộ được đặt quá ngắn ({SYNC_INTERVAL} giây).\n"
            "Điều này có thể gây ra các vấn đề:\n"
            "1. Vượt quá giới hạn API của Google Sheets (100 requests/phút/dự án)\n"
            "2. Tăng tải CPU không cần thiết\n"
            "3. Tạo quá nhiều log"
        )
    
    try:
        while True:
            sync_to_sheets()
            logger.info(f"Đợi {SYNC_INTERVAL} giây trước lần đồng bộ tiếp theo")
            time.sleep(SYNC_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Dịch vụ đồng bộ đã dừng")
    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {str(e)}")

if __name__ == "__main__":
    main() 