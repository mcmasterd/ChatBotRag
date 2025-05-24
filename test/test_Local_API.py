import requests

try:
    response = requests.post(
        "http://192.168.1.131:8000/embed",
        json={"texts": ["test embedding"]},
        timeout=10
    )   
    response.raise_for_status()
    print("✅ API hoạt động. Kết quả trả về:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print("❌ Lỗi khi gọi API:", e)
