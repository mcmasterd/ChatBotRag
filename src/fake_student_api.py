from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import random
import uvicorn
from datetime import date

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="API Học Bổng Sinh Viên",
    description="API để truy vấn thông tin sinh viên và trạng thái học bổng",
    version="1.0.0"
)

# Định nghĩa các enum cho trạng thái học bổng và ngành học
class TrangThaiHocBong(str, Enum):
    DA_NHAN = "đã nhận"
    DANG_CHO_XET_DUYET = "đang chờ xét duyệt"
    KHONG_DU_DIEU_KIEN = "không đủ điều kiện"
    CHUA_DANG_KY = "chưa đăng ký"

class NganhHoc(str, Enum):
    KHOA_HOC_MAY_TINH = "Khoa học máy tính"
    QUAN_TRI_KINH_DOANH = "Quản trị kinh doanh"
    KY_THUAT = "Kỹ thuật"
    Y_KHOA = "Y khoa"
    NGHE_THUAT = "Nghệ thuật và nhân văn"
    GIAO_DUC = "Giáo dục"
    KHOA_HOC_TU_NHIEN = "Khoa học tự nhiên"

# Định nghĩa model SinhVien với các trường tiếng Việt
class SinhVien(BaseModel):
    id: int
    ten: str
    nganh_hoc: NganhHoc
    diem_trung_binh: float = Field(..., ge=0.0, le=10.0)  # Cập nhật thang điểm 10
    ngay_nhap_hoc: date
    trang_thai_hoc_bong: TrangThaiHocBong
    so_tien_hoc_bong: Optional[float] = None
    thanh_tich_hoc_tap: List[str] = []

# Tạo dữ liệu mẫu với điểm trung bình ngẫu nhiên trên thang điểm 10 (ví dụ: từ 5.0 đến 10.0)
students_db = [
    SinhVien(
        id=1,
        ten="Nguyễn Văn A",
        nganh_hoc=NganhHoc.KHOA_HOC_MAY_TINH,
        diem_trung_binh=round(random.uniform(5.0, 10.0), 1),
        ngay_nhap_hoc=date(2021, 9, 1),
        trang_thai_hoc_bong=TrangThaiHocBong.DA_NHAN,
        so_tien_hoc_bong=5000.0,
        thanh_tich_hoc_tap=["Danh sách danh hiệu Dean's List", "Giải Hackathon"]
    ),
    SinhVien(
        id=2,
        ten="Trần Thị B",
        nganh_hoc=NganhHoc.QUAN_TRI_KINH_DOANH,
        diem_trung_binh=round(random.uniform(5.0, 10.0), 1),
        ngay_nhap_hoc=date(2022, 9, 1),
        trang_thai_hoc_bong=TrangThaiHocBong.DANG_CHO_XET_DUYET,
        thanh_tich_hoc_tap=["Vòng chung kết cuộc thi kinh doanh"]
    ),
    SinhVien(
        id=3,
        ten="Lê Văn C",
        nganh_hoc=NganhHoc.KY_THUAT,
        diem_trung_binh=round(random.uniform(5.0, 10.0), 1),
        ngay_nhap_hoc=date(2020, 9, 1),
        trang_thai_hoc_bong=TrangThaiHocBong.KHONG_DU_DIEU_KIEN,
        thanh_tich_hoc_tap=[]
    ),
    SinhVien(
        id=4,
        ten="Phạm Thị D",
        nganh_hoc=NganhHoc.Y_KHOA,
        diem_trung_binh=round(random.uniform(5.0, 10.0), 1),
        ngay_nhap_hoc=date(2019, 9, 1),
        trang_thai_hoc_bong=TrangThaiHocBong.DA_NHAN,
        so_tien_hoc_bong=10000.0,
        thanh_tich_hoc_tap=["Bài báo nghiên cứu", "Sinh viên y khoa xuất sắc"]
    ),
    SinhVien(
        id=5,
        ten="Hoàng Văn E",
        nganh_hoc=NganhHoc.NGHE_THUAT,
        diem_trung_binh=round(random.uniform(5.0, 10.0), 1),
        ngay_nhap_hoc=date(2021, 9, 1),
        trang_thai_hoc_bong=TrangThaiHocBong.CHUA_DANG_KY,
        thanh_tich_hoc_tap=["Triển lãm nghệ thuật"]
    ),
    SinhVien(
        id=6,
        ten="Vũ Thị F",
        nganh_hoc=NganhHoc.GIAO_DUC,
        diem_trung_binh=round(random.uniform(5.0, 10.0), 1),
        ngay_nhap_hoc=date(2022, 9, 1),
        trang_thai_hoc_bong=TrangThaiHocBong.DA_NHAN,
        so_tien_hoc_bong=4000.0,
        thanh_tich_hoc_tap=["Giải thưởng giảng dạy xuất sắc"]
    ),
    SinhVien(
        id=7,
        ten="Đặng Văn G",
        nganh_hoc=NganhHoc.KHOA_HOC_TU_NHIEN,
        diem_trung_binh=round(random.uniform(5.0, 10.0), 1),
        ngay_nhap_hoc=date(2020, 9, 1),
        trang_thai_hoc_bong=TrangThaiHocBong.DANG_CHO_XET_DUYET,
        thanh_tich_hoc_tap=["Giải chào mừng tại hội chợ khoa học"]
    ),
    SinhVien(
        id=8,
        ten="Bùi Thị H",
        nganh_hoc=NganhHoc.KHOA_HOC_MAY_TINH,
        diem_trung_binh=round(random.uniform(5.0, 10.0), 1),
        ngay_nhap_hoc=date(2021, 9, 1),
        trang_thai_hoc_bong=TrangThaiHocBong.KHONG_DU_DIEU_KIEN,
        thanh_tich_hoc_tap=[]
    ),
    SinhVien(
        id=9,
        ten="Đỗ Văn I",
        nganh_hoc=NganhHoc.QUAN_TRI_KINH_DOANH,
        diem_trung_binh=round(random.uniform(5.0, 10.0), 1),
        ngay_nhap_hoc=date(2022, 9, 1),
        trang_thai_hoc_bong=TrangThaiHocBong.CHUA_DANG_KY,
        thanh_tich_hoc_tap=["Dự án khởi nghiệp"]
    ),
    SinhVien(
        id=10,
        ten="Nguyễn Thị K",
        nganh_hoc=NganhHoc.KY_THUAT,
        diem_trung_binh=round(random.uniform(5.0, 10.0), 1),
        ngay_nhap_hoc=date(2020, 9, 1),
        trang_thai_hoc_bong=TrangThaiHocBong.DA_NHAN,
        so_tien_hoc_bong=8000.0,
        thanh_tich_hoc_tap=["Giải kỹ thuật", "Thực tập tại công ty hàng đầu"]
    ),]

# Định nghĩa các endpoint API (giữ nguyên phần còn lại của code như cũ)

@app.get("/students/", response_model=List[SinhVien], tags=["Sinh Viên"])
async def get_students(
    min_diem: Optional[float] = Query(None, ge=0.0, le=10.0, description="Điểm trung bình tối thiểu"),
    nganh_hoc: Optional[NganhHoc] = Query(None, description="Lọc theo ngành học"),
    trang_thai_hoc_bong: Optional[TrangThaiHocBong] = Query(None, description="Lọc theo trạng thái học bổng")
):
    filtered_students = students_db

    if min_diem is not None:
        filtered_students = [s for s in filtered_students if s.diem_trung_binh >= min_diem]
    
    if nganh_hoc is not None:
        filtered_students = [s for s in filtered_students if s.nganh_hoc == nganh_hoc]
    
    if trang_thai_hoc_bong is not None:
        filtered_students = [s for s in filtered_students if s.trang_thai_hoc_bong == trang_thai_hoc_bong]
    
    return filtered_students

@app.get("/students/{student_id}", response_model=SinhVien, tags=["Sinh Viên"])
async def get_student(student_id: int):
    for student in students_db:
        if student.id == student_id:
            return student
    raise HTTPException(status_code=404, detail=f"Không tìm thấy sinh viên có ID {student_id}")

@app.get("/scholarships/stats", tags=["Học Bổng"])
async def get_scholarship_stats():
    total_students = len(students_db)
    awarded = len([s for s in students_db if s.trang_thai_hoc_bong == TrangThaiHocBong.DA_NHAN])
    pending = len([s for s in students_db if s.trang_thai_hoc_bong == TrangThaiHocBong.DANG_CHO_XET_DUYET])
    ineligible = len([s for s in students_db if s.trang_thai_hoc_bong == TrangThaiHocBong.KHONG_DU_DIEU_KIEN])
    not_applied = len([s for s in students_db if s.trang_thai_hoc_bong == TrangThaiHocBong.CHUA_DANG_KY])
    
    total_amount = sum(s.so_tien_hoc_bong or 0 for s in students_db)
    
    return {
        "tong_so_sinh_vien": total_students,
        "so_luong_hoc_bong": {
            "da_nhan": awarded,
            "dang_cho_xet_duyet": pending,
            "khong_du_dieu_kien": ineligible,
            "chua_dang_ky": not_applied
        },
        "tong_so_tien_hoc_bong": total_amount,
        "trung_binh_so_tien_hoc_bong": total_amount / awarded if awarded > 0 else 0
    }

@app.get("/scholarships/eligible", response_model=List[SinhVien], tags=["Học Bổng"])
async def get_eligible_students(min_diem: float = Query(5.0, ge=0.0, le=10.0)):
    return [s for s in students_db if s.diem_trung_binh >= min_diem]

if __name__ == "__main__":
    uvicorn.run("fake_student_api:app", host="127.0.0.1", port=8000, reload=True)
