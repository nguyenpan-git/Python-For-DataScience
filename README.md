# ĐỒ ÁN CUỐI KỲ
# MÔN: PYTHON CHO KHOA HỌC DỮ LIỆU
# 23KDL - Nhóm 23

MSSV       | HỌ VÀ TÊN |
---------- | --------- |
23280002   | Võ Nhật Tiến
23280036   | Nguyễn Văn Phúc An
23280010   | Đào Đức Thịnh

# BÀI TOÁN: CHẨN ĐOÁN NHÓM BỆNH
Chẩn đoán nhóm bệnh (Medical Condition) dựa vào các thông tin, chỉ số sức khoẻ của bệnh nhân.

- Mô hình được sử dụng:
    - Logistic Regression
    - SVM (RBF Kernel)
    - Random Forest
    - XGBoost

- Phương pháp tối ưu siêu tham số:
    - GridSearchCV
    - Optuna

Sử dụng các độ đo đánh giá mô hình hồi quy phổ biến: MAE, MSE, RMSE, R2-score.

## Mục lục
1. [Giới thiệu bài toán](#1-giới-thiệu)
2. [Cấu trúc repo](#2-cấu-trúc-thư-mục)
3. [Cài đặt](#3-cài-đặt)
4. [Cách sử dụng](#4-cách-sử-dụng)
5. [Dataset](#5-dataset)
6. [Kết quả](#6-kết-quả)

## 1. Giới thiệu bài toán

### 1.1. Mục tiêu: 
Chẩn đoán nhóm bệnh (Medical Condition) dựa vào các thông tin, chỉ số sức khỏe của bệnh nhân (ví dụ lượng đường huyết, lượng mỡ máu, thời gian tập luyện trong tuần…).

### 1.2. Phương pháp: 
Xây dựng một mô hình dự đoán có khả năng phân loại bệnh nhân vào một trong các nhóm bệnh (hoặc tình trạng sức khỏe) được xác định trước. Mô hình sẽ sử dụng các chỉ số sức khỏe và thông tin sinh hoạt của bệnh nhân làm dữ liệu đầu vào.

---

## 2. Cấu trúc repo

```
Python-For-DataScience/
├── src/                                          # Source code (.py)
│   ├── pythonkhdl_doancuoiky_eda                 # Khám phá dữ liệu (EDA)
│   ├── pythonkhdl_doancuoiky_preprocessor        # Tiền xử lý dữ liệu
│   ├── pythonkhdl_doancuoiky_modeltrainer        # Mô hình học máy
│   └── pythonkhdl_doancuoiky_pipeline            # Chuỗi các quy trình xử lý
├── data/                                         # Dữ liệu
│   ├── raw/                                      # Gốc (chưa xử lý)
│   │   └── dirty_v3_path.csv   
│   └── interim/                                  # Tài liệu lúc xử lý
│       ├── medical_condition_na.csv
│       └── medical_condition_not_na.csv                
├── outputs/                                      # Kết quả đầu ra
│   ├── logs/                                     # Logs huấn luyện
│   │   └── training_process.log           
│   ├── models/                                   # Models tốt nhất đã lưu
│   │   ├── best_hyperparameters.json             # Siêu tham số
│   │   └── best_model.pkl                        # Mô hình
│   └── results/                                  # Kết quả
│       ├── final_predicted_data                  # Dữ liệu đã dự đoán
│       ├── full_dataset_completed                # Dữ liệu tổng sau pipeline
│       └── model_comparison_results              # Kết quả so sánh model
├── notebooks/                                    # .ipynb
│   ├── Python_KHDL_DoAnCuoiKy_Pipeline           # Pipeline xử lý (bao gồm phân tích)
│   └── PythonKHDL_DoAnCuoiKy_EDA                 # EDA (bao gồm phân tích)
├── requirements.txt
└── readme.md
```
---

## 3. Cài đặt
### 3.1. Clone repository:
```bash
git clone https://github.com/nguyenpan-git/Python-For-DataScience.git
cd "Python-For-DataScience/"
```
### 3.2. Cài đặt dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
---

## 4. Cách sử dụng
### 4.1. Chạy pipeline đầy đủ (EDA → tiền xử lý → huấn luyện → dự đoán)

Bước 1. Import các module trong thư mục src/ (cũng như các thư viện cần thiết cho các lệnh khác tuỳ thuộc):
```python
from src.pythonkhdl_doancuoiky_preprocessor import DataPreprocessor
from src.pythonkhdl_doancuoiky_modeltrainer import ModelTrainer
from src.pythonkhdl_doancuoiky_eda import DistributionComparison
```
Bước 2. Mở file `pythonkhdl_doancuoiky_pipeline.py`
Bước 3. Sửa đường dẫn dữ liệu ở dòng:

```python
prepro = DataPreprocessor.from_file("dirty_v3_path.csv")
```
Bước 4. Chạy lệnh:
```python
python pythonkhdl_doancuoiky_pipeline.py
```
**Lưu ý**: Quá trình **EDA** đã được thực hiện trước đó, với code và kết quả hiển thị trong notebook *PythonKHDL_DoAnCuoiKy_EDA* trong thư mục notebooks. Các lớp đối tượng được sử dụng cho quá trình EDA được lưu trong file *pythonkhdl_doancuoiky_eda.py* thuộc thư mục **src**. Để chạy lại quá trình EDA có thể tải notebook và chạy trực tiếp sau khi truyền dữ liệu tương tự như quá trình chạy pipeline. Tuy nhiên, để notebook EDA có thể chạy hoàn toàn cần thêm 2 datasets *medical_condition_na.csv* và *medical_condition_not_na.csv*.

### 4.2. Chạy trực tiếp script huấn luyện (ModelTrainer – argparse)

#### 4.2.1. Các tham số hỗ trợ
| Tham số            | Mô tả                                      | Kiểu | Giá trị hợp lệ                                                                      |
| ------------------ | ------------------------------------------ | ---- | ----------------------------------------------------------------------------------- |
| `--data_path`      | Đường dẫn file dữ liệu `.csv`              | str  | Bắt buộc                                                                            |
| `--target_col`     | Tên cột Label (ví dụ: `Medical Condition`) | str  | Bắt buộc                                                                            |
| `--model`          | Tên model cần train                        | str  | `RandomForest`, `LogisticRegression`, `SVM`, `XGBoost` hoặc bỏ trống để chạy tất cả |
| `--method`         | Cách tối ưu siêu tham số                   | str  | `grid`, `optuna`, `none`                                                            |
| `--n_trials`       | Số trial cho Optuna                        | int  | > 0 (chỉ dùng khi `--method optuna`)                                                |
| `--output_model`   | Tên file lưu model tốt nhất                | str  | Mặc định: `best_model.pkl`                                                          |
| `--output_params`  | File lưu tham số tối ưu                    | str  | Mặc định: `best_hyperparameters.json`                                               |
| `--output_results` | File lưu bảng kết quả đánh giá             | str  | Mặc định: `experiment_results.csv`                                                  |
#### 4.2.2. Ví dụ chạy
Huấn luyện tất cả model bằng GridSearch:
```bash
python pythonkhdl_doancuoiky_modeltrainer.py \
  --data_path medical_condition_not_na.csv \
  --target_col "Medical Condition" \
  --method grid
```
Chỉ huấn luyện RandomForest, tối ưu bằng Optuna 50 trials:
```bash
python pythonkhdl_doancuoiky_modeltrainer.py \
  --data_path medical_condition_not_na.csv \
  --target_col "Medical Condition" \
  --model RandomForest \
  --method optuna \
  --n_trials 50

```
Đổi tên file lưu model và kết quả:
```bash
python pythonkhdl_doancuoiky_modeltrainer.py \
  --data_path medical_condition_not_na.csv \
  --target_col "Medical Condition" \
  --method grid \
  --output_model rf_best.pkl \
  --output_params rf_params.json \
  --output_results rf_results.csv
```
---

## 5. Dataset
- Nguồn: [Kaggle - Healthcare Risk Factors Dataset](https://www.kaggle.com/datasets/abdallaahmed77/healthcare-risk-factors-dataset/data)
- File: `data/raw/dirty_v3_path.csv`
- Quy mô: ~30,000 bản ghi – 20 đặc trưng
- Biến mục tiêu: `Medical Condition`

| **Tên cột**           | **Kiểu dữ liệu** | **Ý nghĩa mô tả**                                                                      |
| --------------------- | ---------------- | -------------------------------------------------------------------------------------- |
| **Age**               | float            | Tuổi bệnh nhân.                                                                        |
| **Gender**            | object           | Giới tính (Male/Female/NaN).                                                           |
| **Medical Condition** | object           | Nhóm bệnh hoặc tình trạng sức khỏe (Diabetes, Healthy, Asthma, Obesity, Hypertension). |
| **Glucose**           | float            | Giá trị đường huyết.                                                                   |
| **Blood Pressure**    | float            | Giá trị huyết áp.                                                                      |
| **BMI**               | float            | Chỉ số BMI.                                                                            |
| **Oxygen Saturation** | float            | Mức bão hòa oxy.                                                                       |
| **LengthOfStay**      | int/float        | Thời gian nằm viện.                                                                    |
| **Cholesterol**       | float            | Mức cholesterol.                                                                       |
| **Triglycerides**     | float            | Giá trị triglyceride.                                                                  |
| **HbA1c**             | float            | Giá trị HbA1c.                                                                         |
| **Smoking**           | int              | 0/1 theo dữ liệu (không thấy thông tin mô tả thêm).                                    |
| **Alcohol**           | int              | 0/1 theo dữ liệu.                                                                      |
| **Physical Activity** | float            | Chỉ số hoạt động thể chất.                                                             |
| **Diet Score**        | float            | Điểm đánh giá chế độ ăn.                                                               |
| **Family History**    | int              | 0/1 theo dữ liệu (có/không tiền sử gia đình).                                          |
| **Stress Level**      | float            | Mức độ căng thẳng.                                                                     |
| **Sleep Hours**       | float            | Số giờ ngủ.                                                                            |
| **random_notes**      | object           | Chuỗi text ngắn dạng “lorem / ipsum”.                                                  |
| **noise_col**         | float            | Cột nhiễu, không có ý nghĩa rõ ràng (dữ liệu số bất quy tắc).                          |
---
## 6. Kết quả
Kết quả chi tiết được lưu trong thư mục `outputs/` bao gồm: 
- Logs huấn luyện / thực nghiệm.
- Models tốt nhất đã được tối ưu tham số.
- Kết quả dự đoán.