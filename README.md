# Employee_Attrition
# 📊 Employee Attrition Analysis & Prediction

Phân tích và dự đoán nguy cơ nghỉ việc (Attrition) của nhân viên bằng Python, dựa trên bộ dữ liệu **Employee Attrition Analytics Dataset 2026** (Kaggle). Dự án bao gồm đầy đủ quy trình: làm sạch dữ liệu → phân tích khám phá (EDA) → trực quan hóa → xây dựng mô hình Machine Learning.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Dataset](#-dataset)
- [Công nghệ sử dụng](#️-công-nghệ-sử-dụng)
- [Quy trình thực hiện](#-quy-trình-thực-hiện)
- [Kết quả mô hình](#-kết-quả-mô-hình)
- [Insight chính](#-insight-chính)
- [Hạn chế & Hướng phát triển](#️-hạn-chế--hướng-phát-triển)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Hướng dẫn chạy](#️-hướng-dẫn-chạy)

---

## 📖 Giới thiệu

Attrition (nghỉ việc) là một trong những bài toán quan trọng nhất của HR Analytics. Dự án này nhằm:

1. Làm sạch và chuẩn hóa dữ liệu nhân sự thô.
2. Khám phá các yếu tố ảnh hưởng đến quyết định nghỉ việc của nhân viên (lương, phòng ban, mức độ hài lòng, overtime...).
3. Xây dựng và so sánh nhiều mô hình Machine Learning để dự đoán khả năng một nhân viên sẽ rời công ty.

---

## 📂 Dataset

- **Nguồn:** [Employee Attrition Analytics Dataset 2026 – Kaggle](https://www.kaggle.com/datasets/mmumairkhattak/employee-attrition-analytics-dataset-2026)
- **Quy mô:** 15,000 nhân viên × 32 cột
- **Biến mục tiêu:** `Attrition` (Yes/No)

| Nhóm thông tin | Các cột tiêu biểu |
|---|---|
| Nhân khẩu học | `Age`, `Gender`, `Marital_Status`, `Education_Level` |
| Công việc | `Department`, `Job_Role`, `Employment_Type`, `Job_Level`, `Years_At_Company`, `Years_In_Current_Role` |
| Lương thưởng | `Monthly_Income`, `Salary_Hike_Percent`, `Stock_Option` |
| Hiệu suất & mức độ hài lòng | `Performance_Rating`, `Job_Satisfaction`, `Work_Life_Balance`, `Environment_Satisfaction`, `Relationship_Satisfaction`, `Manager_Rating` |
| Điều kiện làm việc | `Overtime`, `Remote_Work`, `Business_Travel`, `Distance_From_Home`, `Commute_Time_Minutes` |
| Khác | `Number_Of_Projects`, `Team_Size`, `Absence_Days`, `Promotion_Last_5_Years`, `Training_Hours_Last_Year`, `Attrition_Risk_Score` |

Dữ liệu được tải trực tiếp trong notebook bằng `kagglehub`, không cần tải file thủ công:

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mmumairkhattak/employee-attrition-analytics-dataset-2026",
    "employee_attrition_dataset.csv"
)
```

---

## 🛠️ Công nghệ sử dụng

- **Xử lý dữ liệu:** `pandas`, `numpy`
- **Trực quan hóa:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `lazypredict`
- **Tải dữ liệu:** `kagglehub`

---

## 🚀 Quy trình thực hiện

### 1️⃣ Data Cleaning

- Loại bỏ cột `Employee_ID` (không có giá trị phân tích).
- Kiểm tra dữ liệu: **0 giá trị null**, **0 dòng trùng lặp** — dữ liệu sạch ngay từ nguồn.
- Encode các cột nhị phân Yes/No → 0/1: `Overtime`, `Remote_Work`, `Promotion_Last_5_Years`, `Stock_Option`, `Attrition`.
- Encode các biến phân loại dạng ordinal/nominal:
  - `Gender`: Female = 0, Male = 1
  - `Marital_Status`: Single = 0, Married = 1, Divorced = 2
  - `Employment_Type`: Part-Time = 0, Full-Time = 1, Contract = 2
  - `Education_Level`: High School = 0, Bachelor = 1, PhD = 2, Master = 3

### 2️⃣ Exploratory Data Analysis (EDA)

| # | Biểu đồ | Mục đích |
|---|---|---|
| 1 | Pie + Bar chart phân phối Attrition | Xem tỷ lệ nghỉ việc tổng thể |
| 2 | Heatmap tương quan | Tìm các cặp biến số có tương quan mạnh |
| 3 | Bar chart tỷ lệ nghỉ việc theo `Department` | So sánh attrition giữa các phòng ban |
| 4 | Box plot `Years_At_Company` theo `Department` | So sánh độ gắn bó theo phòng ban |
| 5 | Bar chart tỷ lệ `Overtime` theo `Department` | Phát hiện phòng ban làm thêm giờ nhiều |
| 6 | Box plot `Monthly_Income` theo `Department` | So sánh thu nhập giữa các phòng ban |
| 7 | Box plot `Monthly_Income` theo `Job_Level` | Xem chênh lệch lương theo cấp bậc |
| 8 | Box plot kiểm tra Outliers cho toàn bộ biến số | Phát hiện giá trị bất thường |
| 9 | Histogram (dodge) các biến số theo `Attrition` | So sánh phân phối giữa nhóm nghỉ/ở lại |

### 3️⃣ Building Model

- **Tách đặc trưng:** `Years_At_Company`, `Monthly_Income`, `Salary_Hike_Percent`, `Job_Level`, `Performance_Rating`, `Job_Satisfaction`, `Work_Life_Balance`, `Environment_Satisfaction`, `Relationship_Satisfaction`, `Absence_Days`, `Distance_From_Home`, `Commute_Time_Minutes`.
- **Tiền xử lý:** `StandardScaler` cho biến số + `OneHotEncoder` cho biến phân loại (qua `ColumnTransformer`).
- **Train/Test split:** 70/30, `stratify` theo `Attrition` để giữ tỷ lệ lớp.
- **Benchmark nhanh:** dùng `LazyPredict` để chạy ~20 thuật toán cùng lúc và so sánh hiệu năng.
- **Huấn luyện chi tiết 3 mô hình:** `AdaBoostClassifier`, `LogisticRegression`, `RandomForestClassifier`.
- **Đánh giá:** Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report.
- **Phân tích Feature Importance** cho AdaBoost và Random Forest.

---

## 📈 Kết quả mô hình

### Benchmark LazyPredict (Top mô hình theo ROC-AUC)

| Model | Accuracy | Balanced Accuracy | ROC-AUC |
|---|---|---|---|
| **AdaBoostClassifier** | 0.748 | 0.670 | **0.777** |
| LGBMClassifier | 0.736 | 0.657 | 0.764 |
| RandomForestClassifier | 0.733 | 0.649 | 0.753 |
| ExtraTreesClassifier | 0.726 | 0.634 | 0.742 |
| XGBClassifier | 0.721 | 0.648 | 0.736 |
| LogisticRegression | 0.723 | 0.626 | 0.733 |

### Đánh giá chi tiết 3 mô hình chính

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **AdaBoostClassifier** | **0.748** | **0.665** | 0.446 | **0.534** |
| RandomForestClassifier | 0.733 | 0.633 | 0.412 | 0.499 |
| LogisticRegression | 0.723 | 0.628 | 0.351 | 0.450 |

➡️ **AdaBoostClassifier** cho kết quả tốt nhất tổng thể (Accuracy & ROC-AUC cao nhất), được chọn làm mô hình chính.

⚠️ **Lưu ý:** Recall cho lớp Attrition=1 ở cả 3 mô hình đều ở mức trung bình–thấp (35–45%), nghĩa là mô hình còn bỏ sót khá nhiều trường hợp nhân viên thực sự sẽ nghỉ việc — đây là điểm cần cải thiện (xem phần [Hướng phát triển](#️-hạn-chế--hướng-phát-triển)).

---

## 💡 Insight chính

- **Phòng ban:** Tỷ lệ nghỉ việc ở **HR cao nhất**, **Marketing thấp nhất**.
- **Thâm niên:** Nhân viên phòng **HR** có thời gian gắn bó trung bình **cao nhất**, **Marketing** thấp nhất. Đáng chú ý, **IT** và **Sales** là 2 phòng ban duy nhất có nhân viên gắn bó **trên 30 năm**.
- **Lương theo cấp bậc:** Nhân viên càng thành thạo công việc (`Job_Level` cao) thì mức lương trung bình càng cao, và có sự **chênh lệch lớn** giữa các cấp bậc.

---

## ⚠️ Hạn chế & Hướng phát triển

- **Recall thấp** ở lớp Attrition=1 → mô hình hiện tại phù hợp hơn cho mục đích tham khảo tổng quan, chưa đủ tin cậy để dùng "báo động sớm" từng cá nhân. Có thể cải thiện bằng:
  - Xử lý mất cân bằng dữ liệu (`SMOTE`, `class_weight='balanced'`).
  - Điều chỉnh ngưỡng phân loại (threshold tuning) thay vì mặc định 0.5.
- **Chưa đưa biến phân loại** (`Department`, `Job_Role`, `Business_Travel`) vào tập đặc trưng huấn luyện mô hình — có thể bổ sung để tăng sức mạnh dự đoán.
- Thử nghiệm thêm **GridSearch/Optuna** để tinh chỉnh hyperparameter cho AdaBoost, LightGBM, XGBoost.
- Bổ sung giải thích mô hình bằng **SHAP values** để hiểu sâu hơn về đóng góp của từng đặc trưng.

---

## 📁 Cấu trúc thư mục

```
.
├── Untitled10.ipynb     # Notebook chính: Cleaning → EDA → Visualization → ML
├── README.md            # Tài liệu mô tả dự án (file này)
└── requirements.txt     # Danh sách thư viện cần cài đặt
```

---

## ▶️ Hướng dẫn chạy

```bash
# 1. Clone repository
git clone <repo-url>
cd <repo-folder>

# 2. Cài đặt thư viện cần thiết
pip install -r requirements.txt

# 3. Mở notebook
jupyter notebook Untitled10.ipynb
```

**requirements.txt** gợi ý:
```
kagglehub
pandas
numpy
matplotlib
seaborn
scikit-learn
lazypredict
```

> 💡 Notebook tự động tải dataset qua `kagglehub` — cần đăng nhập Kaggle (file `kaggle.json`) hoặc chạy trên môi trường Kaggle Notebook/Colab có sẵn xác thực.

---

## 👤 Tác giả

*Cập nhật tên và liên hệ của bạn tại đây.*

## 📄 License

*Cập nhật giấy phép sử dụng (vd: MIT License) tại đây.*
