# Employee_Attrition
# 📊 Employee Attrition Analysis & Prediction

Phân tích và dự đoán nguy cơ nghỉ việc (Attrition) của nhân viên bằng Python, dựa trên bộ dữ liệu **Employee Attrition Analytics Dataset 2026** (Kaggle). Dự án bao gồm đầy đủ quy trình: làm sạch dữ liệu → phân tích khám phá (EDA) → trực quan hóa → xây dựng mô hình Machine Learning.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 🔍 Insight Chính

- **Tỷ lệ Attrition tổng thể:** 32.3% Yes / 67.7% No — gần 1/3 nhân viên trong dataset thuộc nhóm đã nghỉ việc.
- **Theo phòng ban:** `HR` có tỷ lệ nghỉ việc cao nhất (33.3%), `Marketing` thấp nhất (31.4%). Chênh lệch giữa các phòng ban chỉ khoảng ~2 điểm % — Department không phải yếu tố phân hóa mạnh.
- **Thâm niên:** `HR` có thời gian gắn bó trung bình cao nhất (4.6 năm), `Marketing` thấp nhất (4.35 năm). Đáng chú ý, `IT` (35 năm) và `Sales` (34 năm) là 2 phòng ban duy nhất có nhân viên gắn bó **trên 30 năm**.
- **Lương theo cấp bậc:** Thu nhập trung bình tăng đều và rõ rệt theo `Job_Level` — từ ~5.7 triệu (cấp 1) lên ~14.6 triệu (cấp 5), chênh lệch ~2.6 lần giữa cấp thấp nhất và cao nhất.
- **Outlier:** Một số biến số (`Commute_Time_Minutes`, `Absence_Days`...) có điểm dữ liệu vượt ngưỡng IQR thông thường, cần lưu ý nếu đưa vào các mô hình nhạy cảm với outlier (Logistic Regression).

---

## 📈 Kết quả Phân tích

**Từ biểu đồ tương quan:**
- Sau `Attrition_Risk_Score` (biến có sẵn, r = 0.60), hai yếu tố tương quan mạnh nhất với Attrition là **Overtime** (r = +0.23) và **Job_Satisfaction** (r = −0.20) — làm thêm giờ nhiều và mức hài lòng công việc thấp là 2 tín hiệu cảnh báo rõ ràng nhất.
- Các yếu tố điều kiện làm việc (`Work_Life_Balance`, `Distance_From_Home`, `Commute_Time_Minutes`) đều có tương quan ở mức trung bình (|r| ≈ 0.10–0.15).
- Đặc điểm nhân khẩu học (`Age`, `Number_Of_Projects`, `Team_Size`) gần như **không tương quan** với Attrition (|r| < 0.02) → trải nghiệm làm việc hằng ngày tác động đến quyết định nghỉ việc nhiều hơn là đặc điểm cá nhân.

**Từ các mô hình:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **AdaBoostClassifier** ⭐ | **0.748** | **0.665** | 0.446 | **0.534** | **0.777** |
| RandomForestClassifier | 0.733 | 0.633 | 0.412 | 0.499 | 0.753 |
| LogisticRegression | 0.723 | 0.628 | 0.351 | 0.450 | 0.733 |

- **AdaBoostClassifier** cho kết quả tốt nhất tổng thể, vượt trội nhẹ so với Random Forest và Logistic Regression trên cả Accuracy lẫn ROC-AUC.
- Dù Accuracy tổng thể khá khả quan (~75%), **Recall cho nhóm "sẽ nghỉ việc" chỉ đạt 35–45%** — mô hình bỏ sót hơn một nửa số ca thực sự nghỉ việc. Điều này phù hợp với mức tương quan trung bình-thấp quan sát được ở phần EDA: các đặc trưng hiện có chưa đủ mạnh để tách bạch hoàn toàn 2 nhóm.

---

## 🤖 Mô hình hóa dự báo nâng cao

- **Tiền xử lý:** `StandardScaler` cho biến số kết hợp `OneHotEncoder` cho biến phân loại thông qua `ColumnTransformer`.
- **Chia dữ liệu:** Train/test 70/30, `stratify` theo `Attrition` để giữ nguyên tỷ lệ lớp giữa 2 tập.
- **Benchmark sàng lọc:** Dùng `LazyPredict` chạy đồng thời ~20 thuật toán phân loại (AdaBoost, LightGBM, Random Forest, XGBoost, Logistic Regression...) để khoanh vùng ứng viên tiềm năng trước khi đi sâu.
- **Huấn luyện chi tiết:** 3 mô hình được tinh chỉnh và đánh giá đầy đủ bằng Confusion Matrix + Classification Report: `AdaBoostClassifier`, `RandomForestClassifier`, `LogisticRegression`.
- **Giải thích mô hình:** Trích xuất Feature Importance từ AdaBoost và Random Forest để xác định đặc trưng đóng góp nhiều nhất vào quyết định dự đoán.

---

## ⚠️ Hạn chế & Hướng phát triển

- **Recall thấp** ở lớp Attrition = 1 → mô hình hiện tại phù hợp để tham khảo xu hướng tổng quan, chưa đủ tin cậy để "báo động sớm" từng cá nhân cụ thể. Có thể cải thiện bằng:
  - Xử lý mất cân bằng dữ liệu (`SMOTE`, `class_weight='balanced'`).
  - Điều chỉnh ngưỡng phân loại (threshold tuning) thay vì mặc định 0.5.
- **Chưa đưa biến phân loại** (`Department`, `Job_Role`, `Business_Travel`) vào tập đặc trưng huấn luyện — có thể bổ sung để tăng sức mạnh dự đoán, đặc biệt khi `Department` cho thấy sự khác biệt nhẹ về Attrition trong EDA.
- Thử nghiệm **GridSearch/Optuna** để tinh chỉnh hyperparameter cho AdaBoost, LightGBM, XGBoost — hiện mới chạy với tham số mặc định.
- Bổ sung **SHAP values** để giải thích đóng góp của từng đặc trưng ở cấp độ từng dự đoán cá nhân, thay vì chỉ Feature Importance tổng quát.
- Cân nhắc loại `Attrition_Risk_Score` khỏi tập đặc trưng khi training (nếu đây là biến được tính toán *sau* khi biết Attrition) để tránh rủi ro **data leakage**.
