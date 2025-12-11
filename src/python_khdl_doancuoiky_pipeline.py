# PHẦN I: IMPORT MODULE, THU THẬP DỮ LIỆU

# Import các lớp cần thiết từ các module
# Distribution Comparison là lớp đối tượng phục vụ cho quá trình kiểm định phân phối ở hai bộ dữ liệu khác nhau
# Data Preprocessor là lớp đối tượng phục vụ cho quá trình tiền xử lý
# Model Trainer là lớp đối tượng phục vụ cho quá trình huấn luyện và đánh giá mô hình
from pythonkhdl_doancuoiky_preprocessor import DataPreprocessor
from pythonkhdl_doancuoiky_modeltrainer import ModelTrainer, setup_logging
from pythonkhdl_doancuoiky_eda import DistributionComparison
from IPython.display import display


# Đọc file
prepro = DataPreprocessor.from_file("dirty_v3_path.csv")

prepro.df.head()

"""# PHẦN II: TIỀN XỬ LÝ DỮ LIỆU

## Dữ liệu trùng lặp, giá trị âm và các đặc trưng gây nhiễu
"""

# Xóa các dòng trùng lặp
prepro.handle_duplicate(strategy="keepFirst")

# Dữ liệu âm trong Triglycerides và Physical Activity chắc chắn là lỗi, cần phải thay thế bằng giá trị tuyệt đối.
# Dữ liệu âm trong Diet Score, Stress Level cũng lấy giá trị tuyệt đối.
prepro.handle_negative(
    strategy="abs",
    columns=["Triglycerides", "Physical Activity", "Diet Score", "Stress Level"]
)

# Loại bỏ các cột gây nhiễu (random_notes, noise_col)
prepro.drop_columns(["random_notes", "noise_col"])

"""## Dữ liệu trống"""

'''
1. Dữ liệu trống trong Age: đối với các phân phối bị lệch như Age và có ngoại lai, điền dữ liệu trống bằng trung vị
là phương pháp an toàn và mạnh mẽ hơn so với trung bình, vì nó ít bị ảnh hưởng bởi các giá trị cực đoan.'''
prepro.handle_missing(strategy="median", columns=["Age"])

'''
2. Dữ liệu trống trong Gender: do phân phối về tỷ lệ giới tính trong bộ dữ liệu gốc và trong bộ dữ liệu loại bỏ các giá trị trống
là tương đối đồng đều và không quá khác biệt so với nhau (từ kết quả EDA ta có)
nên để xử lý dữ liệu trống trong Gender có thể điền ngẫu nhiên giới tính (Male/Female).'''
prepro.handle_missing(strategy="random-fill", columns=["Gender"])

# 3. Dữ liệu trống trong Glucose: tương tự như Age, điền bằng trung vị
prepro.handle_missing(strategy="median", columns=["Glucose"])

'''
4. Dữ liệu trống trong Blood Pressure: do phân phối dữ liệu trong Blood Pressure có hình dạng tương tự phân phối chuẩn,
nên để tránh làm sai lệch phân phối gốc, sẽ điền các giá trị dữ liệu trống bằng trung vị.'''
prepro.handle_missing(strategy="median", columns=["Blood Pressure"])

"""Ý tưởng được đặt ra là, để xử lý các mẫu có Medical Condition là dữ liệu trống, có thể tách các dòng này thành một tập hợp dữ liệu mô phỏng cho dữ liệu mới (không sử dụng cho mô hình, dù huấn luyện hay đánh giá).
Dữ liệu mô phỏng này được xem nhưng dữ liệu thực sẽ được mô hình tiếp nhận khi triển khai thực tế, chính là các dữ liệu không có nhãn là cần phải được dự đoán.


Để có thể thực hiện được điều này, phải đảm bảo bằng dữ liệu trống trong Medical Condition là hoàn toàn ngẫu nhiên và không ẩn chứa quy luật đặc biệt (MCAR).
Khi đó, phân phối ở tất cả các đặc trưng ở hai nhóm (nhóm có Medical Condition trống và nhóm có đầy đủ) phải là như nhau và có thể hoán đổi qua lại lẫn nhau. 
Tiến hành kiểm định phân phối để đảm bảo yếu tố này. Lưu ý: dữ liệu sẽ được tiền xử lý trước khi kiểm tra phân phối, khi đó dữ liệu đã sạch và chỉ duy nhất cột Medical Condition là còn có giá trị N/A.

"""

# Medical Condition: KHÔNG FILL NA
# Tách thành 2 tập:
# - df_mc_na: các dòng thiếu Medical Condition -> xem như tập không nhãn mô phỏng dữ liệu mới (unseen_data)
# - df_mc_not_na: các dòng có Medical Condition -> dùng cho huấn luyện và đánh giá mô hình
df_mc_na, df_mc_not_na = prepro.split_by_na("Medical Condition")

pre_na = DataPreprocessor(df_mc_na)
pre_not_na = DataPreprocessor(df_mc_not_na)

distribution_comparison = DistributionComparison(seen_df = df_mc_not_na, unseen_df = df_mc_na)

categorical_cols = ['Gender', 'Smoking', 'Alcohol', 'Family History']
numeric_cols = ['Age', 'BMI', 'Glucose', 'Blood Pressure', 'Oxygen Saturation', 'LengthOfStay',
                'Cholesterol', 'Triglycerides', 'HbA1c', 'Physical Activity', 'Diet Score', 'Stress Level', 'Sleep Hours']
distribution_comparison.compare_distributions(numeric_cols, categorical_cols)

"""Như vậy, từ quá trình kiểm định phân phối cho thấy dữ liệu không có sự khác biệt đáng kể ở hai nhóm (ngoại trừ Family History, 
nhưng p-value không quá thấp và vẫn có thể bác bỏ nếu mức ý nghĩa = 0.01),
củng cố lập luận rằng dữ liệu trống ngẫu nhiên (MCAR) và có thể tách các dòng có giá trị trống ở Medical Condition thành tập dữ liệu mô phỏng

## Chuẩn hóa dữ liệu (Scaling)
"""

'''
Chuẩn hóa (Standard/Min-Max Scaling): đối với các đặc trưng dữ liệu số có phân phối xấp xỉ phân phối chuẩn sẽ thực hiện Standard Scaling,
còn các đặc trưng dữ liệu số không xấp xỉ phân phối chuẩn sẽ sử dụng Min-Max Scaling.'''

standard_cols = [
    "BMI",
    "Oxygen Saturation",
    "Cholesterol",
    "Triglycerides",
    "Sleep Hours",
    "Stress Level"
]
minmax_cols = [
    "Glucose",
    "Blood Pressure",
    "LengthOfStay",
    "HbA1c",
    "Physical Activity",
    "Diet Score",
    "Age"
]

# Chỉ giữ lại các cột thực sự tồn tại và là số
num_cols = pre_na.df.select_dtypes(include="number").columns.tolist()
standard_cols = [c for c in standard_cols if c in num_cols]
minmax_cols = [c for c in minmax_cols if c in num_cols and c not in standard_cols]

# Standard Scaling cho nhóm xấp xỉ phân phối chuẩn
prepro.scale(columns=standard_cols, method="standard")

# MinMax Scaling cho nhóm còn lại
prepro.scale(columns=minmax_cols, method="minmax")

prepro.df.head()

# Tách file cho phân tích
df_missing, df_clean = prepro.split_by_na("Medical Condition")

prepro_na_scaled = DataPreprocessor(df_missing)
prepro_not_na_scaled = DataPreprocessor(df_clean)

prepro_na_scaled.save("medical_condition_na.csv", save_index=True)
prepro_not_na_scaled.save("medical_condition_not_na.csv", save_index=True)

"""## Mã hóa đặc trưng phân loại"""

# Mã hóa đặc trưng phân loại: cần phải Label Encoding với Medical Condition do chọn làm biến mục tiêu.
prepro_not_na_scaled.label_encode(["Medical Condition"])

# Vì Gender chỉ có 2 giá trị (Male/Female), dùng label_encode sẽ tự động cho ra nhị phân 0/1
prepro_not_na_scaled.label_encode(["Gender"])

"""# PHẦN III: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH"""

# Thiết lập Logging (Ghi log ra file training_process.log)
setup_logging(log_file='training_process.log')
print("Đã thiết lập logging. Kiểm tra file 'training_process.log' để xem chi tiết quá trình.")

# Khởi tạo đối tượng huấn luyện
trainer = ModelTrainer(
    data = prepro_not_na_scaled.df,
    target_col = 'Medical Condition',
    random_state=42
)

# Tên file CSV để lưu bảng so sánh kết quả các phương pháp
history_file = 'model_comparison_results.csv'
print("--- Đã khởi tạo Trainer ---")

"""## Tham số mặc định"""

# Chạy tất cả model với tham số mặc định
trainer.train_model(tuning_method = None)

# Đánh giá và ghi vào file lịch sử
trainer.save_results_to_history(tuning_method = 'Default', filename = history_file)

df_default = trainer.evaluate_all()
df_default

"""## Optuna tối ưu tham số"""

# Chạy tất cả model với tối ưu tham số bằng Optuna
trainer.train_model(tuning_method = 'optuna')

# Đánh giá và ghi tiếp vào file lịch sử
trainer.save_results_to_history(tuning_method = 'Optuna', filename = history_file)

df_optuna = trainer.evaluate_all()
print(df_optuna)

"""## Lưu kết quả đánh giá mô hình"""

import os
import pandas as pd

#  Tổng hợp kết quả và lưu model
if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)
    display(history_df)
else:
    print("Chưa có file lịch sử.")

# Lưu model tốt nhất (Dựa trên accuracy hiện tại trong trainer)
trainer.save_best_model('best_model.pkl')
trainer.save_best_params('best_hyperparameters.json')

"""# PHẦN IV: PHÂN TÍCH KẾT QUẢ MÔ HÌNH TỐT NHẤT"""

# Lấy mô hình tốt nhất theo Accuracy
best_model_name = max(trainer.results, key=lambda k: trainer.results[k]['accuracy'])
print(f"Model tốt nhất được chọn để vẽ: {best_model_name}")

# Vẽ Confusion Matrix cho model tốt nhất này
print(f"\n--- Confusion Matrix: {best_model_name} ---")
trainer.plot_confusion_matrix(best_model_name)

# Vẽ feature importance (nếu có thể)
print(f"\n--- Feature Importance: {best_model_name} ---")
try:
    trainer.plot_feature_importance(best_model_name, top_n=15)
except Exception as e:
    print(f"Model {best_model_name} không hỗ trợ vẽ feature importance mặc định ({e})")

import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

model = joblib.load('best_model.pkl')
target_col = 'Medical Condition'
X_shap = prepro_not_na_scaled.df.drop(columns=[target_col])

if hasattr(model, 'feature_names_in_'):
    X_shap = X_shap[model.feature_names_in_]

try:
    explainer = shap.TreeExplainer(model, model_output="raw")
    shap_values = explainer.shap_values(X_shap)
except:
    X_shap_sample = X_shap.iloc[:100]
    # Tạo một hàm wrapper cho predict_proba
    def predict_proba_wrapper(X):
        # Đảm bảo đầu vào là DataFrame/array phù hợp
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=X_shap_sample.columns)
        return model.predict_proba(X)

    try:
        # Sử dụng hàm wrapper đã tạo
        explainer = shap.KernelExplainer(predict_proba_wrapper, X_shap_sample)
        shap_values = explainer.shap_values(X_shap_sample)
        X_shap = X_shap_sample 
        print("Sử dụng KernelExplainer thành công.")
    except Exception as ke:
        print(f"Lỗi KernelExplainer: {ke}. Không thể tính toán SHAP values.")
        exit()

if hasattr(model, 'classes_'):
    class_names = model.classes_
elif isinstance(shap_values, list):
    class_names = range(len(shap_values))
else:
    class_names = range(shap_values.shape[-1])

# --- Tạo Biểu đồ cho từng Lớp ---

for i, class_label in enumerate(class_names):
    
    # Lựa chọn SHAP values và Base Value (Expected Value) cho lớp hiện tại
    if isinstance(shap_values, list):
        shap_val_i = shap_values[i]
        base_val_i = explainer.expected_value[i] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        shap_val_i = shap_values[:, :, i]
        base_val_i = explainer.expected_value[i] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        
    elif len(class_names) == 1: 
        shap_val_i = shap_values
        base_val_i = explainer.expected_value
    else:
        print(f"Lỗi: Không thể tìm thấy SHAP values cho lớp {class_label}")
        continue


    print(f"\n--- Đang tạo biểu đồ cho Lớp: {class_label} ---")
    
    # 1. Beeswarm Plot (Summary Plot)
    plt.figure()
    shap.summary_plot(shap_val_i, X_shap, show=False)
    plt.title(f"Beeswarm Plot - Lớp {class_label}")
    plt.show()

    # 2. Waterfall Plot (Local Explanation - chỉ lấy hàng đầu tiên)
    plt.figure()
    
    if not X_shap.empty:
        explanation = shap.Explanation(
            values=shap_val_i[0],
            base_values=base_val_i,
            data=X_shap.iloc[0].values, 
            feature_names=X_shap.columns.tolist() 
        )
        shap.plots.waterfall(explanation, show=False)
        plt.title(f"Waterfall Plot - Lớp {class_label} (Trường hợp 0)")
        plt.show()
    else:
        print("Không có dữ liệu trong X_shap để tạo Waterfall Plot.")

# Phân giải mã hóa để phân tích kết quả
prepro_not_na_scaled.inverse_label_encode(['Medical Condition'], inplace=False)
df_temp = prepro_not_na_scaled.get_df()
mapping = df_temp[['Medical Condition', 'Medical Condition_original']].drop_duplicates().sort_values('Medical Condition')
for _, row in mapping.iterrows():
    print(f"{row['Medical Condition']}   {row['Medical Condition_original']}")

"""## Nhận xét và đánh giá kết quả
1. Đánh giá Hiệu suất Mô hình (Model Performance)

Kết quả so sánh thực nghiệm giữa 4 thuật toán phổ biến (XGBoost, Random Forest, SVM, Logistic Regression) trên tập dữ liệu y tế cho thấy sự vượt trội của các mô hình cây quyết định nâng cao (Ensemble Methods).

- XGBoost là mô hình tối ưu nhất:

  - Sau khi tối ưu hóa bằng Optuna, XGBoost đạt độ chính xác (Accuracy) cao nhất là 93.06% và F1-Score đạt 0.9302. Điều này cho thấy khả năng vượt trội của XGBoost trong việc nắm bắt các mối quan hệ phi tuyến tính phức tạp giữa các chỉ số sức khỏe (như Glucose, BMI, Age) và tình trạng bệnh.

  - So với Random Forest (91.63%), XGBoost có sự cải thiện khoảng 1.4%. Tuy nhiên, cả hai mô hình dựa trên cây (Tree-based) đều vượt xa các mô hình tuyến tính và vector hỗ trợ.

- Hiệu suất của các mô hình khác:

  - Random Forest: Đạt kết quả rất tốt (91.63%) và ổn định qua các lần thử nghiệm. Đây là một lựa chọn thay thế mạnh mẽ nếu cần một mô hình đơn giản hơn, ít tham số cần tinh chỉnh hơn XGBoost.

  - SVM và Logistic Regression: Cả hai đều dừng lại ở mức độ chính xác khoảng 90.3%. Việc Logistic Regression thấp hơn cho thấy dữ liệu có tính chất phi tuyến tính mạnh mà một đường ranh giới quyết định tuyến tính không thể phân tách hoàn hảo.

=> Kết luận: XGBoost được lựa chọn làm mô hình cuối cùng (best_model.pkl) để dự đoán nhãn cho tập dữ liệu thiếu (missing data) nhờ hiệu suất cao nhất và khả năng tổng quát hóa tốt.

2. Phân tích Tầm quan trọng của Đặc trưng (Feature Importance - SHAP)

Để tránh việc mô hình hoạt động như một "hộp đen", kỹ thuật SHAP (SHapley Additive exPlanations) đã được áp dụng để giải thích kết quả dự đoán. Các biểu đồ phân tích SHAP (Summary Plot và Waterfall Plot) cung cấp cái nhìn sâu sắc về cách từng đặc trưng ảnh hưởng đến quyết định của mô hình.

- Phân tích Biểu đồ Beeswarm (SHAP Summary Plot):

  - Biểu đồ này xếp hạng các đặc trưng dựa trên tầm quan trọng trung bình. Đối với bài toán phân loại bệnh, các đặc trưng như Age (Tuổi), BMI, Glucose, và Blood Pressure thường nằm ở top đầu, đóng vai trò quyết định lớn nhất.

  - Mối quan hệ giá trị: Màu sắc của các điểm (đỏ là giá trị cao, xanh là giá trị thấp) giúp ta nhận diện quy luật. Ví dụ: Với lớp bệnh "Diabetes" (Tiểu đường), các điểm màu đỏ (giá trị Glucose cao) sẽ tập trung mạnh về phía dương của trục SHAP value, nghĩa là lượng đường huyết cao làm tăng mạnh xác suất dự báo bệnh tiểu đường. Tương tự, BMI cao thường kéo theo nguy cơ "Obesity" (Béo phì).

- Phân tích Biểu đồ Waterfall (Local Interpretability):

  - Đối với từng bệnh nhân cụ thể, biểu đồ Waterfall minh họa cách các chỉ số cộng gộp để đẩy xác suất dự đoán từ giá trị cơ sở (base value) lên giá trị dự báo cuối cùng. Điều này cực kỳ hữu ích trong y tế để giải thích lý do tại sao một bệnh nhân cụ thể lại được chẩn đoán mắc bệnh đó (ví dụ: "Bệnh nhân được dự đoán Cao huyết áp chủ yếu do chỉ số Blood Pressure là 178 và Tuổi là 66").

3. Đánh giá qua Ma trận Nhầm lẫn (Confusion Matrix)

- Việc trực quan hóa Confusion Matrix cho thấy mô hình XGBoost phân loại tốt đồng đều trên các lớp bệnh (Diabetes, Hypertension, Asthma, v.v.).

- Các lỗi phân loại thường xảy ra giữa các cặp bệnh có triệu chứng hoặc chỉ số sinh học tương đồng (ví dụ: sự nhầm lẫn nhẹ giữa người khỏe mạnh và người có chỉ số bệnh ở mức biên). Tuy nhiên, tỷ lệ chéo trên đường chéo chính là rất cao, khẳng định độ tin cậy của mô hình.

KẾT LUẬN CHUNG

Quy trình xây dựng mô hình từ tiền xử lý dữ liệu, xử lý nhiễu/missing, đến tối ưu hóa tham số bằng Optuna đã mang lại một mô hình XGBoost mạnh mẽ với độ chính xác ~93%. Việc kết hợp phân tích SHAP không chỉ xác nhận độ chính xác về mặt toán học mà còn đảm bảo tính hợp lý về mặt y khoa (các chỉ số quan trọng thực sự chi phối kết quả dự đoán), đáp ứng tốt yêu cầu của bài toán hỗ trợ chẩn đoán bệnh.

# PHẦN V: DỰ ĐOÁN KẾT QUẢ TRÊN TẬP DỮ LIỆU MỚI
"""

import numpy as np
import joblib

# Trích xuất model tốt nhất để dự đoán (lưu ý model này đã được refit trên toàn bộ tập huấn luyện và đánh giá)
best_model = trainer.models[best_model_name]

# Mã hóa biến phân loại
prepro_na_scaled.label_encode(["Gender"])
X_missing = prepro_na_scaled.drop_columns(columns=['Medical Condition'])

# Thực hiện dự đoán
print("Thực hiện dự đoán")
y_pred_encoded = best_model.predict(X_missing.df)

# Giải mã (Inverse Transform) từ số về chữ (Tên bệnh)
le_target = prepro_not_na_scaled.encoders.get(('Medical Condition', 'label'))
if le_target:
    y_pred_label = le_target.inverse_transform(y_pred_encoded.astype(int))
    print("Đã giải mã thành công tên bệnh.")
else:
    print("Cảnh báo: Không tìm thấy Encoder. Kết quả sẽ giữ dạng số.")
    y_pred_label = y_pred_encoded

# Gán kết quả vào DataFrame
df_filled = pre_na.get_df()
df_filled['Medical Condition'] = y_pred_label

# Lưu kết quả cuối cùng
output_filename = "final_predicted_data.csv"
df_filled.to_csv(output_filename, index=False)

print(f"\n--- HOÀN TẤT ---")
print(f"Đã điền khuyết cho {len(df_filled)} dòng.")
print(f"File kết quả đã lưu tại: {output_filename}")

df_original_labeled = pre_not_na.get_df()

# Gộp 2 bảng lại
df_full_final = pd.concat([df_original_labeled, df_filled], axis=0)

# Lưu file tổng
df_full_final.to_csv("full_dataset_completed.csv", index=False)
print(f"Đã tạo file tổng hợp {len(df_full_final)} dòng: full_dataset_completed.csv")