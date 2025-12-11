import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import argparse
import sys
import json
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Định nghĩa hàm setup logging toàn cục
def setup_logging(log_file: str = 'training_log.txt'):
    """
    Cấu hình logging. Gọi hàm này 1 lần duy nhất đầu chương trình.
    """
    # Lấy logger root
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Xóa các handler cũ để tránh bị lặp log khi chạy lại cell trong Colab
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter chung
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Handler ghi ra file (FileHandler)
    # mode='a' để nối tiếp log, 'w' để ghi đè mỗi lần chạy mới
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler ghi ra màn hình (StreamHandler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.info(f"--- Đã thiết lập Logging: {log_file} ---")

# Class ModelTrainer
class ModelTrainer:
    """
    Lớp chịu trách nhiệm:
        - Chia dữ liệu
        - Tối ưu tham số (GridSearch hoặc Optuna)
        - Huấn luyện mô hình
        - Đánh giá
        - Lưu mô hình tốt nhất
    """
    def __init__(
        self,
        target_col: str,
        data: pd.DataFrame | None = None,
        feature_cols: list[str] | None = None,
        test_size: float = 0.2,
        random_state: int = 42
        ) -> None:
        """
        Khởi tạo ModelTrainer.
        :param target_col: Tên cột mục tiêu (Label).
        :param data: DataFrame chứa dữ liệu (có thể là None nếu dùng load_data sau).
        :param feature_cols: Danh sách các cột đặc trưng (nếu None sẽ lấy tất cả trừ target).
        """
        self.target_col = target_col
        self.data = data
        self.feature_cols = feature_cols
        self.test_size = test_size
        self.random_state = random_state

        # Lưu trữ dữ liệu
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Lưu trữ kết quả
        self.models = {}         # Chứa các model đã train
        self.best_params = {}    # Chứa tham số tối ưu của từng model
        self.results = {}        # Chứa kết quả đánh giá (accuracy, f1...)

        # Định nghĩa các mô hình mặc định sẽ hỗ trợ
        self.supported_models = {
            'RandomForest': RandomForestClassifier,
            'LogisticRegression': LogisticRegression,
            'SVM': SVC,
            'XGBoost': XGBClassifier
        }

    def load_data(self, filepath: str) -> None:
        """
        Tải dữ liệu từ file (CSV/Excel) và cập nhật vào thuộc tính self.data.
        Tự động reset các tập train/test để đảm bảo tính nhất quán.

        :param filepath: Đường dẫn tới file dữ liệu.
        """
        if not os.path.exists(filepath):
            logging.error(f"File không tồn tại: {filepath}")
            return

        try:
            if filepath.endswith('.csv'):
                self.data = pd.read_csv(filepath)
            elif filepath.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(filepath)
            else:
                # Mặc định thử đọc CSV nếu không rõ đuôi
                self.data = pd.read_csv(filepath)

            logging.info(f"Đã tải dữ liệu từ '{filepath}'. Kích thước: {self.data.shape}")

            # Reset lại dữ liệu train/test để bắt buộc chia lại (split_data) khi train mới
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None

        except Exception as e:
            logging.error(f"Lỗi khi tải dữ liệu từ {filepath}: {e}")

    def split_data(self) -> None:
        """Chia dữ liệu thành tập huấn luyện và kiểm thử."""
        if self.feature_cols:
            X = self.data[self.feature_cols]
        else:
            X = self.data.drop(columns=[self.target_col])

        y = self.data[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify = y
        )

        logging.info(f"Đã chia dữ liệu: Train shape {self.X_train.shape}, Test shape {self.X_test.shape}")

    def get_hyperparameter_grid(self, model_name: str) -> dict:
        """Định nghĩa không gian tham số để tối ưu cho từng mô hình."""
        grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'LogisticRegression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            },
            'XGBoost': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        return grids.get(model_name, {})

    def optimize_params_grid(self, model_name: str, param_grid: dict | None = None) -> None:
        """
        Tối ưu siêu tham số cho một mô hình cụ thể bằng GridSearchCV
        và gọi hàm refit_best_model để lưu kết quả cuối cùng.
        """
        # Kiểm tra xem mô hình có được hỗ trợ không
        if model_name not in self.supported_models:
            logging.warning(f"Mô hình '{model_name}' không được hỗ trợ hoặc chưa cài đặt thư viện.")
            return

        logging.info(f"--- Bắt đầu tối ưu tham số (GridSearch) cho: {model_name} ---")

        # Lấy mô hình cơ sở từ dictionary đã khai báo trong __init__
        base_model = self.supported_models[model_name](random_state=self.random_state)

        # Lấy lưới tham số
        if param_grid is None:
            param_grid = self.get_hyperparameter_grid(model_name)

        # Thiết lập GridSearchCV
        # Vì đây là bài toán phân loại nên scoring mặc định là accuracy
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,                 # Cross-validation 3 lần
            scoring='accuracy',   # Đánh giá bằng độ chính xác
            n_jobs=-1,            # Dùng tất cả CPU core
            verbose=1,
            refit=False
        )

        # Huấn luyện GridSearch
        grid_search.fit(self.X_train, self.y_train)

        # Lưu tham số tốt nhất tìm được vào self.best_params
        self.best_params[model_name] = grid_search.best_params_
        logging.info(f"Tối ưu xong {model_name}. Tham số tốt nhất: {grid_search.best_params_}")
        logging.info(f"Điểm Cross-Validation tốt nhất: {grid_search.best_score_:.4f}")

        # Gọi hàm refit_best_model để huấn luyện lại
        self.refit_best_model(model_name)

    def optimize_params_optuna(self, model_name: str, n_trials: int = 30) -> None:
        """
        Tối ưu siêu tham số sử dụng thư viện Optuna (Bayesian Optimization).
        :param model_name: Tên mô hình ('RandomForest', 'SVM',...)
        :param n_trials: Số lượng thử nghiệm (trials) để tìm bộ tham số tốt nhất.
        """
        # Kiểm tra xem mô hình có được hỗ trợ không
        if model_name not in self.supported_models:
            logging.warning(f"Mô hình '{model_name}' không được hỗ trợ.")
            return

        logging.info(f"--- Bắt đầu tối ưu tham số (Optuna) cho: {model_name} (n_trials={n_trials}) ---")

        # Định nghĩa hàm mục tiêu (Objective Function) cho Optuna
        def objective(trial) -> float:
            params = {}
            model = None

            # Định nghĩa không gian tìm kiếm (Search Space) cho từng loại model
            if model_name == 'RandomForest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                model = RandomForestClassifier(**params)

            elif model_name == 'LogisticRegression':
                params = {
                    'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
                    'max_iter': 10000,
                    'random_state': self.random_state
                }
                model = LogisticRegression(**params)

            elif model_name == 'SVM':
                params = {
                    'C': trial.suggest_float('C', 1e-3, 10.0, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'probability': True, # Bắt buộc True để tính predict_proba nếu cần
                    'random_state': self.random_state
                }
                model = SVC(**params)

            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'eval_metric': 'mlogloss',
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                model = XGBClassifier(**params)

            logging.info(f"[{model_name}] Trial {trial.number}: Đang chạy...")

            try:
                score = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
            except Exception as e:
                logging.error(f"Lỗi Trial {trial.number}: {e}")
                return 0

            logging.info(f"[{model_name}] Trial {trial.number} hoàn tất -> Acc: {score:.4f}")
            return score

        # Tạo Study và chạy tối ưu
        # TPESampler là thuật toán mặc định và tốt nhất của Optuna cho dạng bài này
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        try:
            study.optimize(objective, n_trials=n_trials)
        except Exception as e:
            logging.error(f"Lỗi trong quá trình chạy Optuna cho {model_name}: {e}")
            return

        # Ghi nhận kết quả tốt nhất
        logging.info(f"Optuna hoàn tất. Best params: {study.best_params}")
        logging.info(f"Best CV score: {study.best_value:.4f}")

        self.best_params[model_name] = study.best_params

        # Huấn luyện lại mô hình tốt nhất trên toàn bộ tập Train (Refit)
        self.refit_best_model(model_name)

    def refit_best_model(self, model_name: str) -> None:
        """
        Khởi tạo lại mô hình với bộ tham số tối ưu (từ self.best_params)
        và huấn luyện trên toàn bộ tập Train (X_train, y_train).
        """
        logging.info(f"--- Refit: Huấn luyện lại {model_name} với tham số tối ưu ---")

        params = self.best_params.get(model_name)

        if params is None:
            logging.error(f"Không có best params cho {model_name}")
            return

        model_class = self.supported_models[model_name]
        if 'random_state' not in params:
            params = {**params, 'random_state': self.random_state}
        model = model_class(**params)
        model.fit(self.X_train, self.y_train)

        self.models[model_name] = model

        logging.info(f"Refit hoàn tất → {model_name}")

    def train_default_params(self, model_name: str | list[str] | None) -> None:
        """
        Huấn luyện mô hình với tham số mặc định (không tối ưu).
        """
        if model_name is None:
            # chạy tất cả
            model_list = list(self.supported_models.keys())
        elif isinstance(model_name, str):
            model_list = [model_name]
        elif isinstance(model_name, list):
            model_list = model_name
        else:
            logging.error("model_name phải là str, list[str], hoặc None")
            return

        for name in model_list:
            if name not in self.supported_models:
                logging.warning(f"Mô hình '{name}' không được hỗ trợ.")
                continue

            logging.info(f"--- Huấn luyện mặc định cho: {name} ---")

            # tạo instance mới
            base_model = self.supported_models[name](random_state=self.random_state)

            base_model.fit(self.X_train, self.y_train)

            self.models[name] = base_model
            self.best_params[name] = {}

            logging.info(f"Đã huấn luyện xong {name} với tham số mặc định.")

    def train_model(
        self,
        model_list: list[str] | None = None,
        tuning_method: str = "grid",
        n_trials: int = 30
    ) -> None:
        """
        Hàm điều phối việc huấn luyện.
        :param tuning_method: "grid", "optuna", hoặc None/False (chạy mặc định)
        :param model_list: Danh sách tên mô hình muốn chạy (vd: ['RandomForest', 'SVM']).
                           Nếu None, sẽ chạy tất cả các mô hình được hỗ trợ.
        """
        # Đảm bảo dữ liệu đã được chia
        if self.X_train is None:
            self.split_data()

        # Xác định danh sách các mô hình cần huấn luyện
        if model_list is None:
            # Nếu không chỉ định, lấy tất cả key trong supported_models
            target_models = self.supported_models.keys()
        else:
            target_models = model_list

        # Duyệt qua từng mô hình và xác định phương pháp tối ưu tương ứng
        for name in target_models:
            try:
                # Nếu tuning_method là False hoặc None -> Chạy mặc định
                if tuning_method in (None, False, 'none'):
                    self.train_default_params(name)

                # Các phương pháp tối ưu
                elif tuning_method == "grid":
                    self.optimize_params_grid(name)
                elif tuning_method == "optuna":
                    self.optimize_params_optuna(name, n_trials=n_trials)
                else:
                    logging.error(f"Phương pháp tuning '{tuning_method}' không hợp lệ. Chọn 'grid', 'optuna' hoặc None.")

            except Exception as e:
                logging.error(f"Lỗi khi huấn luyện {name}: {str(e)}")

    def evaluate_all(self) -> pd.DataFrame | None:
        """Đánh giá tất cả các mô hình đã huấn luyện trên tập Test."""
        if not self.models:
            logging.warning("Chưa có mô hình nào được huấn luyện.")
            return

        comparison_data = []

        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')

            self.results[name] = {'accuracy': acc, 'f1_score': f1}
            comparison_data.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1})

            logging.info(f"Kết quả {name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

        # Trả về DataFrame so sánh để dễ quan sát
        return pd.DataFrame(comparison_data).sort_values(by='Accuracy', ascending=False)

    def save_results_to_history(self, tuning_method: str, filename: str ="model_comparison_results.csv") -> None:
        """
        Lưu kết quả chạy hiện tại vào một file CSV lịch sử.
        Nếu file đã tồn tại, sẽ ghi nối tiếp (append).
        :param tuning_method: Tên phương pháp (ví dụ: 'default', 'optuna', 'grid')
        :param filename: Tên file CSV lưu trữ
        """
        current_results = self.evaluate_all()
        if current_results is None or current_results.empty:
            logging.warning("Không có kết quả để lưu.")
            return

        # Chuẩn bị dữ liệu mới
        current_results['Method'] = tuning_method
        current_results['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cols = ['Timestamp', 'Method', 'Model', 'Accuracy', 'F1-Score']

        # Đảm bảo đủ cột
        for col in cols:
            if col not in current_results.columns:
                current_results[col] = None
        new_df = current_results[cols]

        # Xử lý file cũ
        if os.path.exists(filename):
            try:
                old_df = pd.read_csv(filename)
                # Giữ lại những dòng trong file cũ mà (Model, Method) không có trong file mới
                # Tạo một cột key tạm thời để so sánh
                old_df['key'] = old_df['Model'].astype(str) + "_" + old_df['Method'].astype(str)
                new_df_temp = new_df.copy()
                new_df_temp['key'] = new_df_temp['Model'].astype(str) + "_" + new_df_temp['Method'].astype(str)

                # Lọc bỏ những key đã tồn tại trong lần chạy mới
                keys_to_remove = new_df_temp['key'].unique()
                filtered_old_df = old_df[~old_df['key'].isin(keys_to_remove)].drop(columns=['key'])

                # Gộp lại
                final_df = pd.concat([filtered_old_df, new_df], ignore_index=True)

                # Lưu
                final_df.to_csv(filename, index=False)
                logging.info(f"Đã cập nhật kết quả vào {filename}.")
            except Exception as e:
                logging.error(f"Lỗi khi xử lý file lịch sử: {e}")
                # Fallback: lưu file mới
                new_df.to_csv(filename, index=False)
        else:
            new_df.to_csv(filename, index=False)
            logging.info(f"Đã tạo file mới {filename}")

    def plot_confusion_matrix(self, model_name: str) -> None:
        """Vẽ Confusion Matrix cho một mô hình cụ thể."""
        if model_name not in self.models:
            print(f"Model {model_name} chưa được huấn luyện.")
            return

        y_pred = self.models[model_name].predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('Thực tế')
        plt.xlabel('Dự đoán')
        plt.show()

    def plot_feature_importance(self, model_name: str, top_n: int =10) -> None:
        """
        Vẽ biểu đồ các đặc trưng quan trọng nhất (Chỉ áp dụng cho Tree-based models).
        :param model_name: Tên mô hình (RandomForest hoặc XGBoost)
        :param top_n: Số lượng đặc trưng top đầu muốn hiển thị
        """
        if model_name not in self.models:
            logging.warning(f"Mô hình {model_name} chưa được huấn luyện.")
            return

        model = self.models[model_name]

        # Kiểm tra xem model có thuộc tính feature_importances_ không
        if not hasattr(model, 'feature_importances_'):
            logging.warning(f"Mô hình {model_name} không hỗ trợ Feature Importance (hoặc là SVM/LogisticRegression).")
            return

        # Lấy tên các cột đặc trưng
        if self.feature_cols:
            features = self.feature_cols
        else:
            # Lấy tất cả cột ngoại trừ target
            features = self.data.drop(columns=[self.target_col]).columns

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n] # Lấy index của top_n đặc trưng cao nhất

        # Vẽ biểu đồ
        plt.figure(figsize=(10, 6))
        plt.title(f"Top {top_n} Feature Importances - {model_name}")
        sns.barplot(x=importances[indices], y=[features[i] for i in indices], hue=[features[i] for i in indices], palette="viridis", legend=False)
        plt.xlabel("Mức độ quan trọng")
        plt.ylabel("Đặc trưng")
        plt.tight_layout()
        plt.show()

    def save_best_model(self, filepath: str='best_model.pkl') -> None:
        """Tự động chọn mô hình có Accuracy cao nhất và lưu lại."""
        if not self.results:
            logging.warning("Chưa có kết quả đánh giá.")
            return

        best_model_name = max(self.results, key=lambda k: self.results[k]['accuracy'])
        best_model = self.models[best_model_name]

        joblib.dump(best_model, filepath)
        logging.info(f"Đã lưu mô hình tốt nhất ({best_model_name}) vào {filepath}")

    def save_best_params(self, filepath: str = 'best_hyperparameters.json') -> None:
        """
        Lưu bộ tham số tối ưu (Best Params) của tất cả các model ra file JSON.
        """
        if not self.best_params:
            logging.warning("Chưa có tham số tối ưu nào để lưu.")
            return
        try:
            # Mở file để ghi
            with open(filepath, 'w', encoding='utf-8') as f:
                # default=str: Giúp chuyển các kiểu dữ liệu lạ của numpy thành string để không bị lỗi
                json.dump(self.best_params, f, indent=4, default=str)

            logging.info(f"Đã lưu chi tiết tham số tối ưu vào file: {filepath}")
        except Exception as e:
            logging.error(f"Lỗi khi lưu file tham số: {str(e)}")

    @staticmethod
    def load_model(filepath: str):
        """Load mô hình từ file (Static method)."""
        model = joblib.load(filepath)
        logging.info(f"Đã tải mô hình từ: {filepath}")
        return model

# Phần xử lý argparse và chạy script
if __name__ == "__main__":
    # Khởi tạo parser
    parser = argparse.ArgumentParser(description="Script huấn luyện mô hình Machine Learning tự động.")

    # Định nghĩa các tham số đầu vào
    parser.add_argument('--data_path', type=str, required=True, help="Đường dẫn file dữ liệu (.csv)")
    parser.add_argument('--target_col', type=str, required=True, help="Tên cột Label (mục tiêu)")
    parser.add_argument('--model', type=str, default=None, help="Tên model (RandomForest, SVM, XGBoost...). Bỏ trống để chạy hết.")
    parser.add_argument('--method', type=str, default='grid', choices=['grid', 'optuna', 'none'], help="Phương pháp tối ưu (grid/optuna/none)")
    parser.add_argument('--n_trials', type=int, default=30, help="Số lần thử nghiệm cho Optuna (mặc định 30)")
    parser.add_argument('--output_model', type=str, default='best_model.pkl', help="Tên file lưu model tốt nhất")
    parser.add_argument('--output_params', type=str, default='best_hyperparameters.json', help="File lưu tham số tối ưu")
    parser.add_argument('--output_results', type=str, default='experiment_results.csv', help="Tên file lưu kết quả đánh giá")
    parser.add_argument('--log_file', type=str, default='training_log.txt', help="Tên file log")

    # Đọc tham số từ dòng lệnh
    args = parser.parse_args()

    # Cấu hình Logging
    setup_logging(args.log_file)

    logging.info("--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN ---")
    logging.info(f"Cấu hình: Method={args.method}, Model={args.model}, FileData={args.data_path}")

    try:
        # Khởi tạo Trainer và load dữ liệu
        trainer = ModelTrainer(data=None, target_col=args.target_col)
        trainer.load_data(args.data_path)

        # Xác định danh sách model
        model_list = [args.model] if args.model else None

        # Huấn luyện
        trainer.train_model(model_list=model_list, tuning_method=args.method, n_trials=args.n_trials)

        # Đánh giá và in ra màn hình
        results_df = trainer.evaluate_all()

        if results_df is not None:
            print("\n--- BẢNG KẾT QUẢ ---")
            print(results_df)

            # Lưu lại kết quả thí nghiệm
            results_df['Method'] = args.method
            results_df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_exists = os.path.isfile(args.output_results)
            results_df.to_csv(
                args.output_results,
                mode='a',
                header=not file_exists, # Chỉ ghi header nếu file chưa tồn tại
                index=False
            )
            logging.info(f"Đã cập nhật kết quả vào file chung: {args.output_results}")

            # Lưu model tốt nhất
            trainer.save_best_model(args.output_model)

            # Lưu tham số tốt nhất theo từng model
            trainer.save_best_params(args.output_params)

        else:
            logging.warning("Không có kết quả để lưu.")

    except FileNotFoundError:
        logging.error(f"Không tìm thấy file dữ liệu: {args.data_path}")
    except Exception as e:
        logging.error(f"Đã xảy ra lỗi: {str(e)}")
        import traceback
        traceback.print_exc()