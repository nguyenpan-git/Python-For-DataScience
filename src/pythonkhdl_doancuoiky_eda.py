import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
from scipy.stats import ks_2samp, chi2_contingency
from IPython.display import display


# NHÓM 1: KHỞI TẠO BAN ĐẦU VÀ CÁC KHÁM PHÁ CƠ BẢN TRÊN DATAFRAME
class EDA:
    """
    Lớp hỗ trợ các bước EDA (Exploratory Data Analysis) cơ bản:
    - Khởi tạo từ file CSV hoặc từ DataFrame có sẵn.
    - Hiển thị dataframe, thông tin, kích thước.

    Cách dùng nhanh:
    ----------------
    # Cách 1: Khởi tạo từ file CSV
    eda = EDA(file_path="data.csv")

    # Cách 2: Khởi tạo từ DataFrame có sẵn
    eda = EDA(df=your_dataframe)

    # Các hàm:
    eda.display_dataframe()
    eda.display_info()
    eda.display_shape()
    """

    def __init__(self, file_path: str = None, df: pd.DataFrame = None):
        """
        Khởi tạo đối tượng EDA.

        Tham số:
            file_path (str): Đường dẫn CSV. Nếu dùng, không truyền df.
            df (pd.DataFrame): DataFrame có sẵn. Nếu dùng, không truyền file_path.

        Ngoại lệ:
            ValueError: Khi truyền cả hai hoặc không truyền cái nào.
        """

        if file_path is None and df is None:
            raise ValueError("Cần chọn 'file_path' hoặc 'df'.")
        if file_path is not None and df is not None:
            raise ValueError("Không được chọn đồng thời cả 'file_path' và 'df'.")

        if file_path is not None:
            self.df: pd.DataFrame = pd.read_csv(file_path)
        else:
            self.df: pd.DataFrame = df

    def display_dataframe(self) -> None:
        """In toàn bộ DataFrame."""
        print("\n--- DataFrame ---")
        self.df

    def display_info(self) -> None:
        """In thông tin tổng quát."""
        print("\n--- Thông tin của dataframe: ---\n")
        self.df.info()

    def display_shape(self) -> None:
        """In kích thước DataFrame dạng (số dòng, số cột)."""
        print("\n--- Kích thước của dataframe ---")
        print(self.df.shape)

# NHÓM 2: DỮ LIỆU TRÙNG LẶP VÀ DỮ LIỆU TRỐNG
class DuplicateAndNullAnalyzer(EDA):
    """
    Lớp mở rộng từ EDA, thêm các chức năng:
    duplicate_check(): kiểm tra trùng lặp
    missing_check(): kiểm tra dữ liệu thiếu
    all_empty(): kiểm tra tính trống đồng bộ của dữ liệu
    noise_analyze(): phân tích các cột gây nhiễu
    null_analyze(): phân tích dữ liệu trống

    Cách dùng: Truyền vào dataframe gốc và sử dụng các phương thức nội bộ để khám phá
    """

    def __init__(self, df: pd.DataFrame):
        """Khởi tạo lớp từ một DataFrame."""
        super().__init__(df=df)

    # ---------------------------------------------
    def duplicate_check(self) -> None:
        """
        Kiểm tra dữ liệu trùng lặp trong toàn bộ DataFrame.
        In số lượng và hiển thị các dòng bị trùng.
        """
        print("\n--- Kiểm tra dữ liệu trùng lặp ---")
        duplicated = self.df.duplicated().sum()
        print(f"Tổng số hàng trùng lặp: {duplicated}")

        if duplicated > 0:
            print("\n--- Các dòng trùng lặp: ---")
            display(self.df[self.df.duplicated(keep=False)])

    # ---------------------------------------------
    def missing_check(self) -> None:
        """
        Kiểm tra số lượng giá trị thiếu theo từng cột.
        """
        print("\n--- Kiểm tra dữ liệu thiếu ---")
        null_by_cols = self.df.isnull().sum()
        print(null_by_cols)

    # ---------------------------------------------
    def all_empty(self, cols_to_check: list[str]) -> None:
        """
        Kiểm tra xem có dòng nào trống đồng bộ trên một nhóm cột hay không.

        Args:
            cols_to_check (list[str]): Danh sách các cột cần kiểm tra.
        """
        print(f"\n--- Kiểm tra trống đồng bộ trên: {', '.join(cols_to_check)} ---")

        condition = pd.Series([True] * len(self.df), index=self.df.index)

        for col in cols_to_check:
            if col in self.df.columns:
                condition &= self.df[col].isna()
            else:
                print(f"Cảnh báo: '{col}' không tồn tại. Bỏ qua cột này.")

        empty_sync_rows = self.df[condition]

        if not empty_sync_rows.empty:
            print(f"Tìm thấy {empty_sync_rows.shape[0]} dòng trống đồng bộ:")
            empty_sync_rows
        else:
            print("Không có dòng trống đồng bộ.")

    # ---------------------------------------------
    def noise_analyze(self) -> None:
        """
        Phân tích các cột nhiễu:
        - random_notes: giá trị xuất hiện.
        - noise_col: histogram + KDE + đường chuẩn.
        """
        print("\n--- Bắt đầu phân tích dữ liệu nhiễu ---")

        initial_empty_df = self.df[self.df.isna().any(axis=1)].copy()

        # random_notes (vẽ phân phối)
        print("\n--- Phân tích random_notes ---")
        if "random_notes" in initial_empty_df:
            print(initial_empty_df["random_notes"].value_counts())
        else:
            print("Không có cột random_notes.")

        # noise_col (vẽ phân phối)
        print("\n--- Phân tích noise_col ---")
        if "noise_col" in initial_empty_df:
            col = initial_empty_df["noise_col"]
            mu, sigma = col.mean(), col.std()

            x = np.linspace(col.min(), col.max(), 100)
            pdf_normal = norm.pdf(x, mu, sigma)

            plt.figure(figsize=(10, 6))
            col.hist(bins=30, density=True, alpha=0.6, label="Histogram")
            sns.kdeplot(col, linewidth=2, label="KDE")
            plt.plot(x, pdf_normal, "r--", label="Chuẩn")

            plt.title("Phân phối noise_col")
            plt.legend()
            plt.grid(True, alpha=0.4)
            plt.show()
        else:
            print("Không có cột noise_col.")

    # ---------------------------------------------
    def null_analyze(self, cols: list[str] = None) -> None:
        """
        Phân tích tác động của giá trị thiếu trên từng cột:
        - So sánh mean/median/variance của nhóm không thiếu với nhóm chỉ thiếu đúng 1 cột.
        - So sánh phân phối categorical.

        Args:
            cols (list[str], optional):
                Danh sách cột để phân tích.
                Nếu None → tự động chọn các cột có NA.
        """
        print("\n=== PHÂN TÍCH GIÁ TRỊ THIẾU ===")

        if cols is None:
            cols = self.df.columns[self.df.isna().any()].tolist()
            print(f"Tự động chọn: {cols}")
        else:
            print(f"Phân tích: {cols}")

        df_no_na = self.df.dropna().copy()

        # Các cột số
        numeric_cols = [
            c for c in self.df.select_dtypes(include=np.number).columns
            if c not in ["noise_col", "random_notes"]
        ]

        for col in cols:
            print(f"\n--- Cột: {col} ---")

            if col not in self.df:
                print(f"'{col}' không tồn tại.")
                continue

            rows_with_na = self.df[self.df[col].isna()]
            other_cols = [c for c in self.df.columns if c != col]

            # Hàng chỉ thiếu ở đúng 1 cột đang xét
            exclude_one_col_df = rows_with_na[
                rows_with_na[other_cols].notna().all(axis=1)
            ]

            print(f"Số hàng chỉ thiếu '{col}': {exclude_one_col_df.shape[0]}")

            if exclude_one_col_df.empty:
                continue

            # Các cột có ở dataset chứa null và không chứa null
            common_numeric = list(
                set(numeric_cols)
                & set(exclude_one_col_df.columns)
                & set(df_no_na.columns)
            )

            if common_numeric:
                print("\nSai số tương đối (mean/median/var):")
                stats_no_na = df_no_na[common_numeric].agg(["mean", "median", "var"])
                stats_missing = exclude_one_col_df[common_numeric].agg(["mean", "median", "var"])
                print((stats_missing - stats_no_na) / stats_no_na)


            # categorical
            def compare(cat):
                if cat in exclude_one_col_df and cat in df_no_na:
                    print(f"\nPhân phối '{cat}' (nhóm thiếu {col}):")
                    print(exclude_one_col_df[cat].value_counts(normalize=True))

                    print(f"\nPhân phối '{cat}' (nhóm đầy đủ):")
                    print(df_no_na[cat].value_counts(normalize=True))

            for c in ["Gender", "Medical Condition"]:
                if c != col:
                    compare(c)
                else:
                    compare(col)

class DistributionAnalyzer(EDA):
    """
    Lớp phân tích phân phối dữ liệu, kế thừa từ EDA, thêm các chức năng:

    describe_numerical_data(): thống kê mô tả dữ liệu số
    plot_numerical_density(): vẽ biểu đồ mật độ
    plot_numerical_boxplot(): vẽ boxplot cho dữ liệu số
    describe_categorical_data(): thống kê phân phối biến phân loại
    plot_categorical_distribution(): trực quan hóa biến phân loại
    plot_binary_distribution(): biểu đồ phân phối biến nhị phân
    find_outliers(): tìm giá trị ngoại lai theo IQR
    analyze_outlier_impact(): phân tích ảnh hưởng của outlier
    min_max_values(): xem giá trị nhỏ nhất và lớn nhất
    check_negative_values(): kiểm tra giá trị âm

    Cách dùng: Truyền vào dataframe gốc và sử dụng các phương thức nội bộ để phân tích phân phối,
    trực quan hóa dữ liệu số/phân loại, đánh giá giá trị ngoại lai, và kiểm tra tính hợp lệ của dữ liệu.

    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Khởi tạo với một DataFrame đã có.

        Args:
            df (pd.DataFrame): DataFrame đầu vào cần phân tích.
        """
        super().__init__(df=df)

    # ---------------------- DỮ LIỆU SỐ ----------------------
    def describe_numerical_data(self) -> None:
        """
        In thống kê mô tả cơ bản cho các cột dữ liệu số.
        """
        print("\n--- Thống kê mô tả cho dữ liệu số ---")
        print(self.df.describe())

    def plot_numerical_density(self) -> None:
        """
        Vẽ biểu đồ mật độ (density plot) cho tất cả các cột dạng số.
        """
        print("\n--- Biểu đồ mật độ ---")
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        num_plots = len(numerical_cols)
        fig_rows = int(np.ceil(num_plots / 3))

        self.df.select_dtypes('number').plot(
            kind='density',
            figsize=(20, fig_rows * 6),
            subplots=True,
            layout=(fig_rows, 3),
            title="Density plot of Numerical features",
            sharex=False
        )
        plt.tight_layout()
        plt.show()

    def plot_numerical_boxplot(self) -> None:
        """
        Vẽ boxplot cho các cột dạng số để quan sát phân phối và ngoại lai.

        Lưu ý:
            - Tự động tổ chức subplot 3 cột mỗi hàng.
        """
        print("\n--- Boxplot ---")
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        num_plots = len(numerical_cols)
        fig_rows = int(np.ceil(num_plots / 3))

        self.df.select_dtypes('number').plot(
            kind='box',
            figsize=(16, fig_rows * 5),
            subplots=True,
            layout=(fig_rows, 3),
            title="Box plot of Numerical features",
            sharex=False
        )
        plt.tight_layout()
        plt.show()

    # ---------------------- DỮ LIỆU PHÂN LOẠI ----------------------
    def describe_categorical_data(self) -> None:
        """
        Thống kê mô tả cho các biến phân loại (object).

        In ra tần suất (frequency) và tỷ lệ (%).
        """
        print("\n--- Thống kê biến phân loại ---")
        categorical_cols = self.df.select_dtypes(include='object').columns

        for col in categorical_cols:
            print(f"\nPhân phối của {col}:")
            print(self.df[col].value_counts(dropna=False, normalize=True))

    def plot_categorical_distribution(self) -> None:
        """
        Vẽ biểu đồ phân phối (countplot) cho các cột dạng phân loại.

        Lưu ý:
            - Tự động chia subplot 3 cột.
            - Xoay nhãn trục X cho dễ nhìn.
        """
        print("\n--- Phân phối cho biến phân loại ---")
        categorical_cols = self.df.select_dtypes(include='object').columns

        if len(categorical_cols) == 0:
            print("Không có biến phân loại.")
            return

        num_plots = len(categorical_cols)
        fig_rows = int(np.ceil(num_plots / 3))

        plt.figure(figsize=(16, fig_rows * 5))
        for i, col in enumerate(categorical_cols):
            plt.subplot(fig_rows, 3, i + 1)
            sns.countplot(x=col, data=self.df, palette='viridis')
            plt.title(f'Phân phối của {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    # ---------------------- NHỊ PHÂN ----------------------
    def plot_binary_distribution(self, binary_cols: list[str]) -> None:
        """
        Vẽ biểu đồ countplot cho các biến nhị phân (giá trị 0/1).

        Args:
            binary_cols (list[str]): Danh sách các cột nhị phân.
        """
        print("\n--- Phân phối cho biến nhị phân ---")

        if not binary_cols:
            print("Không có biến nhị phân.")
            return

        num_plots = len(binary_cols)
        fig_rows = int(np.ceil(num_plots / 3))

        plt.figure(figsize=(15, fig_rows * 5))
        for i, col in enumerate(binary_cols):
            if col in self.df.columns:
                plt.subplot(fig_rows, 3, i + 1)
                sns.countplot(x=col, data=self.df, palette='viridis')
                plt.title(f'Phân bố của {col}')
                plt.xlabel(col)
                plt.ylabel('Số lượng')
                plt.xticks([0, 1], ['Không', 'Có'])
            else:
                print(f"Không có cột '{col}'.")

        plt.tight_layout()
        plt.show()

    # ---------------------- NGOẠI LAI ----------------------
    def find_outliers(self, col_name: str) -> pd.DataFrame:
        """
        Tìm các giá trị ngoại lai theo phương pháp IQR.

        Args:
            col_name (str): Tên cột cần tìm ngoại lai.

        Returns:
            pd.DataFrame: Các bản ghi chứa ngoại lai.
        """
        q1 = self.df[col_name].quantile(0.25)
        q3 = self.df[col_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return self.df[
            (self.df[col_name] < lower_bound) |
            (self.df[col_name] > upper_bound)
        ]

    def analyze_outlier_impact(
        self,
        numerical_col: str,
        categorical_col: str = 'Medical Condition',
        return_outliers: bool = False
    ) -> pd.DataFrame | None:
        """
        Phân tích mối quan hệ giữa ngoại lai của biến số và một biến phân loại.

        Args:
            numerical_col (str): Tên cột số cần xem ngoại lai.
            categorical_col (str): Biến phân loại để so sánh.
            return_outliers (bool): Trả về DataFrame ngoại lai nếu True.

        Returns:
            pd.DataFrame | None: DataFrame ngoại lai (tùy chọn).
        """
        print(f"\n--- Phân tích giá trị ngoại lai cho {numerical_col} và quan hệ với {categorical_col} ---")

        if numerical_col not in self.df.columns:
            print(f"Error: Không tìm thấy '{numerical_col}'.")
            return
        if categorical_col not in self.df.columns:
            print(f"Error: Không tìm thấy '{categorical_col}'.")
            return

        outliers_df = self.find_outliers(numerical_col)

        if not outliers_df.empty:
            print(f"Số giá trị '{categorical_col}' trong nhóm ngoại lai:")
            print(outliers_df[categorical_col].value_counts(dropna=False))

            print(f"\nMin ngoại lai của {numerical_col}: {outliers_df[numerical_col].min()}")
            print(f"Max ngoại lai của {numerical_col}: {outliers_df[numerical_col].max()}")
        else:
            print(f"Không có ngoại lai cho '{numerical_col}'.")

        return outliers_df if return_outliers else None

    # ---------------------- GIÁ TRỊ CỰC TRỊ ----------------------
    def min_max_values(self, numerical_col: str, n: int = 50) -> None:
        """
        In ra n giá trị nhỏ nhất và lớn nhất của một cột.

        Args:
            numerical_col (str): Cột dạng số.
            n (int): Số lượng giá trị muốn xem.
        """
        print('\nIn top giá trị thấp và cao nhất:')
        print(self.df[numerical_col].dropna().sort_values().head(n).values)
        print(self.df[numerical_col].dropna().sort_values().tail(n).values)

    # ---------------------- GIÁ TRỊ ÂM ----------------------
    def check_negative_values(self, cols_to_check: list[str]) -> None:
        """
        Kiểm tra các cột xem có xuất hiện giá trị âm hay không.

        Args:
            cols_to_check (list[str]): Danh sách các cột dạng số cần kiểm tra.
        """
        print("\n--- Kiểm tra giá trị âm ---")
        for col in cols_to_check:
            if col in self.df.columns and self.df[col].dtype in ['float64', 'int64']:
                negative_values = self.df[self.df[col] < 0]

                if not negative_values.empty:
                    print(f"Giá trị âm trong '{col}': {negative_values[col].unique()}")
                    print(f"Số lượng: {len(negative_values)}")
                else:
                    print(f"Không có giá trị âm trong '{col}'.")
            else:
                print(f"Column '{col}' not found hoặc không phải dạng số.")

# NHÓM 4: PHÂN TÍCH TƯƠNG QUAN
class CorrelationAnalyzer(EDA):
    """
    Lớp mở rộng từ EDA, thêm các chức năng:
        - plot_correlation_matrix(): vẽ heatmap ma trận tương quan giữa các biến số
        - plot_target_correlation_heatmap(): vẽ heatmap tương quan giữa các biến và biến mục tiêu đã mã hóa
        - analyze_target_correlation(): phân tích tương quan với biến mục tiêu

    Cách dùng: Truyền vào dataframe gốc và sử dụng các phương thức nội bộ để phân tích
               mối quan hệ giữa các biến, trực quan hóa tương quan tổng thể và tương quan với biến mục tiêu.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Khởi tạo lớp CorrelationAnalyzer.

        Tham số:
            df (pd.DataFrame): DataFrame cần phân tích tương quan.
        """
        super().__init__(df=df)

    def plot_correlation_matrix(self) -> None:
        """
        Vẽ heatmap ma trận tương quan giữa tất cả các biến số trong DataFrame.

        - Tự động loại bỏ các giá trị NULL.
        - Chỉ tính trên các biến dạng số (numeric).
        - Kết quả là một heatmap trực quan.

        Trả về:
            None
        """
        print("\n--- Ma trận tương quan và heatmap ---")
        correlation_matrix = self.df.dropna().select_dtypes(include=np.number).corr()
        plt.figure(figsize=(18, 15))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title('Ma trận tương quan giữa các biến số', fontsize=16)
        plt.show()

    def plot_target_correlation_heatmap(self, correlations: pd.Series, target_col: str) -> None:
        """
        Vẽ heatmap tương quan giữa các đặc trưng và biến mục tiêu đã mã hóa.

        Tham số:
            correlations (pd.Series): Các hệ số tương quan đã tính.
            target_col (str): Tên cột mục tiêu đã được mã hóa.

        Trả về:
            None
        """
        print(f"\n--- Heatmap Tương quan với '{target_col}' đã mã hóa ---")

        corr_df = correlations.to_frame(name=target_col)

        plt.figure(figsize=(4, len(corr_df) * 0.7))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.3f', linewidths=.5, cbar=False)
        plt.title(f'Correlation with {target_col}', fontsize=12)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def analyze_target_correlation(
        self,
        target_col: str = 'Medical Condition',
        numerical_cols: list[str] | None = None
    ) -> None:
        """
        Phân tích tương quan giữa các biến số và biến mục tiêu.

        Quy trình:
            - Sao chép dataframe để tránh thay đổi dữ liệu gốc.
            - Điền NA cho: Age, Gender, Glucose, Blood Pressure.
            - Loại bỏ mọi hàng còn NA.
            - LabelEncoding biến mục tiêu nếu là dạng object.
            - Tính toán hệ số tương quan giữa các biến số và biến mục tiêu đã mã hóa.
            - Gọi heatmap hiển thị mức độ tương quan.

        Tham số:
            target_col (str): Cột mục tiêu để phân tích.
            numerical_cols (list[str] | None):
                Danh sách tên cột số để phân tích.
                Nếu None → tự động lấy toàn bộ biến số.

        Trả về:
            None
        """
        print(f"\n--- Phân tích tương quan với biến mục tiêu: '{target_col}' ---")
        df_copy = self.df.copy()

        # Điền giá trị thiếu
        if 'Age' in df_copy.columns:
            df_copy['Age'] = df_copy['Age'].fillna(df_copy['Age'].median())

        if 'Gender' in df_copy.columns:
            random_gender_choice = np.random.choice(['Male', 'Female'])
            df_copy['Gender'] = df_copy['Gender'].fillna(random_gender_choice)

        if 'Glucose' in df_copy.columns:
            df_copy['Glucose'] = df_copy['Glucose'].fillna(df_copy['Glucose'].median())

        if 'Blood Pressure' in df_copy.columns:
            df_copy['Blood Pressure'] = df_copy['Blood Pressure'].fillna(df_copy['Blood Pressure'].median())

        # Loại bỏ phần còn NA
        df_copy.dropna(inplace=True)

        # Label Encoding biến mục tiêu
        encoded_target_col = target_col + ' Encoded'
        if target_col in df_copy.columns and df_copy[target_col].dtype == 'object':
            le = LabelEncoder()
            df_copy[encoded_target_col] = le.fit_transform(df_copy[target_col])
        else:
            print(f"Cột mục tiêu '{target_col}' không phải dạng object hoặc không tồn tại. Bỏ qua Label Encoding.")
            return

        # Lấy danh sách biến số
        if numerical_cols is None:
            numerical_features = df_copy.select_dtypes(include=np.number).columns.tolist()
            if encoded_target_col in numerical_features:
                numerical_features.remove(encoded_target_col)
        else:
            numerical_features = [col for col in numerical_cols if col in df_copy.columns and col != encoded_target_col]

        # Tính tương quan
        correlations = df_copy[numerical_features + [encoded_target_col]].corr()[encoded_target_col].drop(encoded_target_col)

        # Vẽ heatmap
        self.plot_target_correlation_heatmap(correlations.sort_values(ascending=False), encoded_target_col)

# NHÓM 5: KIỂM ĐỊNH PHÂN PHỐI
class DistributionComparison:
    """
    Lớp có chức năng kiểm tra phân phối giữa 2 nhóm seen_data và unseen_data theo từng cột (đặc trưng số hoặc phân loại)

    Cách dùng: Truyền vào 2 dataframe cần kiểm định (hoặc đường dẫn) và gọi hàm kiểm định
    """
    def __init__(self, seen_csv: str = None, unseen_csv: str = None, seen_df: pd.DataFrame = None, unseen_df: pd.DataFrame = None) -> None:
        """
        Khởi tạo với 2 file CSV hoặc dataframe.

        Args:
            seen_df (pd.DataFrame): đối tượng dataframe truyền vào
            unseen_df (pd.DataFrame): đối tượng dataframe truyền vào
            seen_csv (str): đường dẫn file CSV chứa 85% dữ liệu (có nhãn).
            unseen_csv (str): đường dẫn file CSV chứa 15% dữ liệu (thiếu nhãn).
        """
        if seen_df is not None:
          self.seen_data = seen_df
        else:
          seen_data = pd.read_csv(seen_csv)
          self.seen_data = seen_data

        if unseen_df is not None:
          self.unseen_data = unseen_df
        else:
          unseen_data = pd.read_csv(unseen_csv)
          self.unseen_data = unseen_data

    @property
    def return_seen(self) -> pd.DataFrame:
      """ Trả về dataframe có sẵn"""
      return self.seen_data

    @property
    def return_unseen(self) -> pd.DataFrame:
      """Trả về tập dữ liệu mô phỏng"""
      return self.unseen_data

    # So sánh phân phối hai mẫu dữ liệu (seen_data vs unseen_data)
    def compare_distributions(self, numeric_cols, categorical_cols) -> pd.DataFrame:
        """
        Kiểm định và so sánh phân phối giữa 2 dataframe.
        :param numeric_cols: List tên các cột số
        :param categorical_cols: List tên các cột phân loại
        :return: DataFrame tổng hợp kết quả kiểm định
        Các phương pháp thống kê sử dụng:
          1. Biến số (Numerical): Kiểm định Kolmogorov-Smirnov (KS Test)
            - H0: Không có sự khác biệt về phân phối giữa 2 mẫu.
            - H1: Có sự khác biệt có ý nghĩa thống kê về phân phối giữa 2 mẫu
          2. Biến phân loại (Categorical): Kiểm định Chi-Square
            - H0: Không có sự khác biệt về phân phối tần suất giữa 2 mẫu.
            - H1: Có sự khác biệt có ý nghĩa thống kê về phân phối tần suất giữa 2 mẫu.
        """
        results = []

        print(f"--- BẮT ĐẦU KIỂM ĐỊNH PHÂN PHỐI ---")
        print(f"Số lượng mẫu có sẵn (để huấn luyện và đánh giá): {len(self.seen_data)}")
        print(f"Số lượng mẫu Missing: {len(self.unseen_data)}\n")

        # Kiểm phân phối cho biến số (KS test)
        for col in numeric_cols:
            # Lấy dữ liệu và loại bỏ NaN để tránh lỗi
            data1 = self.seen_data[col].dropna()
            data2 = self.unseen_data[col].dropna()

            # Thực hiện KS Test
            stat, p_value = ks_2samp(data1, data2)

            # Đánh giá
            conclusion = "Giống nhau" if p_value > 0.05 else "Khác nhau"
            results.append({
                'Feature': col,
                'Type': 'Numerical',
                'Test': 'KS Test',
                'P-Value': round(p_value, 4),
                'Conclusion (alpha=0.05)': conclusion
            })

            # Vẽ biểu đồ mật độ (KDE Plot)
            plt.figure(figsize=(8, 4))
            sns.kdeplot(data1, label='df_85 (Seen)', fill=True, alpha=0.3, color='blue')
            sns.kdeplot(data2, label='df_15 (Unseen)', fill=True, alpha=0.3, color='orange')
            plt.title(f'Phân phối biến số: {col} (p={p_value:.4f})')
            plt.legend()
            plt.show()

        # Kiểm định phân phối cho biến phân loại (Chi-square test)
        for col in categorical_cols:
            # Tạo bảng tần suất (tỷ lệ)
            prop_seen = self.seen_data[col].value_counts(normalize=True).sort_index()
            prop_unseen = self.unseen_data[col].value_counts(normalize=True).sort_index()

            # Ghép 2 df để tạo Contingency Table
            tmp1 = pd.DataFrame({col: self.seen_data[col], 'Group': 'Train'})
            tmp2 = pd.DataFrame({col: self.unseen_data[col], 'Group': 'Missing'})
            combined = pd.concat([tmp1, tmp2])

            contingency_table = pd.crosstab(combined[col], combined['Group'])

            # Chi-Square test
            stat, p_value, dof, expected = chi2_contingency(contingency_table)

            # Đánh giá
            conclusion = "Giống nhau" if p_value > 0.05 else "Khác nhau"
            results.append({
                'Feature': col,
                'Type': 'Categorical',
                'Test': 'Chi-Square',
                'P-Value': round(p_value, 4),
                'Conclusion (alpha=0.05)': conclusion
            })

            # Vẽ biểu đồ cột so sánh tỷ lệ
            plt.figure(figsize=(8, 4))
            df_plot = pd.DataFrame({'Train': prop_seen, 'Missing': prop_unseen})
            df_plot.plot(kind='bar', color=['blue', 'orange'], alpha=0.7, ax=plt.gca())
            plt.title(f'Tỷ lệ biến phân loại: {col} (p={p_value:.4f})')
            plt.ylabel('Tỷ lệ (%)')
            plt.xticks(rotation=0)
            plt.legend()
            plt.show()

        print("\n===== HOÀN TẤT KIỂM ĐỊNH =====\n")
        return pd.DataFrame(results)



