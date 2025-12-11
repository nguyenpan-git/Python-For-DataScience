import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import IsolationForest
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
class DataPreprocessor:
  """
  Lớp tiền xử lý dữ liệu:
  - Đọc dữ liệu từ nhiều định dạng
  - Kiểm tra và xử lý dữ liệu bị thiếu, âm, ngoại lai, trùng lặp
  - Chuẩn hoá
  - Mã hoá biến phân loại
  - Biến đổi ngược về giá trị gốc từ LabelEncoder
  - Tạo đặc trưng datetime
  - Tự động phát hiện kiểu dữ liệu
  - Xoá cột tuỳ chọn
  - Tách thành 2 DataFrame dựa trên giá trị NA trên 1 cột chỉ định
  - Ghép 2 file theo index gốc
  - Ghi dữ liệu ra file
  - Lấy ra một bản DataFrame sau khi preprocessing
  - Phân loại BMI
  - Xử lý text
  """

  # Khởi tạo đối tượng, lưu DataFrame gốc và các bộ lưu trạng thái
  def __init__(self, df: pd.DataFrame | None = None):
    self.df = df

    self.scalers = {} # lưu các bộ chuẩn hoá đã fit trên dữ liệu train -> dùng lại để transform dữ liệu test cho nhất quán
    self.encoders = {} # lưu các bộ mã hoá đã fit trên dữ liệu train -> dùng lại cho test để không bị lệch nhãn / thiếu cột
    self.outlier_masks = {} # chọn hoặc bỏ phần tử của dữ liệu theo True/False, lưu vị trí outlier đã phát hiện ở train -> áp dụng xử lý giống nhau cho test để tránh sai lệch phân phối

  # Dùng khi in đối tượng, giúp xem nhanh trạng thái
  def __repr__(self):
    if self.df is None:
      return "DataPreprocessor(df=None)"
    return f"DataPreprocessor(shape={self.df.shape})"

  def display_dataframe(self) -> None:
      """In toàn bộ DataFrame."""
      print("\n--- DataFrame ---")
      self.df

  def display_info(self) -> None:
      """In thông tin tổng quát."""
      print("\n--- Thông tin của dataframe: ---\n")
      self.df.info()

  # Đọc dữ liệu
  @classmethod # không cần tạo object trước, có thể gọi bằng class
  def from_file(cls, path: str, index_col: int | None = None, **kwargs):
    """
    index_col: cột dùng làm index (giống tham số của pandas)
    - None: không dùng cột nào làm index (mặc định)
    - 0: cột đầu tiên trong file
    """
    try:
      if path.endswith(".csv"):
        df = pd.read_csv(path, index_col=index_col, **kwargs) # **kwargs dùng để truyền thêm các tham số tuỳ chọn (ex: sep, encoding...)
      elif path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path, index_col=index_col, **kwargs) # **kwargs ví dụ sheet_name=...
      elif path.endswith(".json"):
        df = pd.read_json(path, **kwargs)
      else:
        raise ValueError("Định dạng file chưa hỗ trợ!")
    except Exception as e:
      raise RuntimeError(f"Lỗi khi đọc file: {e}")

    # Gọi constructor của class, tạo một instance mới của class, truyền df vào __init__ và trả về object đã tạo
    return cls(df)

  # Xử lý giá trị thiếu
  def handle_missing(self, strategy: str = "mean", columns: list[str] = None):
    """
        Xử lý giá trị thiếu (Missing Values) trong DataFrame bằng các chiến lược
        Tham số:
            strategy (str):
                Phương pháp xử lý giá trị thiếu. Các giá trị chấp nhận:
                - 'mean': điền trung bình
                - 'median': điền trung vị
                - 'mode': điền giá trị xuất hiện nhiều nhất
                - 'forward-fill': điền giá trị gần nhất phía trên cùng cột
                - 'backward-fill': điền giá trị gần nhất phía dưới cùng cột
                - 'random-fill': điền ngẫu nhiên
                - 'drop': loại bỏ hoàn toàn các cột có chứa giá trị thiếu.
                Mặc định là 'mean'.
            columns (list[str] | str | None):
                Danh sách các cột cần áp dụng xử lý. Nếu là None, áp dụng cho tất cả các cột.
        """
    if self.df is None:
      raise ValueError("Chưa có DataFrame!")

    if columns is None:
        columns = list(self.df.columns)
    elif isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)

     # Đếm tổng số giá trị NA trong các cột cần xử lý
    na_count = 0
    for col in columns:
        na_count += self.df[col].isna().sum()

    # Nếu không có NA → thông báo rồi dừng
    if na_count == 0:
        print("Không có giá trị thiếu trong các cột cần xử lý")
        return self

    # Nếu người dùng muốn drop cột thay vì fill
    if strategy == "drop":
      cols_drop = [col for col in columns if self.df[col].isna().sum() > 0]
      self.df.drop(columns=cols_drop, inplace=True)

      return self

    for col in columns:
      if strategy == "mean":
        # Kiểm tra kiểu dữ liệu có phải số không, chỉ cột số mới có ý nghĩa lấy trung bình
        if pd.api.types.is_numeric_dtype(self.df[col]):
          val = self.df[col].mean()
          self.df[col].fillna(val, inplace=True)

      elif strategy == "median":
        # Tương tự cũng kiểm tra cột số như trên
        if pd.api.types.is_numeric_dtype(self.df[col]):
          val = self.df[col].median()
          self.df[col].fillna(val, inplace=True)

      # Áp dụng được cho cả cột số và cột phân loại
      elif strategy == "mode":
        # Lấy Series các mode (có thể nhiều hơn 1 nếu có giá trị nhiều nhất bằng nhau)
        if self.df[col].mode().empty:
          continue
        # Lấy mode đầu tiên trong Series
        val = self.df[col].mode()[0]
        self.df[col].fillna(val, inplace=True)

      # Fill na bằng giá trị gần nhất phía trên cùng cột
      elif strategy == "forward-fill":
        self.df[col].fillna(method="ffill", inplace=True)

      # Fill na bằng giá trị gần nhất phía dưới cùng cột
      elif strategy == "backward-fill":
        self.df[col].fillna(method="bfill", inplace=True)

      elif strategy == "random-fill":
        # Lấy các giá trị khác na trong cột
        nonNullVal = self.df[col].dropna()

        # Nếu cột chỉ toàn Na -> bỏ qua
        if len(nonNullVal) == 0:
          continue

        # Xác định các vị trí NULL
        na_idx = self.df[self.df[col].isna()].index

        # Random lấy các giá trị khác NULL (có thể lặp lại)
        random_val = np.random.choice(nonNullVal, size=len(na_idx), replace=True)

        self.df.loc[na_idx, col] = random_val

      else:
        raise ValueError(f"Strategy {strategy} không hỗ trợ!")

    print(f"Đã xử lý giá trị thiếu ở cột {columns} theo strategy {strategy}")
    return self

  # Xử lý giá trị âm
  def handle_negative(self, strategy: str = "abs", columns: list[str] = None):
    """
    Xử lý giá trị âm trong các cột số với strategy:
    - 'abs': lấy giá trị tuyệt đối
    - 'setZero': giá trị âm thành 0
    - 'dropRow': xoá hàng có giá trị âm
    """
    if self.df is None:
      raise ValueError("Chưa có DataFrame!")

    # Nếu không truyền columns thì xử lý tất cả các cột
    if columns is None:
      columns = self.df.select_dtypes(include='number').columns.tolist()
    elif isinstance(columns, str):
      columns = [columns]

    # Đếm tổng số giá trị âm trong toàn bộ các cột cần xử lý
    negative_count = 0
    for col in columns:
        if pd.api.types.is_numeric_dtype(self.df[col]):
            negative_count += (self.df[col] < 0).sum()

    # Nếu không có giá trị âm
    if negative_count == 0:
        print("Không có giá trị âm trong các cột cần xử lý")
        return self

    for col in columns:
      # Chỉ xử lý cột số
      if not pd.api.types.is_numeric_dtype(self.df[col]):
        continue

      if strategy == "abs":
        self.df[col] = self.df[col].abs()
      elif strategy == "setZero":
        self.df.loc[self.df[col] < 0, col] = 0
      elif strategy == "dropRow":
        tmp = self.df[col] < 0
        self.df = self.df[~tmp].reset_index(drop=True)
      else:
        raise ValueError(f"Strategy {strategy} không hỗ trợ!")

    print(f"Đã xử lý {negative_count} giá trị âm theo strategy {strategy}")
    return self

  # Hàm phụ IQR
  # Hàm không dùng self, không cần truy cập trạng thái của object, có thể gọi trực tiếp bằng class
  @staticmethod
  def _iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return lower, upper

  # Phát hiện ngoại lai
  def detect_outliers_iqr(self, column: str, k: float = 1.5) -> pd.Series:
    """
    Phát hiện ngoại lai trong một cột số bằng phương pháp IQR (Interquartile Range).

    Tham số:
        column (str): Tên cột số cần kiểm tra.
        k (float): Hệ số nhân cho IQR (mặc định là 1.5).

    Trả về:
        pd.Series: Mask Boolean, True là ngoại lai. (Đồng thời lưu mask vào self.outlier_masks).
    """

    # Ở đây, dropna mục đích không phải xử lý giá trị thiếu mà để tránh lỗi thuật toán, nếu đã chủ động xử lý thì sẽ không cần đến câu lệnh này
    s = self.df[column].dropna()
    lower, upper = self._iqr_bounds(s, k)
    mask = (self.df[column] < lower) | (self.df[column] > upper)

    # Dùng dict outlier_masks ở trên để ghi lại kết quả, key là tuple, mask giá trị bool
    # Sau có thể dùng lại mask này cho loại bỏ outlier hoặc thống kê
    self.outlier_masks[(column, "iqr")] = mask
    return mask

  def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> pd.Series:
    """
    Phát hiện ngoại lai trong một cột số bằng phương pháp Z-Score.

    Tham số:
        column (str): Tên cột số cần kiểm tra.
        threshold (float): Ngưỡng Z-score (mặc định là 3.0).

    Trả về:
        pd.Series: Mask Boolean, True là ngoại lai. (Đồng thời lưu mask vào self.outlier_masks).
    """
    s = self.df[column]
    z = np.abs(stats.zscore(s, nan_policy="omit")) # bỏ qua na khi tính, giữ nguyên giá trị na trong ouput zscore
    mask = z > threshold

    self.outlier_masks[(column, "zscore")] = mask
    return mask

  # Phát hiện outlier theo nhiều chiều, mặc định tỉ lệ giả định ước lượng outlier 1% và cố định kết quả random_state=42 để chạy lại ra kết quả giống nhau
  def detect_outliers_isolation_forest(self, columns: list[str], contamination: float =0.01, random_state: int =42) -> pd.Series:
    """
    Phát hiện ngoại lai đa chiều bằng thuật toán Isolation Forest.

    Tham số:
        columns (list[str]): Danh sách các cột số dùng để huấn luyện mô hình.
        contamination (float): Tỷ lệ giả định ước tính của ngoại lai (mặc định 0.01).
        random_state (int): Seed để cố định kết quả (mặc định 42).

    Trả về:
        pd.Series: Mask Boolean, True là ngoại lai. (Đồng thời lưu mask vào self.outlier_masks).
    """
    X = self.df[columns].select_dtypes(include="number").dropna()

    model = IsolationForest(contamination=contamination, random_state=random_state)
    pred = model.fit_predict(X)

    # Tạo 1 Series bool toàn False với index giống df gốc
    mask = pd.Series(False, index = self.df.index)

    # Xác định các dòng được mô hình đánh dấu là outlier (pred = -1)
    # Gán True cho các index đó trong mask, các dòng không nằm trong X (do bị dropna) sẽ giữ nguyên False
    mask.loc[X.index] = pred == -1

    self.outlier_masks[("multi", "isof")] = mask
    return mask

  # Nếu muốn loại bỏ outlier, key đã lưu ví dụ (col, "iqr") hoặc (..., "isof")
  def remove_outliers(self, key):
    """
    Loại bỏ các bản ghi được đánh dấu là ngoại lai dựa trên key đã lưu trong self.outlier_masks.
    """
    mask = self.outlier_masks.get(key)
    if mask is None:
      raise KeyError("Không tìm thấy ngoại lai với key này!")

    # ~mask đảo lại True False và giữ lại toàn bộ hàng không ngoại lai
    self.df = self.df[~mask].reset_index(drop=True)
    return self

  # Xử lý trùng lặp dữ liệu
  def handle_duplicate(self, strategy: str = "keepFirst", columns: list[str] = None):
    """
    Xử lý dữ liệu trùng lặp với strategy
    xoá toàn bộ các dòng trùng (giữ lại 1 bản duy nhất):
    - 'keepFirst': giữ dòng xuất hiện đầu tiên
    - 'keepLast': giữ dòng xuất hiện cuối cùng
    """
    if self.df is None:
      raise ValueError("Chưa có DataFrame!")

    # # Nếu không truyền columns thì xử lý tất cả các cột
    if columns is None:
        columns = list(self.df.columns)
    elif isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)

    # Đếm số dòng trùng lặp nếu không có thì dừng
    dup_count = self.df.duplicated(subset=columns, keep=False).sum()
    if dup_count == 0:
      print("Không có trùng lặp\n")
      return self

    if strategy == "keepFirst":
      self.df = self.df.drop_duplicates(subset=columns, keep="first").reset_index(drop=True)
    elif strategy == "keepLast":
      self.df = self.df.drop_duplicates(subset=columns, keep="last").reset_index(drop=True)
    else:
      raise ValueError(f"Strategy {strategy} không hỗ trợ!")

    print(f"Đã xử lý giá trị trùng lặp theo strategy {strategy}")
    return self

  # Chuẩn hoá dữ liệu
  def scale(self, columns: list[str] | None = None, method: str = "standard"):
    # Nếu không truyền columns, chọn tất cả cột số
    if columns is None:
      columns = self.df.select_dtypes(include='number').columns.tolist()
    elif isinstance(columns, str):
      columns = [columns]

    if method == "standard":
      scaler = StandardScaler()
    elif method == "minmax":
      scaler = MinMaxScaler()
    else:
      raise ValueError("Không hỗ trợ method này!")

    # Học tham số và biến đổi dữ liệu theo tham số đã học
    # Sau khi chạy, các cột được scale sẽ được thay thế trực tiếp trong df
    self.df[columns] = scaler.fit_transform(self.df[columns])

    # Lưu scaler vào dict scalers đã tạo
    # Mục đích để transform test set đúng tham số
    self.scalers[method] = scaler

    print(f"Đã chuẩn hoá dữ liệu số ở cột {columns} theo phương pháp {method}")
    return self

  # Mã hoá biến phân loại thành vector nhị phân bằng OneHotEncoder
  def onehot_encoder(self, columns: list[str], drop_original: bool = True):
    """
    Mã hóa các cột phân loại thành vector nhị phân (One-Hot Encoding) và ghép vào DataFrame.

    Tham số:
        columns (list[str]): Danh sách các cột phân loại cần mã hóa.
        drop_original (bool): True để loại bỏ các cột gốc sau khi mã hóa (mặc định True).
    """
    # sparse_ouput=False: trả về mảng đầy đủ thay vì sparse matrix để dễ đưa vào df
    # handle_unknown="ignore": nếu gặp giá trị mới trong test set mà không có trong train thì bỏ qua, không báo lỗi
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # Chuyển các cột sang kiểu chuỗi để encoder xử lí an toàn
    X_cate = self.df[columns].astype(str)

    arr = ohe.fit_transform(X_cate)
    new_cols = ohe.get_feature_names_out(columns)

    # Chuyển mảng onehot thành df
    df_ohe = pd.DataFrame(arr, columns=new_cols, index=self.df.index)

    # Ghép df gốc với onehot
    self.df = pd.concat([self.df, df_ohe], axis=1)

    # Xoá cột gốc nếu cần
    if drop_original:
      self.df.drop(columns=columns, inplace=True)

    # Lưu encoder để dùng lại cho test set
    self.encoders[("multi", "onehot")] = ohe

    print(f"Đã mã hoá biến phân loại {columns} bằng OneHotEncoder")
    return self

  # Mã hoá biến phân loại thành số nguyên 0, 1, 2 bằng LabelEncoder
  def label_encode(self, columns: list[str]):
    """
    Mã hóa các cột phân loại thành số nguyên (0, 1, 2, ...) bằng LabelEncoder.

    Tham số:
        columns (list[str]): Danh sách các cột phân loại cần mã hóa.
    """
    for col in columns:
      # Tạo 1 LabelEncoder mới cho từng cột
      le = LabelEncoder()

      # Đảm bảo mọi giá trị đều là string
      # Và nếu có na sẽ thay bằng "MISSING" để tránh lỗi khi fit vì LabelEncoder sẽ báo lỗi vì không encode được na
      self.df[col] = self.df[col].astype(str).fillna("MISSING")

      self.df[col] = le.fit_transform(self.df[col])

      # Lưu encoder vào dict encoders đã tạo
      self.encoders[(col, "label")] = le

    print(f"Đã mã hoá biến phân loại {columns} bằng LabelEncoder")
    return self

  # Biến đổi ngược các cột đã LabelEncoder về giá trị gốc
  def inverse_label_encode(self, columns: list[str], inplace: bool = False, restore_nan: bool = True):
    """
    Biến đổi ngược các biến phân loại đã mã hoá bằng LabelEncoder về dạng ban đầu
    - inplace: True thì sẽ ghi đè lên cột hiện tại, False thì tạo cột mới với tên kèm hậu tố là _original
    - restore_nan: True thì giá trị "MISSING" theo như hàm label_encode() hoạt động sẽ được chuyển lại thành NaN, False thì giữ nguyên chuỗi "MISSING"
    """
    if self.df is None:
      raise ValueError("Chưa có DataFrame!")

    # Nếu người dùng truyền chuỗi thay vì list thì đổi thành list
    if columns is None:
      columns = list(self.df.columns)
    elif isinstance(columns, str):
      columns = [columns]
    else:
      columns = list(columns)

    for col in columns:
      # Mỗi cột khi label_encode() được lưu dưới key (col, "label")
      # Lấy đúng encoder cũ để decode lại
      key = (col, "label")
      le = self.encoders.get(key)

      # Lấy dữ liệu đã mã hoá
      encoder_series = self.df[col]

      # Tra ngược lại lớp tương ứng bằng inverse_transform
      original_val = le.inverse_transform(encoder_series.astype(int))

      # Chuyển sang Series kèm index đúng với df
      original_series = pd.Series(original_val, index=self.df.index)

      # Nếu muốn khôi phục NaN
      if restore_nan:
        original_series.replace("MISSING", np.nan, inplace=True)

      # Ghi kết quả
      if inplace:
        # Ghi đè lên cột hiện tại
        self.df[col] = original_series
      else:
        # Tạo cột mới
        new_col = f"{col}_original"
        self.df[new_col] = original_series

    print(f"Đã biến đổi ngược cột {columns} từ LabelEncoder về dạng gốc")
    return self

  # Mã hoá biến phân loại bằng hàm tự định nghĩa chuyển đổi đặc trưng dạng text sang dạng số (ở đây tự định nghĩa là độ dài chuỗi)
  # Hàm này chuyển 1 series dạng text thành độ dài chuỗi của từng phần tử
  @staticmethod
  def text_to_length(series: pd.Series) -> pd.Series:
    return series.astype(str).fillna("").str.len()
  # hàm này tạo ra cột mới chứa các độ dài chuỗi

  # Dùng TF-IDF: mỗi số trong vector cho biết mức độ “quan trọng” của một từ trong dòng đó
  def encode_text_columns(self, columns: list[str]):
    """
    Mã hóa các cột văn bản bằng phương pháp TF-IDF (Term Frequency-Inverse Document Frequency).

    Tham số:
        columns (list[str]): Danh sách các cột văn bản cần mã hóa.
    """
    for col in columns:
      # Bảo đảm kiểu string, thay nan bằng chuỗi rỗng
      text_data = self.df[col].astype(str).fillna("")

      # Khởi tạo và fit TF-IDF
      tfidf = TfidfVectorizer()
      # tfidf_matrix là một ma trận dạng sparse (số dòng = số bản ghi, số cột = số từ trong vocab)
      # Mỗi phần tử [i, j] là trọng số TF-IDF của từ j trong dòng i
      tfidf_matrix = tfidf.fit_transform(text_data)

      # Tạo tên cột mới từ vocab TF-IDF
      # Lấy danh sách tất cả các token trong vocab mà TF-IDF đã học được, theo một thứ tự cố định (tương ứng với cột của ma trận TF-IDF)
      feature_names = [f"{col}_tfidf_{t}" for t in tfidf.get_feature_names_out()]

      # Chuyển sang df và ghép vào df gốc
      df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=self.df.index)
      self.df = pd.concat([self.df, df_tfidf], axis=1)

      # Lưu vectorizer để dùng lại cho tập test
      self.encoders[(col, "tfidf")] = tfidf

    return self

  # Tạo đặc trưng thời gian
  def create_datetime_features(self, columns: list[str]):
    """
    Trích xuất các đặc trưng thời gian (năm, tháng, ngày, thứ, giờ) từ các cột datetime.

    Tham số:
        columns (list[str]): Danh sách các cột chứa dữ liệu thời gian.
    """
    for col in columns:
      # Chuyển dữ liệu thành datetime, nếu không được trả về NaT
      self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
      self.df[f"{col}_year"] = self.df[col].dt.year
      self.df[f"{col}_month"] = self.df[col].dt.month
      self.df[f"{col}_day"] = self.df[col].dt.day
      self.df[f"{col}_weekday"] = self.df[col].dt.weekday # thứ
      self.df[f"{col}_hour"] = self.df[col].dt.hour
    # Không xoá cột gốc, tránh mất dữ liệu
    print(f"Đã tạo đặc trưng thời gian từ cột {columns}")
    return self

  # Tự động phân loại kiểu dữ liệu
  # Heuristic ở đây dùng để phân biệt các kiểu object dựa trên cách dữ liệu thường xuất hiện
  # Đây không phải một quy tắc thuật toán dùng để phân loại sơ loại về 2 loại categorical và text
  # có thể thay đổi (chỉ mang tính chất gần đúng), sau bước này có thể phân loại chuẩn hơn dựa theo data
  def detect_types(self, heuristic: float = 0.5) -> dict[str, list[str]]:
    """
    Phân loại sơ bộ các cột thành 'numeric', 'datetime', 'categoricals', và 'text'
    dựa trên kiểu dữ liệu và heuristic về số lượng giá trị duy nhất (nunique).

    Tham số:
        heuristic (float): Ngưỡng tỷ lệ (so với tổng số dòng) để phân biệt giữa
                           categorical (thấp) và text (cao). Mặc định 0.5.
    """
    numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
    datetime_cols = self.df.select_dtypes(include="datetime").columns.tolist()

    # others là nơi chứa object (chuỗi), category, dữ liệu text không chuẩn hoá, giá trị hỗn hợp
    others = [c for c in self.df.columns if c not in numeric_cols + datetime_cols]

    # Phân loại categorical và text
    categoricals = []
    text = []

    for col in others:
      # Số lượng giá trị khác nhau
      nunique = self.df[col].nunique(dropna=True)

      # Nếu số giá trị khác nhau < 50% số dòng xem như categorical vì giá trị lặp lại nhiều
      # Nếu số giá trị khác nhau >= 50% số dòng xem như text vì gần như mỗi dòng là 1 chuỗi khác nhau
      if nunique < heuristic * len(self.df):
        categoricals.append(col)
      else:
        text.append(col)

    print("Đã phân loại các cột")
    return {
        "numeric": numeric_cols,
        "datetime": datetime_cols,
        "categoricals": categoricals,
        "text": text,
    }

  # Xoá cột tuỳ chọn
  def drop_columns(self, columns: list[str]):
    """
    Xóa một hoặc nhiều cột khỏi DataFrame.
    """

    # Xoá 1 hoặc nhiều cột theo yêu cầu
    if self.df is None:
      raise ValueError("Chưa có DataFrame!")

    # Nếu truyền 1 cột dạng chuỗi thì chuyển thành list
    # vì ví dụ truyền "age" thì sẽ xử lý thẳng "age" như 1 list, khi chạy sẽ xoá từng chữ
    if isinstance(columns, str):
      columns = [columns]

    # Chỉ có những cột có tồn tại
    cols_exist = [col for col in columns if col in self.df.columns]

    if len(cols_exist) > 0:
      self.df.drop(columns=cols_exist, inplace=True)

    print(f"Đã loại bỏ cột {columns}")
    return self

  # Tách thành 2 DataFrame dựa trên giá trị NA trên 1 cột chỉ định
  def split_by_na(self, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trả về (df_na, df_not_na):
    - df_na: các dòng mà column có giá trị NA
    - df_not_na: các dòng mà columsn không có giá trị NA
    """
    if self.df is None:
      raise ValueError("Chưa có DataFrame!")

    if column not in self.df.columns:
      raise ValueError(f"Cột {column} không tồn tại trong DataFrame!")

    mask = self.df[column].isna()

    df_na = self.df[mask].copy()
    df_not_na = self.df[~mask].copy()

    print(f"Đã tách dữ liệu thành 2 DF na và not na theo cột {column}")
    return df_na, df_not_na

  # Khôi phục lại DataFrame đúng index file gốc
  def restore_after_split(self, df_na: pd.DataFrame, df_not_na: pd.DataFrame) -> pd.DataFrame:
    """
    df_na: phần có NA, giữ nguyên index gốc
    df_not_na: phần không có NA, giữ nguyên index gốc
    """
    # Ghép 2 phần lại
    df_full = pd.concat([df_na, df_not_na])

    # Sort lại theo index gốc
    df_full = df_full.sort_index()

    return df_full

  # Phân loại BMI
  def classify_bmi(self, bmi_col: str = "bmi", category_col: str = "BMI_Category", detailed_obesity: bool = True):
    """
    # detailed_obesity: True thì tách béo phì độ I/II/III, False: gộp chung "Béo phì"
    WHO adult BMI classification:
      < 18.5        : Underweight
      18.5 – 24.9   : Normal weight
      25.0 – 29.9   : Overweight
      30.0 – 34.9   : Obesity class I
      35.0 – 39.9   : Obesity class II
      ≥ 40.0        : Obesity class III
    """
    if self.df is None:
      raise ValueError("Chưa có DataFrame!")

    if bmi_col not in self.df.columns:
      raise ValueError(f"Không thấy cột {bmi_col}!")

    # Ép BMI về số, giá trị lỗi -> NaN
    bmi = pd.to_numeric(self.df[bmi_col], errors="coerce")
    bmi = bmi.replace([np.inf, -np.inf], np.nan)

    if detailed_obesity:
      conditions = [
        bmi < 18.5,
        (bmi >= 18.5) & (bmi < 25),
        (bmi >= 25)   & (bmi < 30),
        (bmi >= 30)   & (bmi < 35),
        (bmi >= 35)   & (bmi < 40),
        bmi >= 40
      ]
      labels = [
        "Underweight",
        "Normal weight",
        "Overweight",
        "Obesity class I",
        "Obesity class II",
        "Obesity class III",
      ]
    else:
      conditions = [
        bmi < 18.5,
        (bmi >= 18.5) & (bmi < 25),
        (bmi >= 25)   & (bmi < 30),
        bmi >= 30,
      ]
      labels = [
        "Underweight",
        "Normal weight",
        "Overweight",
        "Obesity",
      ]

    cat = np.select(conditions, labels, default=np.nan)
    self.df[category_col] = pd.Categorical(cat, categories=labels, ordered=True)

    print(f"Đã phân loại BMI từ cột {bmi_col} vào cột {category_col}")
    return self

  # Ghi dữ liệu ra file
  def save(self, path: str, format: str = None, save_index: bool = False, **kwargs) -> None:
    """
    Lưu DataFrame hiện tại (self.df) ra file.

    Hỗ trợ các định dạng: CSV, Excel (.xlsx, .xls), và JSON. Định dạng sẽ được
    tự động suy luận từ đuôi file nếu tham số 'format' là None.

    Tham số:
        path (str): Đường dẫn bao gồm tên file và đuôi file.
        format (str, optional): Định dạng file mong muốn ('csv', 'excel', 'json'). Mặc định None.
        save_index (bool): True để lưu index (số thứ tự hàng) vào file. Mặc định False.
        **kwargs: Các đối số tùy chọn bổ sung được truyền cho hàm ghi file của pandas (ví dụ: 'sep' cho CSV).
    """
    if self.df is None:
      raise ValueError("Chưa có DataFrame!")

    # Nếu không chọn format thì tự chọn từ đuôi file
    if format is None:
      if path.endswith(".csv"):
        format = "csv"
      elif path.endswith(".xlsx") or path.endswith(".xls"):
        format = "excel"
      elif path.endswith(".json"):
        format = "json"
      else:
        raise ValueError("Không xác định được định dạng từ tên file và cũng không có tham số format")

    try:
      if format == "csv":
        self.df.to_csv(path, index=save_index, **kwargs)
      elif format == "excel":
        self.df.to_excel(path, index=save_index, **kwargs)
      elif format == "json":
        self.df.to_json(path, orient="records", force_ascii=False, indent=2, **kwargs)
      else:
        raise ValueError("Định dạng file chưa hỗ trợ!")

    except Exception as e:
      raise RuntimeError(f"Lỗi khi ghi file: {e}")

  # Lấy DataFrame sau khi preprocessing
  def get_df(self) -> pd.DataFrame:
    # Dùng để đưa vào model mà không làm thay đổi bản trong class nếu chỉnh sửa bên ngoài
    return self.df.copy()