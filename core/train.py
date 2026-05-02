import os
import sys
import io
import pandas as pd

# Fix encoding cho Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.preprocess import TextPreprocessor


def train_model(csv_path, domain='general'):
    """
    Huấn luyện mô hình phân tích cảm xúc cho một miền (domain) cụ thể.

    Args:
        csv_path: Đường dẫn tới file CSV chứa dữ liệu huấn luyện.
        domain: Tên miền (general, tech, fnb).

    Returns:
        Tuple (model, vectorizer, accuracy).
    """
    print(f"=== Bắt đầu huấn luyện mô hình cho miền: {config.DOMAINS.get(domain, domain)} ===")

    # Đọc dữ liệu
    df = pd.read_csv(csv_path).dropna(subset=['text', 'label'])
    print(f"Số lượng dữ liệu: {len(df)} bình luận")

    # Tiền xử lý
    preprocessor = TextPreprocessor(config.DATA_DIR)
    print("Đang tiền xử lý văn bản...")
    df['clean_text'] = df['text'].apply(preprocessor.clean_text)

    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Vector hóa TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=config.TFIDF_NGRAM_RANGE,
        max_features=config.TFIDF_MAX_FEATURES
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Huấn luyện SVM
    model = SVC(
        kernel=config.SVM_KERNEL,
        C=config.SVM_C,
        class_weight='balanced'
    )
    model.fit(X_train_vec, y_train)

    # Đánh giá trên tập test
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nĐộ chính xác trên tập test: {acc * 100:.2f}%")
    print("\nBáo cáo chi tiết:")
    print(classification_report(y_test, y_pred, target_names=['Tiêu cực', 'Tích cực']))

    # Huấn luyện lại trên toàn bộ dữ liệu để có mô hình tốt nhất
    X_final_vec = vectorizer.fit_transform(df['clean_text'])
    model.fit(X_final_vec, df['label'])

    # Lưu mô hình
    model_dir = os.path.join(config.MODELS_DIR, domain)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'sentiment_model.pkl')
    vec_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    print(f"\nĐã lưu mô hình tại: {model_path}")
    print(f"Đã lưu vectorizer tại: {vec_path}")
    print(f"=== Hoàn tất huấn luyện miền: {config.DOMAINS.get(domain, domain)} ===\n")

    return model, vectorizer, acc


if __name__ == '__main__':
    # Huấn luyện mô hình general từ data.csv gốc
    csv_path = os.path.join(config.DATA_DIR, 'data.csv')
    if os.path.exists(csv_path):
        train_model(csv_path, domain='general')
    else:
        print(f"Không tìm thấy file dữ liệu: {csv_path}")
