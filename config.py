import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Các miền (domain) hỗ trợ
DOMAINS = {
    'general': 'Tổng quát',
    'tech': 'Công nghệ',
    'fnb': 'Ẩm thực'
}
DEFAULT_DOMAIN = 'general'

# Tham số mô hình
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 3)
SVM_KERNEL = 'linear'
SVM_C = 1.5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Ngưỡng phân loại
NEUTRAL_THRESHOLD = 0.3
MIXED_PART_THRESHOLD = 0.1

# Flask
SECRET_KEY = 'sentiment-analysis-2026'
DEBUG = True
