# 📘 GIẢI THÍCH CHI TIẾT DỰ ÁN PHÂN TÍCH CẢM XÚC BÌNH LUẬN

## 1. GIỚI THIỆU DỰ ÁN

Hệ thống **Phân Tích Cảm Xúc Bình Luận** (Sentiment Analysis) là ứng dụng web xử lý ngôn ngữ tự nhiên tiếng Việt, dự đoán cảm xúc của bình luận người dùng.

### Mục tiêu:
- Phân loại bình luận thành: **Tích cực** (1), **Tiêu cực** (0)
- Hỗ trợ các miền khác nhau: Công nghệ, Ẩm thực, Tổng quát
- Cung cấp giải thích chi tiết cho mỗi dự đoán
- Xử lý các trường hợp đặc biệt: Mỉa mai, Hỗn hợp cảm xúc, Không rõ

---

## 2. KIẾN TRÚC HỆ THỐNG

### 2.1 Sơ đồ Tổng Quan

```
INPUT (Bình luận)
    ↓
[API /api/predict]
    ↓
[TextPreprocessor - Xử lý văn bản]
    ↓
[Kiểm tra bình luận không rõ (is_unclear)]
    ↓
[Phát hiện mỉa mai (detect_sarcasm)]
    ↓
[Phát hiện đối lập (contrast analysis)]
    ↓
[Model SVM + TF-IDF]
    ↓
[Nhãn + Giải thích]
    ↓
OUTPUT (Kết quả JSON)
```

### 2.2 Các Thành Phần Chính

#### **A. Flask Web Server** (app.py)
- Xây dựng API REST cho việc dự đoán
- Quản lý phiên (session) lịch sử phân tích
- Xử lý upload file CSV/Excel cho phân tích hàng loạt
- Định tuyến các endpoint:
  - `GET /` - Giao diện chính
  - `POST /api/predict` - Dự đoán từ bình luận đơn lẻ
  - `POST /api/import_predict` - Phân tích file tải lên
  - `GET /api/history` - Lấy lịch sử
  - `POST /api/history/clear` - Xóa lịch sử

#### **B. Text Preprocessor** (core/preprocess.py)
Xử lý và chuẩn hóa văn bản Tiếng Việt:
- **Chuẩn hóa ký tự lặp**: "đẹpppppp" → "đẹp"
- **Xóa URL và ký tự đặc biệt**
- **Thay thế teencode**: "ơi" → "ơi", "tks" → "cảm ơn"
- **Xử lý phủ định**: "không đẹp" → "không_đẹp"
- **Tokenize Tiếng Việt**: Sử dụng PyVi ViTokenizer
- **Kiểm tra bình luận không rõ**:
  - Quá ngắn (< 2 ký tự)
  - Chỉ là spam/link
  - Chỉ chữ số
  - Toàn ký tự vô nghĩa

#### **C. Sentiment Predictor** (core/predictor.py)
Module chính dự đoán cảm xúc:

**Các nhãn (Labels):**
- **Tích cực** 😊: Bình luận dương tính (label=1)
- **Tiêu cực** 😞: Bình luận âm tính (label=0)
- **Trung lập** 😐: Không có rõ cảm xúc
- **Hỗn hợp** 🤔: Vừa tích cực vừa tiêu cực
- **Không rõ** ❓: Bình luận không thể phân tích

**Quy trình dự đoán:**

1. **Kiểm tra bình luận không rõ** (is_unclear)
2. **Phát hiện miền** (detect_domain): Công nghệ? Ẩm thực?
3. **Phân tích đối lập** (_check_contrast): Tìm từ "nhưng", "song", "tuy nhiên"
4. **Dự đoán bằng SVM**: Sử dụng model đã huấn luyện
5. **Kiểm tra sarcasm** (detect_sarcasm): Phát hiện mỉa mai
6. **Tính confidence**: Nếu < 0.8, kiểm tra từ cảm xúc
7. **Kiểm tra neutral**: Nếu confidence < 0.3, đánh giá là trung lập
8. **Xây dựng giải thích** (_build_explanation): Hiển thị từ quan trọng

#### **D. Model Training** (core/train.py)
Huấn luyện mô hình Machine Learning:

**Dữ liệu:**
- **data.csv**: 11,386 bình luận (training)
- **dev.csv**: Tập phát triển
- **test.csv**: Tập kiểm tra

**Preprocessing:**
- Làm sạch văn bản
- Cân bằng dữ liệu (Oversampling)
- Phân tách 5-Fold Cross-Validation

**Vectorization (TF-IDF):**
- **Word features**: N-gram (1-2), max 5000 features
- **Char features**: Character n-gram (3-5), max 2500 features
- **FeatureUnion**: Kết hợp cả hai

**Mô hình:**
- **LinearSVC**: Support Vector Machine tuyến tính
- **class_weight='balanced'**: Xử lý dữ liệu không cân bằng

**Kết quả:**
- Lưu model: `models/general/sentiment_model.pkl`
- Lưu vectorizer: `models/general/tfidf_vectorizer.pkl`

#### **E. Cấu Hình** (config.py)
- Đường dẫn dữ liệu, mô hình
- Các miền hỗ trợ (domains)
- Siêu tham số TF-IDF, SVM
- Các ngưỡng (threshold) cho neutral/mixed

---

## 3. QUY TRÌNH XỬ LÝ CHI TIẾT

### 3.1 Phân Tích Bình Luận Đơn Lẻ

```
Bình luận: "Sản phẩm này rất đẹp!"
   ↓
[Bước 1] is_unclear()
   - Độ dài >= 2 ✓
   - Không phải link ✓
   - Không phải toàn số ✓
   - Không phải ký tự vô nghĩa ✓
   → Bình luận hợp lệ
   ↓
[Bước 2] detect_domain()
   - Tìm từ khóa trong domain_keywords.json
   - Nếu không tìm → 'general'
   ↓
[Bước 3] _check_contrast()
   - Tìm từ "nhưng", "song", "tuy nhiên"...
   - Nếu có → phân tích phần trước/sau
   - Nếu không → tiếp tục
   ↓
[Bước 4] SVM Prediction
   - Làm sạch: "sản phẩm nàyrất đẹp"
   - Vectorize bằng TF-IDF
   - Dự đoán: prediction = 1 (Tích cực)
   - Confidence: 0.85
   ↓
[Bước 5] detect_sarcasm()
   - Kiểm tra pattern trong sarcasm_patterns.json
   - Nếu phát hiện → đảo nhãn
   ↓
[Bước 6] Kiểm tra Threshold
   - confidence (0.85) > NEUTRAL_THRESHOLD (0.3) ✓
   - confidence (0.85) < 0.8? Kiểm tra từ cảm xúc
   - Có "đẹp" → Tích cực
   ↓
[Bước 7] _build_explanation()
   - Tính contribution từ mỗi từ
   - Top từ tích cực: "đẹp" (0.45), "rất" (0.32)
   - Top từ tiêu cực: (không có)
   ↓
OUTPUT:
{
  'label': 'Tích cực',
  'icon': '😊',
  'explanation': 'Bình luận được đánh giá là tích cực.\n📗 Từ/cụm từ mang tính tích cực: "đẹp", "rất"',
  'domain_used': 'Tổng quát'
}
```

### 3.2 Phân Tích Hỗn Hợp

```
Bình luận: "Thiết kế đẹp nhưng tốc độ chậm"
   ↓
[_check_contrast] Tìm "nhưng"
   → parts = ["Thiết kế đẹp", "tốc độ chậm"]
   ↓
   part1: "Thiết kế đẹp"
     - Có từ "đẹp" → Tích cực
   part2: "tốc độ chậm"
     - Có từ "chậm" → Tiêu cực
   ↓
OUTPUT:
{
  'label': 'Hỗn hợp',
  'icon': '🤔',
  'explanation': 'Bình luận chứa cả cảm xúc tích cực và tiêu cực.\n- Phần tích cực: "Thiết kế đẹp"\n- Phần tiêu cực: "tốc độ chậm"'
}
```

---

## 4. DỮ LIỆU VÀ HỌC MÁY

### 4.1 Tập Dữ Liệu

**Cấu trúc:**
```csv
label,text
1,Android nhập liệu bằng lời nói rất chuẩn...
0,Sau một thời gian sử dụng AW mình thấy...
```

**Thống kê:**
- Tổng cộng: ~11,386 bình luận
- Label 0 (Tiêu cực): ~50%
- Label 1 (Tích cực): ~50%

### 4.2 Xử Lý Dữ Liệu

**Lưu Ý Đặc Biệt Tiếng Việt:**
- Không có chia từ (word segmentation) tự động
- Sử dụng **PyVi** để tokenize
- Xử lý teencode: "tks", "ơi", "đúng ko"
- Xử lý phủ định: "không tốt" vs "tốt"

**Cân Bằng Dữ Liệu:**
- Nếu có nhiều class 1, ít class 0
- **Oversampling**: Nhân rộng class thiểu số
- **StratifiedKFold**: Đảm bảo tỷ lệ nhãn trong mỗi fold

### 4.3 Tính Năng (Features)

**TF-IDF (Term Frequency - Inverse Document Frequency):**
- Đo lường tầm quan trọng của từ trong tài liệu

**Word Features:**
- Bigram: "sản phẩm tốt", "rất đẹp"
- Max 5000 features
- Min_df=2: Từ phải xuất hiện ít nhất 2 lần
- Max_df=0.95: Từ không xuất hiện trong >95% tài liệu

**Char Features:**
- 3-5 ký tự liên tiếp
- Giúp bắt lỗi chính tả
- Max 2500 features

**FeatureUnion:**
- Kết hợp cả Word + Char features
- Tổng: ~7500 features

### 4.4 Mô Hình SVM

**Support Vector Machine (SVM):**
- Tìm hyperplane tách biệt tốt nhất
- Linear kernel: Dễ hiểu, nhanh
- C=1.5: Hyperparameter regularization
- class_weight='balanced': Xử lý dữ liệu không cân bằng

**Ưu điểm:**
- Hiệu quả với văn bản
- Dễ sử dụng với TF-IDF
- Có thể lấy feature importance (coef_)

### 4.5 Đánh Giá Mô Hình

**Cross-Validation:**
- 5-Fold Stratified K-Fold
- Tính trung bình độ chính xác

**Metrics:**
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Điều hòa của Precision & Recall

**Test Set:**
- Đánh giá trên test.csv
- Độc lập với huấn luyện

---

## 5. TỪ KHÓA VÀ PATTERNS

### 5.1 Từ Khóa Tích Cực
```
tốt, đẹp, ngon, hay, thích, yêu, tuyệt, xuất sắc,
xinh, bền, mượt, chất, chuẩn, ổn, ok, oke, rẻ,
kỹ, cẩn thận, nhiệt tình, lịch sự, vui vẻ, thân thiện
```

### 5.2 Từ Khóa Tiêu Cực
```
xấu, tệ, dở, chậm, yếu, hỏng, lỗi, lag, giật,
kém, đắt, nóng, ồn, mờ, nặng, khó, thái độ, hách,
ghẻ lạnh, khó chịu, bẩn, tồi, toxic, scam, lừa, bịp
```

### 5.3 Mẫu Mỉa Mai
```
"...thì như chó"
"...quá tuyệt vời"
"...chắc đúng rồi"
```

---

## 6. GIAO DIỆN NGƯỜI DÙNG

### 6.1 Frontend (HTML/CSS/JS)
- **index.html**: Giao diện chính
- **style.css**: CSS styling
- **app.js**: JavaScript xử lý AJAX

### 6.2 Các Chức Năng Chính
1. **Nhập bình luận** → Chọn miền → Dự đoán
2. **Xem lịch sử** → Lưu tất cả phân tích
3. **Tải file lên** → Phân tích hàng loạt
4. **Tải kết quả** → Xuất Excel

### 6.3 Hiển Thị Kết Quả
```
📊 Kết Quả Phân Tích:

Bình luận: "Sản phẩm rất tốt!"

😊 Nhãn: TÍCH CỰC
📍 Miền: Tổng quát
⭐ Giải thích:
   Bình luận được đánh giá là tích cực.
   📗 Từ/cụm từ mang tính tích cực: "tốt", "rất"
```

---

## 7. CÁC LỖI VÀ CỬ XỬ CẠN BIỆT

### 7.1 Bình Luận Không Rõ
- Quá ngắn: "ok", "ơi"
- Chỉ ký tự đặc biệt: "!!!"
- Chỉ link: "http://..."
- Toàn chữ số: "12345"
- Ký tự vô nghĩa: "asdfgh"

### 7.2 Phát Hiện Lỗi Tiếng Việt
- "đppp" → "đp" (normalize chars)
- "t1ế̀m" → "tiểm" (clean chars)
- Sử dụng regex để xóa ký tự không Tiếng Việt

### 7.3 Hiển Thị Lỗi
**Vấn đề:** Hiển thị từ khóa không đúng format
**Nguyên nhân:** PyVi tokenizer xử lý khác với mong đợi
**Giải pháp:** Kiểm tra output trực tiếp từ model

---

## 8. HƯỚNG PHÁT TRIỂN TƯƠNG LAI

1. **Thêm mô hình Deep Learning**:
   - LSTM, Transformer, BERT

2. **Phát hiện cảm xúc chi tiết**:
   - Không chỉ tích cực/tiêu cực
   - Mà "vui", "buồn", "tức giận", "sợ"

3. **Đa ngôn ngữ**:
   - Hỗ trợ tiếng Anh, Trung Quốc

4. **API mở rộng**:
   - Batch processing
   - Webhook callback

5. **Dashboard analytics**:
   - Thống kê cảm xúc theo thời gian
   - Phân tích trend

---

## 9. HƯỚNG DẪN CHẠY DỰ ÁN

### Yêu cầu:
- Python 3.8+
- Các thư viện: flask, scikit-learn, pandas, pyvi, joblib

### Cài đặt:
```bash
pip install -r requirements.txt
```

### Huấn luyện mô hình:
```bash
python core/train.py
```

### Chạy web app:
```bash
python app.py
```
Truy cập: http://127.0.0.1:5000

### Dự đoán batch:
```bash
python app.py
# Upload file CSV/Excel từ giao diện
```

---

## 10. KẾT LUẬN

Hệ thống phân tích cảm xúc này kết hợp:
- **NLP tiền xử lý** mạnh mẽ cho Tiếng Việt
- **Machine Learning** với SVM + TF-IDF
- **Web interface** thân thiện với người dùng
- **Xử lý trường hợp đặc biệt** (mỉa mai, đối lập, không rõ)
- **Giải thích chi tiết** từng dự đoán

Phù hợp cho các ứng dụng:
- Phân tích review sản phẩm
- Giám sát phương tiện truyền thông xã hội
- Đánh giá độ hài lòng khách hàng
- Nghiên cứu và học tập

---

*Tài liệu cập nhật: 2026*

