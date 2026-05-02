# BÁO CÁO ĐỒ ÁN: HỆ THỐNG AI PHÂN TÍCH CẢM XÚC BÌNH LUẬN TIẾNG VIỆT

---

## 1. CÁC KIẾN THỨC CƠ BẢN SỬ DỤNG TRONG DỰ ÁN (Dành cho sinh viên)

Để hiểu và thuyết trình được dự án này, bạn cần nắm qua các khái niệm cơ bản sau:

*   **Machine Learning (Học máy) - Thuật toán SVM (Support Vector Machine):** Đây là thuật toán AI cốt lõi được sử dụng để phân loại bình luận thành "Tích cực" hoặc "Tiêu cực". SVM hoạt động bằng cách tìm ra một "đường ranh giới" tốt nhất để chia tách các nhóm dữ liệu dựa trên các đặc trưng của từ ngữ.
*   **NLP (Xử lý ngôn ngữ tự nhiên) - TF-IDF:** Máy tính không hiểu được chữ viết, nó chỉ hiểu số. TF-IDF là công nghệ giúp biến đổi các câu văn thành các vector số học. Nó đánh giá tầm quan trọng của một từ trong câu: từ nào xuất hiện nhiều trong một câu nhưng hiếm gặp ở các câu khác (ví dụ: "xuất sắc", "tệ hại") sẽ được đánh trọng số cao.
*   **Phương pháp Hybrid (Lai - Kết hợp AI và Rule-based):** Không chỉ dựa 100% vào AI (đôi khi AI sẽ đoán sai nếu thiếu dữ liệu). Dự án này kết hợp AI với các **Rule-based (Hệ luật)** do con người viết ra (ví dụ: kiểm tra chữ "nhưng", kiểm tra từ văng tục, bắt mẫu câu mỉa mai). Nhờ vậy, hệ thống cực kỳ linh hoạt và chính xác.
*   **Flask Framework:** Là một thư viện của Python dùng để xây dựng trang web (Backend) giao tiếp với giao diện người dùng (Frontend).

## 2. NGUỒN DỮ LIỆU HUẤN LUYỆN (DATASET)

*   **Dữ liệu đến từ đâu?** Dữ liệu huấn luyện nằm trong file `data/data.csv`. Tập dữ liệu này được xây dựng bằng cách thu thập các bình luận thực tế của người dùng trên các sàn thương mại điện tử (Shopee, Tiki) và mạng xã hội (Facebook, Tiktok).
*   **Chỉnh sửa thủ công:** Sinh viên đã chủ động bổ sung thêm các bình luận cố ý viết không dấu, viết teencode, và các bình luận mang tính chất mỉa mai vào tập dữ liệu để AI học được cách người Việt thực sự giao tiếp trên mạng.

---

## 3. QUY TẮC HOẠT ĐỘNG CHUNG CỦA HỆ THỐNG

Khi một người dùng nhập bình luận, hệ thống sẽ chạy qua một "đường ống" (Pipeline) gồm các bước sau:

1.  **Bước 1 - Lọc rác (Unclear):** Kiểm tra xem bình luận có quá ngắn, chỉ chứa số, chứa link, hay chỉ có 1 từ chửi bậy duy nhất không. Nếu có → Trả về **Không rõ**.
2.  **Bước 2 - Tiền xử lý (Preprocess):** Rút gọn ký tự kéo dài (đẹpppp -> đẹp), dịch teencode (vl -> rất), dịch tiếng Anh, nối từ phủ định (không tốt -> không_tốt), và tách từ tiếng Việt.
3.  **Bước 3 - Xét vế đối lập (Contrast):** Tìm xem câu có chữ "nhưng", "tuy nhiên"... không. Nếu có, cắt câu làm 2 vế. Nếu vế 1 khen, vế 2 chê → **Hỗn hợp**. Nếu cả 2 vế cùng chê → **Tiêu cực**.
4.  **Bước 4 - AI Dự đoán (Machine Learning):** Đưa câu vào mô hình SVM để tính điểm số tự tin (Confidence score).
5.  **Bước 5 - Xét Trung lập:** Nếu điểm tự tin của AI quá thấp (không rõ khen chê), HOẶC không có các từ khóa cảm xúc mạnh → Trả về **Trung lập**.
6.  **Bước 6 - Bắt mỉa mai (Sarcasm):** Kiểm tra xem cấu trúc câu có giống mỉa mai không (VD: Khen + thời gian ngắn + hỏng). Nếu có mỉa mai → Đảo ngược kết quả (Tích cực thành Tiêu cực).

---

## 4. CẤU TRÚC THƯ MỤC DỰ ÁN

*   **`core/`**: Chứa não bộ của AI (xử lý logic, tiền xử lý, huấn luyện).
*   **`data/`**: Chứa dữ liệu (file CSV để AI học) và các từ điển JSON (teencode, phủ định, mỉa mai).
*   **`models/`**: Nơi lưu trữ mô hình AI (`.pkl`) sau khi đã học xong.
*   **`static/`**: Chứa giao diện (CSS để làm đẹp, JS để tạo tương tác web).
*   **`templates/`**: Chứa file `index.html` (khung xương của trang web).
*   **`app.py`**: File chạy Server Web.
*   **`config.py`**: File cấu hình các thông số chung.

---

## 5. CHI TIẾT MÃ NGUỒN TỪNG TỆP TIN

### 5.1. Thư mục `core/` (Lõi xử lý logic)

#### File: `core/preprocess.py` (Bộ tiền xử lý)
**Nhiệm vụ:** Chuẩn bị và làm sạch dữ liệu trước khi đưa cho AI. AI giống như đứa trẻ, bạn phải đút thức ăn đã nghiền nát thì nó mới tiêu hóa được.
*   **`def is_unclear(self, text)`**: Chặn các câu vô nghĩa. Nó check nếu độ dài < 2, chứa link URL, toàn số, hoặc là chỉ có 1 từ văng tục (vd: "cc", "vl") thì sẽ báo là "Không rõ".
*   **`def normalize_repeated_chars(self, text)`**: Rút gọn ký tự lặp. Chuyển "khônggggg" thành "không" bằng biểu thức chính quy Regex.
*   **`def process_negation(self, text)`**: Xử lý từ phủ định. Máy tính hay nhầm chữ "không tốt" thành 2 từ riêng "không" và "tốt" (từ "tốt" là khen). Hàm này nối chúng lại thành "không_tốt" để AI hiểu đây là chê.
*   **`def clean_text(self, text)`**: Hàm làm sạch tổng hợp. Nó gọi các hàm trên, đồng thời dò trong từ điển teencode.json để dịch "dc" -> "được", dịch từ văng tục thành "rất" (từ nhấn mạnh mức độ).
*   **`def detect_sarcasm(self, text)`**: Tìm mỉa mai. Dùng biểu thức chính quy quét xem câu có khớp với các mẫu mỉa mai lưu trong `sarcasm_patterns.json` không.

#### File: `core/predictor.py` (Bộ ra quyết định)
**Nhiệm vụ:** Kết hợp các quy tắc và AI để đưa ra kết luận cuối cùng (1 trong 5 nhãn).
*   **`def predict(self, text, domain='auto')`**: Luồng chạy chính. Nó gọi `is_unclear` trước, nếu ổn thì gọi `_check_contrast`. Nếu không có từ đối lập, nó lấy câu đưa vào model AI. Dựa vào điểm số trả ra từ AI, nó check tiếp xem có phải "Trung lập" hay không. Cuối cùng gọi `detect_sarcasm` để xem có cần đảo ngược kết quả không.
*   **`def _check_contrast(self, text, model, vectorizer)`**: Xử lý các câu có từ "nhưng", "tuy nhiên". Nó cắt câu ra làm các vế nhỏ. Sau đó phân tích độc lập từng vế. Nếu vế 1 tích cực, vế 2 tiêu cực -> trả về "Hỗn hợp". Nếu cả 2 vế tiêu cực -> trả về "Tiêu cực".

#### File: `core/train.py` (File huấn luyện)
**Nhiệm vụ:** Đọc file `data.csv`, dạy AI học và lưu kết quả lại.
*   **`def train_model(csv_path, domain)`**: Đọc file CSV, tách ra 80% để học, 20% để làm bài kiểm tra. Dùng `TfidfVectorizer` để ép chữ thành số, rồi nhét vào thuật toán `LinearSVC` để học. Cuối cùng lưu mô hình ra file `.pkl` để dùng sau này (không phải chạy web lần nào cũng học lại).

### 5.2. Thư mục `data/` (Dữ liệu)

*   **`data.csv`**: File chứa hàng ngàn bình luận có gắn nhãn sẵn (Tích cực / Tiêu cực) để làm sách giáo khoa cho AI học.
*   **`teencode.json`**: Từ điển sinh viên tự tạo. Rất đặc biệt vì nó chứa cả các từ tiếng Việt/tiếng Anh viết tắt (ko, dc), slang Gen Z (slay, khét, đỉnh), và quy định **từ văng tục đứng kèm sẽ đóng vai trò là từ phóng đại** (ví dụ "vl", "cc", "fuck" được map thành nghĩa "rất"). Nhờ vậy "cc đẹp quá" sẽ hiểu là "rất đẹp quá" -> Tích cực.
*   **`sarcasm_patterns.json`**: Chứa các cấu trúc Regex bắt lỗi mỉa mai (ví dụ: mẫu câu Khen + Thời gian + Hỏng/Lỗi).
*   **`domain_keywords.json`**: Từ khóa để tự động nhận biết người dùng đang review đồ công nghệ hay đồ ăn uống.

### 5.3. Thư mục gốc & File cấu hình

#### File: `app.py` (Khởi chạy Web Backend)
**Nhiệm vụ:** Giao tiếp giữa người dùng (trình duyệt) và bộ não AI.
*   **`def get_predictor()`**: Khởi tạo AI. Áp dụng kỹ thuật "Lazy loading", chỉ tải mô hình 1 lần duy nhất vào RAM để web chạy nhanh, không bị đơ.
*   **`def api_predict()`**: Khi user bấm nút trên web, hàm này nhận text, gọi AI xử lý, lấy kết quả, tạo ID thời gian lưu vào lịch sử truy cập (Session) rồi trả về cho web hiển thị.

#### File: `config.py` (Cấu hình)
**Nhiệm vụ:** Nơi khai báo các biến dùng chung. Nếu sinh viên muốn chỉnh AI nhạy hơn hay khó tính hơn thì sửa ở đây.
*   **`NEUTRAL_THRESHOLD`**: Ngưỡng trung lập. AI tự tin dưới điểm này sẽ coi là câu không có cảm xúc (Trung lập).

### 5.4. Thư mục giao diện (Frontend)

*   **`templates/index.html`**: Code bố cục trang web, các nút bấm, ô nhập liệu. Không dùng framework nặng nề, chỉ dùng HTML thuần túy cho sinh viên dễ hiểu.
*   **`static/css/style.css`**: Làm đẹp web. Dùng các biến màu sắc (tone Trắng - Hồng) trẻ trung. Code được tổ chức rõ ràng theo layout, input, result.
*   **`static/js/app.js`**: Code hiệu ứng web. Lấy dữ liệu người dùng gõ, gọi API gửi lên Python. Nhận kết quả từ Python về và thay đổi DOM (nhúng icon, đổi màu chữ, thêm lịch sử ở thanh bên trái).

---

## 6. ĐƯỜNG ĐI CỦA MỘT BÌNH LUẬN (LIFECYCLE)

Để dễ hình dung nhất cách cả hệ thống phối hợp với nhau, hãy tưởng tượng một người dùng vào web và gõ câu: *"cc xấu vl, không bao giờ mua nữa"*. Dưới đây là hành trình của câu nói đó:

1.  **Tại Trình duyệt (Frontend - HTML/JS):**
    *   Người dùng gõ câu trên vào ô input và bấm nút "Phân tích".
    *   File `app.js` nhảy vào can thiệp. Nó đóng gói câu nói đó thành định dạng JSON và bắn (gửi request) theo đường truyền mạng lên máy chủ Python thông qua cổng `/api/predict`.
    *   Nút "Phân tích" lúc này quay quay ⏳ chờ đợi.

2.  **Tại Cổng tiếp tân (Backend - app.py):**
    *   File `app.py` đang lắng nghe tại cổng `/api/predict`. Nó nhận gói hàng JSON.
    *   Nó kiểm tra xem gói hàng có bị rỗng không. Khi thấy có chữ, nó gọi "bộ não" AI bằng lệnh: `predictor.predict("cc xấu vl...")`.

3.  **Tại Bộ tiền xử lý (core/preprocess.py):**
    *   Câu nói bị lột trần để làm sạch.
    *   Nó thấy chữ `"cc"`, tra từ điển `teencode.json` và nhận ra đây là từ văng tục đóng vai trò nhấn mạnh, nó biến thành `"rất"`.
    *   Nó thấy chữ `"vl"`, tra từ điển, biến thành `"rất"`.
    *   Nó thấy chữ `"không bao giờ"`, phát hiện từ phủ định, nó nối chữ lại thành `"không_bao_giờ"`.
    *   Câu bây giờ biến thành: *"rất xấu rất, không_bao_giờ mua nữa"*.

4.  **Tại Bộ phân tích quy tắc (core/predictor.py):**
    *   Hệ thống kiểm tra xem có chữ "nhưng", "tuy nhiên" không. Câu này không có.
    *   Hệ thống kiểm tra xem có mỉa mai không. Câu này chê thẳng mặt, không mỉa mai.

5.  **Tại Bộ não AI (Machine Learning - SVM):**
    *   Câu *"rất xấu rất, không_bao_giờ mua nữa"* được ép thành các con số ma trận (bằng `tfidf_vectorizer`).
    *   Mô hình AI nhận thấy các trọng số siêu mạnh: "xấu", "rất", "không_bao_giờ". Nó chấm điểm rớt đài xuống mức âm sâu.
    *   Nó tự tin 99% kết luận đây là **Tiêu cực**.

6.  **Trả kết quả về cho Trình duyệt:**
    *   `predictor.py` gói kết quả lại gồm: Nhãn (Tiêu cực), Icon (😞), và Giải thích (Từ mang tính tiêu cực: "xấu", "không_bao_giờ").
    *   `app.py` nhận kết quả này, lưu ngay vào thẻ Nhớ (Session) để làm lịch sử truy cập, rồi đóng gói thành JSON trả về cho trình duyệt.
    *   File `app.js` nhận được JSON, tắt cái vòng quay quay ⏳ đi. Nó bóc gói hàng ra, đổi màu chữ trên màn hình thành màu đỏ, in icon 😞 và hiện dòng chữ "Tiêu cực" to đùng lên màn hình cho người dùng xem.

Kết thúc quá trình! Toàn bộ diễn ra trong chưa tới 0.1 giây.
