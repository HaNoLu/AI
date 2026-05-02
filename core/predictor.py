import os
import re
import joblib
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.preprocess import TextPreprocessor


class SentimentPredictor:
    # Class du doan cam xuc 5 nhan

    POSITIVE = "Tích cực"
    NEGATIVE = "Tiêu cực"
    NEUTRAL = "Trung lập"
    MIXED = "Hỗn hợp"
    UNCLEAR = "Không rõ"

    LABEL_ICONS = {
        "Tích cực": "😊",
        "Tiêu cực": "😞",
        "Trung lập": "😐",
        "Hỗn hợp": "🤔",
        "Không rõ": "❓"
    }

    def __init__(self):
        self.preprocessor = TextPreprocessor(config.DATA_DIR)
        self.models = {}  # domain -> (model, vectorizer)
        self._load_models()

    def _load_models(self):
        # load model tu thu muc
        for domain in config.DOMAINS:
            model_path = os.path.join(config.MODELS_DIR, domain, 'sentiment_model.pkl')
            vec_path = os.path.join(config.MODELS_DIR, domain, 'tfidf_vectorizer.pkl')
            if os.path.exists(model_path) and os.path.exists(vec_path):
                model = joblib.load(model_path)
                vectorizer = joblib.load(vec_path)
                self.models[domain] = (model, vectorizer)

    def reload_models(self):
        """Tải lại tất cả mô hình (dùng sau khi train lại)."""
        self.models = {}
        self._load_models()

    def predict(self, text, domain='auto'):
        # ham du doan chinh
        # check Không rõ
        is_unclear, reason = self.preprocessor.is_unclear(text)
        if is_unclear:
            return self._result(self.UNCLEAR, reason, 'none')

        # tu nhan dien mien
        if domain == 'auto':
            domain = self.preprocessor.detect_domain(text)

        # fallback ve general
        if domain not in self.models:
            domain = 'general'
        if domain not in self.models:
            return self._result(self.UNCLEAR,
                                "Chưa có mô hình AI nào được huấn luyện. Vui lòng huấn luyện mô hình trước.",
                                'none')

        model, vectorizer = self.models[domain]

        # check cac tu doi lap nhu: nhung, tuy nhien...
        contrast_result = self._check_contrast(text, model, vectorizer)
        if contrast_result:
            result_type, explanation = contrast_result
            if result_type == "mixed":
                return self._result(self.MIXED, explanation, domain)
            elif result_type == "negative":
                return self._result(self.NEGATIVE, explanation, domain)
            elif result_type == "positive":
                return self._result(self.POSITIVE, explanation, domain)

        # predict bang ml
        cleaned = self.preprocessor.clean_text(text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        confidence = abs(model.decision_function(vec)[0])

        # check trung lap
        # do tin cay thap -> trung lap
        if confidence < config.NEUTRAL_THRESHOLD:
            explanation = self._build_explanation(text, cleaned, vectorizer, model, "trung lập")
            return self._result(self.NEUTRAL, explanation, domain)

        # 6b: Nếu confidence vừa phải VÀ không chứa từ cảm xúc mạnh → trung lập
        if confidence < 0.8:
            sentiment_keywords = {
                'tốt', 'tot', 'đẹp', 'dep', 'ngon', 'hay', 'thích', 'thich',
                'yêu', 'yeu', 'tuyệt', 'tuyet', 'xuất sắc', 'xuat sac',
                'xấu', 'xau', 'tệ', 'te', 'dở', 'do', 'chậm', 'cham',
                'hỏng', 'hong', 'lỗi', 'loi', 'ghét', 'ghet', 'kém', 'kem',
                'giỏi', 'gioi', 'nhanh', 'chất', 'chat', 'mượt', 'muot',
                'lag', 'giật', 'giat', 'nóng', 'nong', 'rẻ', 're', 'đắt', 'dat',
                'bền', 'ben', 'xinh', 'mạnh', 'manh', 'yếu', 'thái độ', 'thai do',
                'đỉnh', 'dinh', 'xịn', 'xin', 'gắt', 'gat', 'phê', 'phe',
                'slay', 'peak', 'goat', 'khét', 'khet', 'cháy', 'chay',
                'shit', 'terrible', 'horrible', 'awful', 'trash', 'garbage',
                'perfect', 'amazing', 'awesome', 'beautiful', 'love', 'hate',
                'suck', 'sucks', 'crap', 'fuck', 'damn',
                'ngu', 'chó', 'cc', 'vcl', 'vl', 'toxic', 'cringe',
                'scam', 'lừa', 'lua', 'bịp', 'bip'
            }
            text_lower = text.lower()
            has_sentiment = any(kw in text_lower for kw in sentiment_keywords)
            if not has_sentiment:
                explanation = "Bình luận không chứa từ ngữ cảm xúc rõ ràng, được đánh giá là trung lập."
                return self._result(self.NEUTRAL, explanation, domain)

        # ket luan tich cuc hay tieu cuc
        if prediction == 1:
            label = self.POSITIVE
            sentiment_word = "tích cực"
        else:
            label = self.NEGATIVE
            sentiment_word = "tiêu cực"

        # check sarcasm
        is_sarcasm, sarcasm_explanation = self.preprocessor.detect_sarcasm(text)
        if is_sarcasm:
            # Đảo ngược kết quả
            if label == self.POSITIVE:
                label = self.NEGATIVE
                sentiment_word = "tiêu cực"
            else:
                label = self.POSITIVE
                sentiment_word = "tích cực"

        explanation = self._build_explanation(text, cleaned, vectorizer, model, sentiment_word)
        if is_sarcasm:
            explanation = f"Phat hien mia mai: {sarcasm_explanation}\n{explanation}"

        return self._result(label, explanation, domain)

    def _check_contrast(self, text, model, vectorizer):
        # kiem tra cau co tu noi doi lap va phan tich tung ve
        # Các từ nối đối lập (có dấu + không dấu)
        splitters = (
            r'\bnhưng\b|\bnhung\b|\bnhưng mà\b|\bnhung ma\b'
            r'|\btuy nhiên\b|\btuy nhien\b|\bthế nhưng\b|\bthe nhung\b'
            r'|\bsong\b|\bnhưng lại\b|\bnhung lai\b'
            r'|\btuy vậy\b|\btuy vay\b|\bdù vậy\b|\bdu vay\b'
        )

        if not re.search(splitters, text, re.IGNORECASE):
            return None

        parts = re.split(splitters, text, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if len(p.strip()) > 2]

        if len(parts) < 2:
            return None

        # Từ khóa tích cực / tiêu cực
        pos_keywords = {'tốt', 'tot', 'đẹp', 'dep', 'nhanh', 'giỏi', 'gioi', 'hay',
                        'ngon', 'thích', 'thich', 'yêu', 'yeu', 'tuyệt', 'tuyet',
                        'xinh', 'bền', 'ben', 'mượt', 'muot', 'chất', 'chat',
                        'chuẩn', 'chuan', 'ổn', 'on', 'ok', 'oke', 'rẻ', 're',
                        'kỹ', 'ky', 'cẩn thận', 'can than', 'nhiệt tình', 'nhiet tinh',
                        'lịch sự', 'lich su', 'vui vẻ', 'vui ve', 'thân thiện', 'than thien'}
        neg_keywords = {'xấu', 'xau', 'tệ', 'te', 'dở', 'do', 'chậm', 'cham',
                        'yếu', 'yeu', 'hỏng', 'hong', 'lỗi', 'loi', 'lag', 'giật',
                        'giat', 'kém', 'kem', 'đắt', 'dat', 'nóng', 'nong',
                        'ồn', 'mờ', 'mo', 'nặng', 'nang', 'khó', 'kho',
                        'thái độ', 'thai do', 'hách', 'hach', 'ghẻ lạnh', 'ghe lanh',
                        'cọc', 'coc', 'khó chịu', 'kho chiu', 'bẩn', 'ban', 'tồi', 'toi'}

        pos_parts = []
        neg_parts = []

        for part in parts:
            part_lower = part.lower()
            cleaned_part = self.preprocessor.clean_text(part)
            vec = vectorizer.transform([cleaned_part])
            score = model.decision_function(vec)[0]

            has_pos_kw = any(kw in part_lower for kw in pos_keywords)
            has_neg_kw = any(kw in part_lower for kw in neg_keywords)

            # Phân loại dứt khoát
            if has_pos_kw and not has_neg_kw:
                pos_parts.append(part.strip())
            elif has_neg_kw and not has_pos_kw:
                neg_parts.append(part.strip())
            elif has_pos_kw and has_neg_kw:
                if score > 0:
                    pos_parts.append(part.strip())
                else:
                    neg_parts.append(part.strip())
            else:
                if score > config.MIXED_PART_THRESHOLD:
                    pos_parts.append(part.strip())
                elif score < -config.MIXED_PART_THRESHOLD:
                    neg_parts.append(part.strip())

        # Phân tích kết quả
        if pos_parts and neg_parts:
            explanation = "Bình luận chứa cả cảm xúc tích cực và tiêu cực."
            explanation += f"\n- Phần tích cực: \"{'; '.join(pos_parts)}\""
            explanation += f"\n- Phần tiêu cực: \"{'; '.join(neg_parts)}\""
            return ("mixed", explanation)
        elif neg_parts and not pos_parts:
            explanation = f"Cả hai vế của bình luận đều mang tính tiêu cực."
            explanation += f"\n- Phần tiêu cực: \"{'; '.join(neg_parts)}\""
            return ("negative", explanation)
        elif pos_parts and not neg_parts:
            explanation = f"Cả hai vế của bình luận đều mang tính tích cực."
            explanation += f"\n- Phần tích cực: \"{'; '.join(pos_parts)}\""
            return ("positive", explanation)

        return None

    def _build_explanation(self, original_text, cleaned_text, vectorizer, model, sentiment_word):
        """Xây dựng đoạn giải thích chi tiết cho kết quả dự đoán."""
        try:
            feature_names = vectorizer.get_feature_names_out()
            coef = model.coef_.toarray().flatten()
            vec = vectorizer.transform([cleaned_text])
            vec_array = vec.toarray().flatten()

            # Tính mức đóng góp của từng từ/cụm từ
            contributions = vec_array * coef

            # Lấy top từ tích cực và tiêu cực
            top_pos_idx = contributions.argsort()[-3:][::-1]
            top_neg_idx = contributions.argsort()[:3]

            pos_words = [(feature_names[i], contributions[i]) for i in top_pos_idx if contributions[i] > 0]
            neg_words = [(feature_names[i], abs(contributions[i])) for i in top_neg_idx if contributions[i] < 0]

            explanation = f"Bình luận được đánh giá là {sentiment_word}."

            if pos_words:
                words_str = ", ".join([f'"{w}"' for w, _ in pos_words])
                explanation += f"\n📗 Từ/cụm từ mang tính tích cực: {words_str}"

            if neg_words:
                words_str = ", ".join([f'"{w}"' for w, _ in neg_words])
                explanation += f"\n📕 Từ/cụm từ mang tính tiêu cực: {words_str}"

            if not pos_words and not neg_words:
                explanation += "\nKhông tìm thấy từ khóa nổi bật rõ ràng, kết quả dựa trên ngữ cảnh tổng thể."

            return explanation

        except Exception:
            return f"Bình luận được đánh giá là {sentiment_word} dựa trên phân tích tổng thể nội dung."

    def _result(self, label, explanation, domain):
        """Trả về kết quả dạng dict chuẩn."""
        return {
            'label': label,
            'icon': self.LABEL_ICONS.get(label, ''),
            'explanation': explanation,
            'domain_used': config.DOMAINS.get(domain, domain)
        }
