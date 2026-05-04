import re
import json
import os
from pyvi import ViTokenizer


class TextPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.teencode_dict = self._load_json('teencode.json', default={})
        self.negations = set(self._load_json('negations.json', default=[]))
        for w in ['tệ', 'kém', 'ít']:
            if w in self.negations:
                self.negations.remove(w)
        self.sarcasm_patterns = self._load_json('sarcasm_patterns.json', default=[])
        self.domain_keywords = self._load_json('domain_keywords.json', default={})
        self.re_clean = re.compile(r'[^\w\s]')
        self.re_url = re.compile(r'https?://\S+|www\.\S+')
        self.re_all_numbers = re.compile(r'^\d[\d\s]*$')

        self.viet_chars = set('àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ')

    def _load_json(self, filename, default=None):
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default if default is not None else {}

    def reload_dictionaries(self):
        self.teencode_dict = self._load_json('teencode.json', default={})
        self.negations = set(self._load_json('negations.json', default=[]))
        for w in ['tệ', 'kém', 'ít']:
            if w in self.negations:
                self.negations.remove(w)
        self.sarcasm_patterns = self._load_json('sarcasm_patterns.json', default=[])
        self.domain_keywords = self._load_json('domain_keywords.json', default={})

    def is_unclear(self, text):
        text = text.strip()

        if len(text) < 2:
            return True, "Bình luận quá ngắn, không đủ thông tin để phân tích."

        words = text.lower().split()
        vulgar_intensifiers = {
            'cc', 'cặc', 'cứt', 'cức', 'lồn', 'đĩ', 'chó', 'đm', 'dm',
            'đcm', 'dcm', 'đkm', 'dkm', 'vl', 'vcl', 'vkl', 'vch',
            'shit', 'fuck', 'fk', 'fck', 'damn', 'wtf', 'ỉa', 'cút'
        }
        if len(words) == 1 and words[0] in vulgar_intensifiers:
            return True, "Bình luận chỉ chứa từ cảm thán, không đủ ngữ cảnh để phân tích."

        if self.re_url.search(text):
            return True, "Bình luận chứa đường link, có thể là spam."

        if self.re_all_numbers.match(text):
            return True, "Bình luận chỉ chứa toàn chữ số, không thể phân tích cảm xúc."

        words = text.split()
        if len(words) > 0:
            gibberish_count = 0
            for w in words:
                w_lower = w.lower()
                if len(w_lower) > 2:
                    has_vowel = any(c in w_lower for c in 'aeiouàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ')
                    if not has_vowel:
                        gibberish_count += 1
            if gibberish_count / len(words) > 0.7:
                return True, "Bình luận chứa nhiều ký tự không rõ nghĩa."

        return False, ""

    def process_negation(self, text):
        words = text.split()
        processed = []
        i = 0
        while i < len(words):
            if words[i] in self.negations and i + 1 < len(words):
                w = words[i+1]
                pos_to_neg = {
                    "tốt": "xấu", "đẹp": "xấu", "ngon": "dở", "hay": "dở",
                    "thích": "ghét", "yêu": "ghét", "tuyệt": "tệ",
                    "xuất sắc": "tệ", "giỏi": "kém", "nhanh": "chậm",
                    "chất": "kém", "mượt": "lag", "rẻ": "đắt", "bền": "hỏng",
                    "xinh": "xấu", "mạnh": "yếu", "đỉnh": "tệ", "xịn": "đểu",
                    "ổn": "tệ", "ok": "tệ", "oke": "tệ", "chuẩn": "lỗi"
                }
                neg_to_pos = {
                    "xấu": "đẹp", "tệ": "tốt", "dở": "hay", "chậm": "nhanh",
                    "hỏng": "bền", "lỗi": "chuẩn", "ghét": "thích",
                    "kém": "tốt", "lag": "mượt", "giật": "mượt", "nóng": "mát",
                    "đắt": "rẻ", "yếu": "mạnh", "bẩn": "sạch", "tồi": "tốt",
                    "độc": "lành"
                }
                if w in pos_to_neg:
                    processed.append(f"không_{w}")
                    processed.append(pos_to_neg[w])
                elif w in neg_to_pos:
                    processed.append(f"không_{w}")
                    processed.append(neg_to_pos[w])
                else:
                    processed.append(f"không_{w}")
                i += 2
            else:
                processed.append(words[i])
                i += 1
        return " ".join(processed)

    def normalize_repeated_chars(self, text):
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def clean_text(self, text):
        text = str(text).lower()
        text = self.normalize_repeated_chars(text)
        text = self.re_clean.sub(' ', text)
        words = [self.teencode_dict.get(w, w) for w in text.split()]
        text = self.process_negation(" ".join(words))
        return ViTokenizer.tokenize(text)

    def detect_sarcasm(self, text):
        text_lower = text.lower()
        for pattern_info in self.sarcasm_patterns:
            pattern = pattern_info.get('pattern', '')
            try:
                if re.search(pattern, text_lower):
                    return True, pattern_info.get('explanation', 'Phát hiện dấu hiệu mỉa mai trong bình luận.')
            except re.error:
                continue
        return False, ""

    def detect_domain(self, text):
        text_lower = text.lower()
        scores = {}
        for domain, info in self.domain_keywords.items():
            keywords = info.get('keywords', [])
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores, key=scores.get)
        return 'general'
