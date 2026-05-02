import re
import json
import os
from pyvi import ViTokenizer


class TextPreprocessor:
    # Module tien xu ly text

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.teencode_dict = self._load_json('teencode.json', default={})
        self.negations = set(self._load_json('negations.json', default=[]))
        self.sarcasm_patterns = self._load_json('sarcasm_patterns.json', default=[])
        self.domain_keywords = self._load_json('domain_keywords.json', default={})
        self.re_clean = re.compile(r'[^\w\s]')
        self.re_url = re.compile(r'https?://\S+|www\.\S+')
        self.re_all_numbers = re.compile(r'^\d[\d\s]*$')

        # Vietnamese vowels for language detection
        self.viet_chars = set('àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ')

    def _load_json(self, filename, default=None):
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default if default is not None else {}

    def reload_dictionaries(self):
        # cap nhat lai dict tu json
        self.teencode_dict = self._load_json('teencode.json', default={})
        self.negations = set(self._load_json('negations.json', default=[]))
        self.sarcasm_patterns = self._load_json('sarcasm_patterns.json', default=[])
        self.domain_keywords = self._load_json('domain_keywords.json', default={})

    def is_unclear(self, text):
        # check nhung binh luan ko ro nghia
        text = text.strip()

        # Quá ngắn
        if len(text) < 2:
            return True, "Bình luận quá ngắn, không đủ thông tin để phân tích."

        # Chỉ có 1 từ duy nhất và là từ thô tục / cảm thán → không rõ ngữ cảnh
        words = text.lower().split()
        vulgar_intensifiers = {
            'cc', 'cặc', 'cứt', 'cức', 'lồn', 'đĩ', 'chó', 'đm', 'dm',
            'đcm', 'dcm', 'đkm', 'dkm', 'vl', 'vcl', 'vkl', 'vch',
            'shit', 'fuck', 'fk', 'fck', 'damn', 'wtf', 'ỉa', 'cút'
        }
        if len(words) == 1 and words[0] in vulgar_intensifiers:
            return True, "Bình luận chỉ chứa từ cảm thán, không đủ ngữ cảnh để phân tích."

        # Chứa URL/link
        if self.re_url.search(text):
            return True, "Bình luận chứa đường link, có thể là spam."

        # Toàn chữ số
        if self.re_all_numbers.match(text):
            return True, "Bình luận chỉ chứa toàn chữ số, không thể phân tích cảm xúc."

        # Kiểm tra ký tự vô nghĩa (gibberish)
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
        # ghep tu phu dinh vao tu tiep theo
        words = text.split()
        processed = []
        i = 0
        while i < len(words):
            if words[i] in self.negations and i + 1 < len(words):
                processed.append(f"không_{words[i+1]}")
                i += 2
            else:
                processed.append(words[i])
                i += 1
        return " ".join(processed)

    def normalize_repeated_chars(self, text):
        # rut gon cac ky tu bi keo dai, vd: khongggg -> khong
        # Rút gọn ký tự lặp 3+ lần thành 1 lần
        # Ví dụ: "khôngggggg" → "không", "quááááá" → "quá"
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def clean_text(self, text):
        # lam sach data: xoa ky tu, teencode, phu dinh...
        text = str(text).lower()
        text = self.normalize_repeated_chars(text)
        text = self.re_clean.sub(' ', text)
        words = [self.teencode_dict.get(w, w) for w in text.split()]
        text = self.process_negation(" ".join(words))
        return ViTokenizer.tokenize(text)

    def detect_sarcasm(self, text):
        # phat hien mia mai bang regex
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
        # tu nhan dien domain cua binh luan
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
