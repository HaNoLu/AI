import os 
import re 
import joblib 
import numpy as np 

import sys 
sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))

import config 
from core .preprocess import TextPreprocessor 


class SentimentPredictor :
    POSITIVE ="Tích cực"
    NEGATIVE ="Tiêu cực"
    NEUTRAL ="Trung lập"
    MIXED ="Hỗn hợp"
    UNCLEAR ="Không rõ"

    LABEL_ICONS ={
    "Tích cực":"😊",
    "Tiêu cực":"😞",
    "Trung lập":"😐",
    "Hỗn hợp":"🤔",
    "Không rõ":"❓"
    }

    def __init__ (self ):
        self .preprocessor =TextPreprocessor (config .DATA_DIR )
        self .models ={}
        self ._load_models ()

    def _load_models (self ):
        for domain in config .DOMAINS :
            model_path =os .path .join (config .MODELS_DIR ,domain ,'sentiment_model.pkl')
            vec_path =os .path .join (config .MODELS_DIR ,domain ,'tfidf_vectorizer.pkl')
            if os .path .exists (model_path )and os .path .exists (vec_path ):
                model =joblib .load (model_path )
                vectorizer =joblib .load (vec_path )
                self .models [domain ]=(model ,vectorizer )

    def reload_models (self ):
        self .models ={}
        self ._load_models ()

    def predict (self ,text ,domain ='auto'):
        is_unclear ,reason =self .preprocessor .is_unclear (text )
        if is_unclear :
            return self ._result (self .UNCLEAR ,reason ,'none')

        if domain =='auto':
            domain =self .preprocessor .detect_domain (text )

        if domain not in self .models :
            domain ='general'
        if domain not in self .models :
            return self ._result (self .UNCLEAR ,
            "Chưa có mô hình AI nào được huấn luyện. Vui lòng huấn luyện mô hình trước.",
            'none')

        model ,vectorizer =self .models [domain ]

        contrast_result =self ._check_contrast (text ,model ,vectorizer )
        if contrast_result :
            result_type ,explanation =contrast_result 
            if result_type =="mixed":
                return self ._result (self .MIXED ,explanation ,domain )
            elif result_type =="negative":
                return self ._result (self .NEGATIVE ,explanation ,domain )
            elif result_type =="positive":
                return self ._result (self .POSITIVE ,explanation ,domain )

        cleaned =self .preprocessor .clean_text (text )
        vec =vectorizer .transform ([cleaned ])
        prediction =model .predict (vec )[0 ]
        confidence =abs (model .decision_function (vec )[0 ])

        if confidence <config .NEUTRAL_THRESHOLD :
            explanation =self ._build_explanation (text ,cleaned ,vectorizer ,model ,"trung lập")
            return self ._result (self .NEUTRAL ,explanation ,domain )



        if prediction ==1 :
            label =self .POSITIVE 
            sentiment_word ="tích cực"
        else :
            label =self .NEGATIVE 
            sentiment_word ="tiêu cực"

        is_sarcasm ,sarcasm_explanation =self .preprocessor .detect_sarcasm (text )
        if is_sarcasm :
            if label ==self .POSITIVE :
                label =self .NEGATIVE 
                sentiment_word ="tiêu cực"
            else :
                label =self .POSITIVE 
                sentiment_word ="tích cực"

        explanation =self ._build_explanation (text ,cleaned ,vectorizer ,model ,sentiment_word )
        if is_sarcasm :
            explanation =f"Phat hien mia mai: {sarcasm_explanation}\n{explanation}"

        return self ._result (label ,explanation ,domain )

    def _check_contrast (self ,text ,model ,vectorizer ):
        splitters =(
        r'\bnhưng\b|\bnhung\b|\bnhưng mà\b|\bnhung ma\b'
        r'|\btuy nhiên\b|\btuy nhien\b|\bthế nhưng\b|\bthe nhung\b'
        r'|\bsong\b|\bnhưng lại\b|\bnhung lai\b'
        r'|\btuy vậy\b|\btuy vay\b|\bdù vậy\b|\bdu vay\b'
        )

        if not re .search (splitters ,text ,re .IGNORECASE ):
            return None 

        parts =re .split (splitters ,text ,flags =re .IGNORECASE )
        parts =[p .strip ()for p in parts if len (p .strip ())>2 ]

        if len (parts )<2 :
            return None 

        pos_keywords ={'tốt','tot','đẹp','dep','nhanh','giỏi','gioi','hay',
        'ngon','thích','thich','yêu','yeu','tuyệt','tuyet',
        'xinh','bền','ben','mượt','muot','chất','chat',
        'chuẩn','chuan','ổn','on','ok','oke','rẻ','re',
        'kỹ','ky','cẩn thận','can than','nhiệt tình','nhiet tinh',
        'lịch sự','lich su','vui vẻ','vui ve','thân thiện','than thien'}
        neg_keywords ={'xấu','xau','tệ','te','dở','do','chậm','cham',
        'yếu','yeu','hỏng','hong','lỗi','loi','lag','giật',
        'giat','kém','kem','đắt','dat','nóng','nong',
        'ồn','mờ','mo','nặng','nang','khó','kho',
        'thái độ','thai do','hách','hach','ghẻ lạnh','ghe lanh',
        'cọc','coc','khó chịu','kho chiu','bẩn','ban','tồi','toi'}

        pos_parts =[]
        neg_parts =[]

        for part in parts :
            part_lower =part .lower ()
            cleaned_part =self .preprocessor .clean_text (part )
            vec =vectorizer .transform ([cleaned_part ])
            score =model .decision_function (vec )[0 ]

            has_pos_kw =any (kw in part_lower for kw in pos_keywords )
            has_neg_kw =any (kw in part_lower for kw in neg_keywords )

            if has_pos_kw and not has_neg_kw :
                pos_parts .append (part .strip ())
            elif has_neg_kw and not has_pos_kw :
                neg_parts .append (part .strip ())
            elif has_pos_kw and has_neg_kw :
                if score >0 :
                    pos_parts .append (part .strip ())
                else :
                    neg_parts .append (part .strip ())
            else :
                if score >config .MIXED_PART_THRESHOLD :
                    pos_parts .append (part .strip ())
                elif score <-config .MIXED_PART_THRESHOLD :
                    neg_parts .append (part .strip ())

        if pos_parts and neg_parts :
            explanation ="Bình luận chứa cả cảm xúc tích cực và tiêu cực."
            explanation +=f"\n- Phần tích cực: \"{'; '.join(pos_parts)}\""
            explanation +=f"\n- Phần tiêu cực: \"{'; '.join(neg_parts)}\""
            return ("mixed",explanation )
        elif neg_parts and not pos_parts :
            explanation =f"Cả hai vế của bình luận đều mang tính tiêu cực."
            explanation +=f"\n- Phần tiêu cực: \"{'; '.join(neg_parts)}\""
            return ("negative",explanation )
        elif pos_parts and not neg_parts :
            explanation =f"Cả hai vế của bình luận đều mang tính tích cực."
            explanation +=f"\n- Phần tích cực: \"{'; '.join(pos_parts)}\""
            return ("positive",explanation )

        return None 

    def _build_explanation (self ,original_text ,cleaned_text ,vectorizer ,model ,sentiment_word ):
        try :
            feature_names =vectorizer .get_feature_names_out ()
            coef =np .asarray (model .coef_ ).ravel ()
            vec =vectorizer .transform ([cleaned_text ])
            vec_array =vec .toarray ().flatten ()

            contributions =vec_array *coef 

            word_indices =[idx for idx ,name in enumerate (feature_names )if name .startswith ('word__')]
            pos_contribs =[(idx ,contributions [idx ])for idx in word_indices if contributions [idx ]>0 ]
            neg_contribs =[(idx ,contributions [idx ])for idx in word_indices if contributions [idx ]<0 ]

            pos_contribs .sort (key =lambda x :x [1 ],reverse =True )
            neg_contribs .sort (key =lambda x :x [1 ])

            top_pos_idx =[idx for idx ,_ in pos_contribs [:3 ]]
            top_neg_idx =[idx for idx ,_ in neg_contribs [:3 ]]

            def clean_feature_name (name ):
                if name .startswith ('word__'):
                    name =name [6 :]
                elif name .startswith ('char__'):
                    name =name [6 :]
                return name .replace ('_',' ').strip ()

            pos_seen =set ()
            pos_words =[]
            for i in top_pos_idx :
                if contributions [i ]>0 :
                    cleaned_name =clean_feature_name (feature_names [i ])
                    if cleaned_name and cleaned_name not in pos_seen :
                        pos_words .append ((cleaned_name ,contributions [i ]))
                        pos_seen .add (cleaned_name )

            neg_seen =set ()
            neg_words =[]
            for i in top_neg_idx :
                if contributions [i ]<0 :
                    cleaned_name =clean_feature_name (feature_names [i ])
                    if cleaned_name and cleaned_name not in neg_seen :
                        neg_words .append ((cleaned_name ,abs (contributions [i ])))
                        neg_seen .add (cleaned_name )

            explanation =f"Bình luận được đánh giá là {sentiment_word}."

            if pos_words :
                words_str =", ".join ([f'"{w}"'for w ,_ in pos_words ])
                explanation +=f"\n📗 Từ/cụm từ mang tính tích cực: {words_str}"

            if neg_words :
                words_str =", ".join ([f'"{w}"'for w ,_ in neg_words ])
                explanation +=f"\n📕 Từ/cụm từ mang tính tiêu cực: {words_str}"

            if not pos_words and not neg_words :
                explanation +="\nKhông tìm thấy từ khóa nổi bật rõ ràng, kết quả dựa trên ngữ cảnh tổng thể."

            return explanation 

        except Exception :
            return f"Bình luận được đánh giá là {sentiment_word} dựa trên phân tích tổng thể nội dung."

    def _result (self ,label ,explanation ,domain ):
        return {
        'label':label ,
        'icon':self .LABEL_ICONS .get (label ,''),
        'explanation':explanation ,
        'domain_used':config .DOMAINS .get (domain ,domain )
        }
