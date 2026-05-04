import os 
import sys 
import io 
import pandas as pd 
import joblib 
import numpy as np 

if sys .stdout .encoding !='utf-8':
    sys .stdout =io .TextIOWrapper (sys .stdout .buffer ,encoding ='utf-8',errors ='replace')
    sys .stderr =io .TextIOWrapper (sys .stderr .buffer ,encoding ='utf-8',errors ='replace')

from sklearn .model_selection import StratifiedKFold 
from sklearn .feature_extraction .text import TfidfVectorizer 
from sklearn .svm import LinearSVC 
from sklearn .pipeline import FeatureUnion 
from sklearn .metrics import accuracy_score ,classification_report 
from sklearn .utils import resample 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
import config 
from core .preprocess import TextPreprocessor 





def _safe_read_csv (path ):
    try :
        return pd .read_csv (path )
    except Exception :
        return None 


def train_model (csv_path ,domain ='general'):
    print (f"=== Bắt đầu huấn luyện mô hình cho miền: {config.DOMAINS.get(domain, domain)} ===")

    data_dir =os .path .dirname (csv_path )
    main_df =_safe_read_csv (csv_path )
    dev_df =_safe_read_csv (os .path .join (data_dir ,'dev.csv'))
    test_df =_safe_read_csv (os .path .join (data_dir ,'test.csv'))

    if main_df is None or 'text'not in main_df .columns or 'label'not in main_df .columns :
        raise ValueError (f"File dữ liệu chính không hợp lệ: {csv_path}")


    main_df =main_df [['label','text']].dropna (subset =['text','label'])
    if dev_df is not None and 'text'in dev_df .columns and 'label'in dev_df .columns :
        dev_df =dev_df [['label','text']].dropna (subset =['text','label'])
    else :
        dev_df =None 
    if test_df is not None and 'text'in test_df .columns and 'label'in test_df .columns :
        test_df =test_df [['label','text']].dropna (subset =['text','label'])
    else :
        test_df =None 


    if dev_df is not None :
        train_df =pd .concat ([main_df ,dev_df ],ignore_index =True )
    else :
        train_df =main_df .copy ()

    train_df ['label']=train_df ['label'].astype (int )
    if test_df is not None :
        test_df ['label']=test_df ['label'].astype (int )

    print (f"Số lượng train: {len(train_df)}")
    if test_df is not None :
        print (f"Số lượng test (từ test.csv): {len(test_df)}")

    preprocessor =TextPreprocessor (config .DATA_DIR )
    print ("Đang tiền xử lý văn bản...")
    train_df ['clean_text']=train_df ['text'].apply (preprocessor .clean_text )
    if test_df is not None :
        test_df ['clean_text']=test_df ['text'].apply (preprocessor .clean_text )


    X =train_df ['clean_text'].values 
    y =train_df ['label'].values 

    n_splits =5 

    min_count =train_df ['label'].value_counts ().min ()
    if min_count <n_splits :
        n_splits =min_count 
        
    if n_splits < 2:
        print("Cảnh báo: Dữ liệu không đủ để chạy StratifiedKFold. Sẽ bỏ qua Cross Validation.")
        fold_accs = []
    else:
        print (f"Chạy StratifiedKFold với {n_splits} splits")
    
        skf =StratifiedKFold (n_splits =n_splits ,shuffle =True ,random_state =config .RANDOM_STATE )
        fold_accs =[]

    if n_splits >= 2:
        for fold_idx ,(train_idx ,val_idx )in enumerate (skf .split (X ,y ),start =1 ):
            X_tr =pd .Series (X [train_idx ])
            y_tr =pd .Series (y [train_idx ])
            X_val =pd .Series (X [val_idx ])
            y_val =pd .Series (y [val_idx ])
    
            tr_df =pd .DataFrame ({'clean_text':X_tr ,'label':y_tr })
    
            vectorizer =FeatureUnion ([
            (
            'word',
            TfidfVectorizer (
            analyzer ='word',
            ngram_range =config .TFIDF_NGRAM_RANGE ,
            max_features =config .TFIDF_MAX_FEATURES ,
            min_df =1 ,
            max_df =0.95 ,
            sublinear_tf =True ,
            token_pattern =r'(?u)\b\w+\b',
            )
            ),
            (
            'char',
            TfidfVectorizer (
            analyzer ='char_wb',
            ngram_range =(2 ,6 ),
            max_features =5000 ,
            min_df =1 ,
            sublinear_tf =True ,
            )
            ),
            ])
            X_tr_vec =vectorizer .fit_transform (tr_df ['clean_text'])
            model =LinearSVC (C =config .SVM_C ,class_weight ='balanced')
            model .fit (X_tr_vec ,tr_df ['label'])
    
            X_val_vec =vectorizer .transform (X_val )
            y_pred =model .predict (X_val_vec )
            acc =accuracy_score (y_val ,y_pred )
            fold_accs .append (acc )
            print (f"Fold {fold_idx} acc: {acc * 100:.2f}%")
    
        print (f"Avg CV accuracy: {np.mean(fold_accs) * 100:.2f}% (+/- {np.std(fold_accs) * 100:.2f}%)")


    full_train =train_df [['clean_text','label']]
    vectorizer =FeatureUnion ([
    (
    'word',
    TfidfVectorizer (
    analyzer ='word',
    ngram_range =config .TFIDF_NGRAM_RANGE ,
    max_features =config .TFIDF_MAX_FEATURES ,
    min_df =1 ,
    max_df =0.95 ,
    sublinear_tf =True ,
    token_pattern =r'(?u)\b\w+\b',
    )
    ),
    (
    'char',
    TfidfVectorizer (
    analyzer ='char_wb',
    ngram_range =(2 ,6 ),
    max_features =5000 ,
    min_df =1 ,
    sublinear_tf =True ,
    )
    ),
    ])
    X_full_vec =vectorizer .fit_transform (full_train ['clean_text'])
    final_model =LinearSVC (C =config .SVM_C ,class_weight ='balanced')
    final_model .fit (X_full_vec ,full_train ['label'])

    if test_df is not None :
        X_test_vec =vectorizer .transform (test_df ['clean_text'])
        y_test =test_df ['label']
        y_pred =final_model .predict (X_test_vec )
        acc =accuracy_score (y_test ,y_pred )
        print (f"\nĐộ chính xác trên tập test.csv: {acc * 100:.2f}%")
        print ("\nBáo cáo chi tiết:")
        print (classification_report (y_test ,y_pred ,target_names =['Tiêu cực','Tích cực']))
    else :
        acc =np .mean (fold_accs )

    model_dir =os .path .join (config .MODELS_DIR ,domain )
    os .makedirs (model_dir ,exist_ok =True )

    model_path =os .path .join (model_dir ,'sentiment_model.pkl')
    vec_path =os .path .join (model_dir ,'tfidf_vectorizer.pkl')

    joblib .dump (final_model ,model_path )
    joblib .dump (vectorizer ,vec_path )

    print (f"\nĐã lưu mô hình tại: {model_path}")
    print (f"Đã lưu vectorizer tại: {vec_path}")
    print (f"=== Hoàn tất huấn luyện miền: {config.DOMAINS.get(domain, domain)} ===\n")

    return final_model ,vectorizer ,acc 


if __name__ =='__main__':
    csv_path =os .path .join (config .DATA_DIR ,'data.csv')
    if os .path .exists (csv_path ):
        train_model (csv_path ,domain ='general')
    else :
        print (f"Không tìm thấy file dữ liệu: {csv_path}")
