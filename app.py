import os
import sys
import io
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, send_file

if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from core.predictor import SentimentPredictor

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = SentimentPredictor()
    return predictor


@app.route('/')
def index():
    return render_template('index.html', domains=config.DOMAINS)

#Rou
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data or 'comment' not in data:
        return jsonify({'error': 'Vui lòng nhập bình luận.'}), 400

    comment = data['comment'].strip()
    domain = data.get('domain', 'auto')

    if not comment:
        return jsonify({'error': 'Bình luận không được để trống.'}), 400

    try:
        pred = get_predictor()
        result = pred.predict(comment, domain=domain)

        if 'history' not in session:
            session['history'] = []

        history_entry = {
            'id': len(session['history']) + 1,
            'comment': comment,
            'label': result['label'],
            'icon': result['icon'],
            'explanation': result['explanation'],
            'domain': result['domain_used'],
            'time': datetime.now().strftime('%H:%M:%S %d/%m/%Y')
        }
        session['history'].insert(0, history_entry)
        session.modified = True

        return jsonify({
            'success': True,
            'result': result,
            'history_entry': history_entry
        })

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500


@app.route('/api/history', methods=['GET'])
def api_history():
    history = session.get('history', [])
    return jsonify({'history': history})


@app.route('/api/history/clear', methods=['POST'])
def api_clear_history():
    session['history'] = []
    session.modified = True
    return jsonify({'success': True})


@app.route('/api/import_predict', methods=['POST'])
def api_import_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file tải lên.'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file nào được chọn.'}), 400
        
    domain = request.form.get('domain', 'auto')
    
    try:
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Định dạng file không được hỗ trợ. Vui lòng tải lên file .csv hoặc .xlsx'}), 400
            
        comment_col = None
        possible_cols = ['comment', 'text', 'nội dung', 'noi dung', 'binh luan', 'bình luận', 'review', 'content']
        
        for col in df.columns:
            if str(col).lower().strip() in possible_cols:
                comment_col = col
                break
                
        if comment_col is None:
            if len(df.columns) > 0:
                comment_col = df.columns[0]
            else:
                return jsonify({'error': 'File không có dữ liệu hợp lệ.'}), 400
                
        pred = get_predictor()
        labels = []
        domains = []
        explanations = []
        
        for index, row in df.iterrows():
            comment = str(row[comment_col]) if pd.notna(row[comment_col]) else ""
            if not comment.strip():
                labels.append("Không có nội dung")
                domains.append("")
                explanations.append("")
                continue
                
            result = pred.predict(comment, domain=domain)
            labels.append(result['label'])
            domains.append(result['domain_used'])
            explanations.append(result['explanation'])
            
        df['Nhãn dự đoán'] = labels
        df['Miền dữ liệu'] = domains
        df['Giải thích'] = explanations
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'ket_qua_phan_tich_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
        
    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý file: {str(e)}'}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("  HỆ THỐNG PHÂN TÍCH CẢM XÚC BÌNH LUẬN")
    print("=" * 50)

    general_model = os.path.join(config.MODELS_DIR, 'general', 'sentiment_model.pkl')
    if not os.path.exists(general_model):
        print("\n⚠️  Chưa có mô hình AI! Đang tiến hành huấn luyện...")
        from core.train import train_model
        csv_path = os.path.join(config.DATA_DIR, 'data.csv')
        if os.path.exists(csv_path):
            train_model(csv_path, domain='general')
        else:
            print(f"❌ Không tìm thấy file dữ liệu: {csv_path}")
            sys.exit(1)

    print("\n🚀 Khởi động server...")
    print("📍 Truy cập: http://127.0.0.1:5000\n")
    app.run(debug=config.DEBUG, host='0.0.0.0', port=5000)
