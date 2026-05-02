// lay DOM elements
const commentInput = document.getElementById('comment-input');
const btnAnalyze = document.getElementById('btn-analyze');
const domainSelect = document.getElementById('domain-select');
const charCount = document.getElementById('char-count');
const resultSection = document.getElementById('result-section');
const resultIcon = document.getElementById('result-icon');
const resultLabel = document.getElementById('result-label');
const resultDomain = document.getElementById('result-domain');
const resultComment = document.getElementById('result-comment');
const resultExplanation = document.getElementById('result-explanation');
const historyList = document.getElementById('history-list');
const btnClearHistory = document.getElementById('btn-clear-history');
const btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
const sidebar = document.getElementById('sidebar');

// map label sang class css
const labelClassMap = {
    'Tích cực': 'positive',
    'Tiêu cực': 'negative',
    'Trung lập': 'neutral',
    'Hỗn hợp': 'mixed',
    'Không rõ': 'unclear'
};

// dem ky tu
commentInput.addEventListener('input', () => {
    charCount.textContent = `${commentInput.value.length}/1000`;
});

// click nut phan tich
btnAnalyze.addEventListener('click', () => {
    analyzeComment();
});

// Enter key to submit (Shift+Enter for new line)
commentInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        analyzeComment();
    }
});

// ham phan tich chinh
async function analyzeComment() {
    const comment = commentInput.value.trim();
    if (!comment) {
        alert('Vui lòng nhập bình luận!');
        commentInput.focus();
        return;
    }

    const domain = domainSelect.value;

    // UI: loading state
    const btnText = btnAnalyze.querySelector('.btn-text');
    const btnLoading = btnAnalyze.querySelector('.btn-loading');
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline';
    btnAnalyze.disabled = true;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ comment, domain })
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Show result
        displayResult(data.result, comment);

        // Update history
        if (data.history_entry) {
            addHistoryItem(data.history_entry);
        }

    } catch (err) {
        alert('Lỗi kết nối server! Vui lòng thử lại.');
        console.error(err);
    } finally {
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
        btnAnalyze.disabled = false;
    }
}

// hien thi ket qua len UI
function displayResult(result, comment) {
    resultIcon.textContent = result.icon;
    resultLabel.textContent = result.label;
    resultLabel.className = 'result-label ' + (labelClassMap[result.label] || '');
    resultDomain.textContent = 'Miền: ' + result.domain_used;
    resultComment.textContent = '"' + comment + '"';
    resultExplanation.textContent = result.explanation;

    resultSection.style.display = 'block';
    // Scroll to result
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// them item vao lich su
function addHistoryItem(entry) {
    // Remove empty message
    const emptyMsg = historyList.querySelector('.history-empty');
    if (emptyMsg) emptyMsg.remove();

    const div = document.createElement('div');
    div.className = 'history-item';
    div.dataset.id = entry.id;
    div.innerHTML = `
        <div class="history-item-header">
            <span class="history-item-label">${entry.label}</span>
            <span class="history-item-time">${entry.time}</span>
        </div>
        <div class="history-item-comment">${escapeHtml(entry.comment)}</div>
    `;

    div.addEventListener('click', () => {
        // Deselect all
        document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
        div.classList.add('active');

        // Show this result again
        displayResult({
            label: entry.label,
            icon: entry.icon,
            explanation: entry.explanation,
            domain_used: entry.domain
        }, entry.comment);

        // Fill input
        commentInput.value = entry.comment;
        charCount.textContent = `${entry.comment.length}/1000`;
    });

    historyList.prepend(div);
}

// nut xoa lich su
btnClearHistory.addEventListener('click', async () => {
    if (!confirm('Bạn có chắc muốn xóa toàn bộ lịch sử?')) return;

    try {
        await fetch('/api/history/clear', { method: 'POST' });
        historyList.innerHTML = '<div class="history-empty"><p>Chưa có lịch sử tra cứu.</p></div>';
        resultSection.style.display = 'none';
    } catch (err) {
        console.error(err);
    }
});

// an hien sidebar
btnToggleSidebar.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
});

// load lich su khi vao trang
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        if (data.history && data.history.length > 0) {
            const emptyMsg = historyList.querySelector('.history-empty');
            if (emptyMsg) emptyMsg.remove();

            data.history.forEach(entry => {
                addHistoryItem(entry);
            });
        }
    } catch (err) {
        console.error('Could not load history:', err);
    }
}

// ham chong XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// chay code khi web load xong
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    commentInput.focus();
    
    // Xu ly hover tren nut upload
    const btnUploadLabel = document.getElementById('btn-upload-label');
    if (btnUploadLabel) {
        btnUploadLabel.addEventListener('mouseover', () => {
            btnUploadLabel.style.background = 'var(--accent-lighter)';
        });
        btnUploadLabel.addEventListener('mouseout', () => {
            btnUploadLabel.style.background = 'transparent';
        });
    }
});

// xu ly upload file
const fileUploadInput = document.getElementById('file-upload-input');
if (fileUploadInput) {
    fileUploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const btnUploadLabel = document.getElementById('btn-upload-label');
        const originalText = btnUploadLabel.innerHTML;
        btnUploadLabel.innerHTML = '⏳ Đang xử lý...';
        btnUploadLabel.style.pointerEvents = 'none';
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('domain', domainSelect.value);
        
        try {
            const response = await fetch('/api/import_predict', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                // Tải file về
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                
                // Lay ten file tu header Content-Disposition neu co, khong thi dung default
                const contentDisposition = response.headers.get('Content-Disposition');
                let fileName = 'ket_qua_phan_tich.xlsx';
                if (contentDisposition) {
                    const match = contentDisposition.match(/filename="?([^"]+)"?/);
                    if (match && match[1]) {
                        fileName = match[1];
                    }
                }
                
                a.download = fileName;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                a.remove();
                
                alert('Đã xử lý xong! File kết quả đang được tải xuống.');
            } else {
                const data = await response.json();
                alert(data.error || 'Đã xảy ra lỗi khi xử lý file.');
            }
        } catch (err) {
            console.error(err);
            alert('Lỗi kết nối đến server!');
        } finally {
            btnUploadLabel.innerHTML = originalText;
            btnUploadLabel.style.pointerEvents = 'auto';
            fileUploadInput.value = ''; // Reset input
        }
    });
}

