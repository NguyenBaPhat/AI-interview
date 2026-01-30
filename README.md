# AI Interview Assistant

Ứng dụng web đơn giản để hỗ trợ phỏng vấn với AI, tương tự ParakeetAI nhưng đơn giản hơn.

## Tính năng hiện tại

- ✅ UI đơn giản với nút chia sẻ
- ✅ Chọn cửa sổ/tab để chia sẻ âm thanh
- ✅ Ghi âm real-time từ cửa sổ được chọn
- ✅ Chuyển đổi âm thanh sang văn bản real-time bằng Whisper
- ✅ Hiển thị bản ghi trên UI

## Cài đặt

1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Cài đặt ffmpeg (cần cho audio processing):
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# Hoặc sử dụng conda
conda install ffmpeg

# Hoặc sử dụng snap
sudo snap install ffmpeg
```

## Chạy ứng dụng

```bash
python app.py
```

Sau đó mở trình duyệt và truy cập: `http://localhost:8000`

## Sử dụng

1. Nhấn nút "🔊 Chọn và Chia sẻ Âm thanh"
2. Chọn cửa sổ/tab bạn muốn ghi âm (ví dụ: Zoom, Teams, Google Meet)
3. **Quan trọng**: Khi chọn tab, nhớ bật "Chia sẻ âm thanh" trong dialog của trình duyệt
4. Ứng dụng sẽ tự động ghi âm và hiển thị bản ghi real-time

## Lưu ý

- Lần đầu chạy sẽ tự động tải Faster-Whisper model (tiny model ~75MB)
- Model được tối ưu cho real-time transcription với tốc độ nhanh
- Đảm bảo bạn đã bật "Chia sẻ âm thanh" khi chọn tab/cửa sổ
- Ứng dụng hỗ trợ tiếng Anh (English) cho transcription real-time

## Cấu trúc project

```
AI-interview/
├── app.py              # FastAPI backend
├── templates/
│   └── index.html      # Frontend HTML
├── static/
│   ├── style.css       # CSS styling
│   └── script.js       # Frontend JavaScript
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

