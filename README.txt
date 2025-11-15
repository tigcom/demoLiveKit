================================================================================
DỰ ÁN: HỆ THỐNG CHATBOT NÓI ĐƯỢC (AI VOICE AGENT)
================================================================================

Đây là một dự án AI giúp bạn nói chuyện với máy tính.

Bạn nói gì → AI nghe → AI suy nghĩ → AI nói lại câu trả lời

================================================================================
CÁC TÀI LIỆU CÓ TRONG THƯ MỤC "docs/"
================================================================================

1. overview.txt
   → Giới thiệu dự án, giải thích hệ thống hoạt động như thế nào

2. components.txt
   → Giải thích chi tiết từng thành phần (Frontend, Backend, Token Server...)

3. setup_and_install.txt
   → Hướng dẫn cài đặt chi tiết từng bước

4. run_project.txt
   → Hướng dẫn chạy dự án từng bước

5. troubleshooting.txt
   → Danh sách lỗi phổ biến và cách sửa


NẾU BẠN ĐÃ CÀI ĐẶT:
  → Đọc: docs/run_project.txt
  → Thực hiện: Chạy dự án theo hướng dẫn

NẾU BẠN CHƯA CÀI ĐẶT:
  → Đọc: docs/setup_and_install.txt
  → Thực hiện: Cài đặt từng bước
  → Rồi: Đọc docs/run_project.txt

NẾU BẠN MUỐN HIỂU DỰ ÁN:
  → Đọc: docs/overview.txt
  → Đọc: docs/components.txt

NẾU BẠN GẶP LỖI:
  → Đọc: docs/troubleshooting.txt
  → Tìm lỗi của bạn
  → Thực hiện cách sửa

================================================================================
CẤU TRÚC DỰ ÁN
================================================================================

livekit-voice-demo/
  ├── docs/                    ← Các file tài liệu
  │   ├── overview.txt
  │   ├── components.txt
  │   ├── setup_and_install.txt
  │   ├── run_project.txt
  │   └── troubleshooting.txt
  │
  ├── backend/                 ← Bộ não AI
  │   ├── agent.py            (chương trình chính)
  │   ├── token_server.py     (cấp vé truy cập)
  │   ├── audio_utils.py      (xử lý âm thanh)
  │   ├── audio_source.py     (phát âm thanh)
  │   ├── test_pipeline.py    (test chức năng)
  │   ├── requirements.txt    (danh sách thư viện)
  │   ├── .env.example        (ví dụ cấu hình)
  │   └── .env                (cấu hình - tự tạo từ .env.example, không share)
  │
  └── frontend/               ← Trang web giao diện
      ├── index.html
      ├── src/
      │   ├── main.js
      │   └── style.css
      ├── package.json
      ├── vite.config.js
      ├── .env.example        (ví dụ cấu hình)
      ├── .env                (cấu hình - tự tạo từ .env.example, không share)
      └── node_modules/       (thư viện - tự động cài)

================================================================================
NHỮNG DỊCH VỤ CẦN CHUẨN BỊ
================================================================================

1. Python (ngôn ngữ lập trình backend)
   → Tải từ: https://www.python.org/downloads/

2. Node.js (ngôn ngữ lập trình frontend)
   → Tải từ: https://nodejs.org/

3. Hugging Face Transformers + PyTorch (mô hình LLM cục bộ)
    → Cài torch và transformers theo hướng dẫn trong docs/setup_and_install.txt
    → Ví dụ cài CPU-only:
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
       pip install transformers
    → Hoặc cài PyTorch cho CUDA 12.8 nếu bạn có GPU:
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

================================================================================
CÁC BƯỚC NHANH
================================================================================

1. Cài Python, Node.js, PyTorch + Transformers

2. Vào thư mục backend, tạo virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Vào thư mục frontend:
   ```powershell
   npm install
   ```

4. Tạo 2 file `.env`:
   
   Backend:
   ```powershell
   cd backend
   copy .env.example .env
   notepad .env  # chỉnh sửa với LiveKit credentials
   ```
   
   Frontend:
   ```powershell
   cd frontend
   copy .env.example .env
   notepad .env  # chỉnh sửa với LiveKit URL
   ```
   
   Nhớ: 2 file `.env` này chứa credentials nhạy cảm và được thêm vào `.gitignore`

5. Chạy dự án (đọc docs/run_project.txt để biết chi tiết):
   ```powershell
   # Terminal 1: Token Server
   cd backend
   uvicorn token_server:app --port 8000
   
   # Terminal 2: Frontend
   cd frontend
   npm run dev
   
   # Terminal 3: Backend Agent
   cd backend
   python agent.py --local-debug --sample sample.wav
   # Hoặc: python agent.py --livekit  (để kết nối thật)
   ```

