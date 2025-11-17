# Debug: Backend không nhận được audio từ Frontend

## Vấn đề đã fix:

### 1. Thiếu auto-subscribe tracks
**Vấn đề:** Backend chỉ lắng nghe `track_subscribed` nhưng không chủ động subscribe.

**Fix:** Đã thêm:
- Handler `_on_track_published` để tự động subscribe khi participant publish track
- Logic subscribe tracks có sẵn khi agent connect vào room

### 2. Thiếu logging
**Fix:** Đã thêm logging chi tiết để debug:
- Log số lượng participants và tracks mỗi 5 giây
- Log khi tạo buffer cho participant mới
- Log RMS của audio frames

## Cách test:

### Bước 1: Verify token permissions
```powershell
cd backend
python test_token.py
```

Phải thấy:
```
✅ Token generated successfully!
   can_publish: True
   can_subscribe: True
   can_publish_data: True
```

### Bước 2: Chạy backend với logging đầy đủ
```powershell
cd backend
.venv\Scripts\activate
python agent.py --livekit
```

### Bước 3: Chạy frontend
```powershell
cd frontend
npm run dev
```

### Bước 4: Kiểm tra logs

**Frontend console phải thấy:**
```
✓ Connected: demo-room, Local participant: user1
✓ Local mic published
📡 trackPublished → audio
```

**Backend console phải thấy:**
```
👤 Participant connected: user1
🎵 Track published by user1: kind=audio, sid=TR_xxxxx
🔔 Auto-subscribing to audio track from user1
📢 Track subscribed from user1, kind=audio
📦 Created buffer for participant user1
🔍 Polling 1 participants, 1 have tracks
  → user1: 1 track(s)
  ▶ Frame received from user1: len=3840 bytes, rms=1234.56
```

## Nếu vẫn không nhận được audio:

### Check 1: Verify LiveKit server
```powershell
# Kiểm tra LIVEKIT_URL trong .env
# Phải là wss:// không phải ws://
```

### Check 2: Verify mic permissions
- Mở browser console
- Kiểm tra có lỗi "Permission denied" không
- Thử refresh page và cho phép mic

### Check 3: Verify track stats
Frontend console phải thấy:
```
📢 Mic stats → packets=XXX, bytes=YYY
```
Nếu packets=0 → mic không hoạt động

### Check 4: Network issues
```powershell
# Test kết nối đến LiveKit server
curl -v https://your-livekit-server.livekit.cloud
```

### Check 5: Token mismatch
Đảm bảo:
- Backend và Frontend dùng cùng LIVEKIT_URL
- Token server trả về đúng wsUrl
- Room name giống nhau

## Common issues:

### Issue: "Track subscribed" log xuất hiện nhưng không có frames
**Nguyên nhân:** Track bị muted hoặc không có data

**Fix:** Kiểm tra frontend:
```javascript
// Trong main.js, sau khi publish track
console.log("Track muted?", audioTrack.isMuted);
console.log("Track enabled?", audioTrack.mediaStreamTrack.enabled);
```

### Issue: Frames có data nhưng RMS = 0
**Nguyên nhân:** Mic không capture được âm thanh

**Fix:**
1. Test mic bằng cách ghi âm: Settings → Privacy → Microphone
2. Thử browser khác (Chrome recommended)
3. Kiểm tra mic không bị mute ở OS level

### Issue: Backend crash khi recv() frames
**Nguyên nhân:** Timeout hoặc track bị close

**Fix:** Đã xử lý trong code với try/except và timeout=0.5s

## Debug commands:

```powershell
# Xem log chi tiết hơn
$env:LIVEKIT_LOG_LEVEL="debug"
python agent.py --livekit

# Test pipeline locally (không cần LiveKit)
python agent.py --local-debug --sample sample.wav --output output.wav
```
