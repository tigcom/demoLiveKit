# Quick Test - Kiểm tra audio flow

## Các vấn đề đã fix:

1. ✅ **Backend không auto-subscribe tracks** → Đã thêm `_on_track_published` handler
2. ✅ **Token thiếu permissions** → Đã thêm `can_publish`, `can_subscribe`, `can_publish_data`
3. ✅ **Thiếu logging** → Đã thêm logs chi tiết để debug

## Test ngay:

### Terminal 1: Token Server
```powershell
cd livekit-voice-demo/backend
.venv\Scripts\activate
uvicorn token_server:app --port 8000
```

Phải thấy:
```
✓ LIVEKIT_API_KEY: (set)
✓ LIVEKIT_API_SECRET: (set)
```

### Terminal 2: Backend Agent
```powershell
cd livekit-voice-demo/backend
.venv\Scripts\activate
python agent.py --livekit
```

Phải thấy:
```
Connected to room demo-room
🔊 Audio receiver thread started
```

### Terminal 3: Frontend
```powershell
cd livekit-voice-demo/frontend
npm run dev
```

Mở browser: http://localhost:5173

### Test flow:

1. **Click "Join & Publish Mic"**
   
   Frontend console phải thấy:
   ```
   ✓ Connected: demo-room, Local participant: user1
   ✓ Local mic published
   📡 trackPublished → audio
   📢 Mic stats → packets=XXX, bytes=YYY
   ```

2. **Backend phải log:**
   ```
   👤 Participant connected: user1
   Published greeting for user1
   🎵 Track published by user1: kind=audio, sid=TR_xxxxx
   🔔 Auto-subscribing to audio track from user1
   📢 Track subscribed from user1, kind=audio
   📦 Created buffer for participant user1
   ```

3. **Nói vào mic**
   
   Backend phải log:
   ```
   🔍 Polling 1 participants, 1 have tracks
     → user1: 1 track(s)
   ▶ Frame received from user1: len=3840 bytes, rms=1234.56
   ```

4. **Sau khi ngừng nói (10 frames im lặng)**
   
   Backend phải log:
   ```
   📨 Published data: transcript
   📨 Published data: greeting
   ```
   
   Frontend phải nhận được:
   - Transcript của bạn
   - Audio reply từ agent

## Nếu không thấy frames:

### Debug 1: Kiểm tra token
```powershell
cd backend
python test_token.py
```

### Debug 2: Kiểm tra mic trong browser
```javascript
// Paste vào browser console
navigator.mediaDevices.getUserMedia({audio: true})
  .then(stream => {
    console.log("✅ Mic OK:", stream.getAudioTracks()[0].label);
    stream.getTracks().forEach(t => t.stop());
  })
  .catch(err => console.error("❌ Mic error:", err));
```

### Debug 3: Kiểm tra LiveKit connection
Frontend console → Network tab → Filter "ws" → Phải thấy WebSocket connection "101 Switching Protocols"

### Debug 4: Kiểm tra track stats
Frontend console phải thấy mỗi 2 giây:
```
📢 Mic stats → packets=XXX, bytes=YYY
```

Nếu packets tăng → mic đang gửi data
Nếu packets = 0 → mic không hoạt động

## Common fixes:

### Fix 1: Mic bị mute
```javascript
// Trong frontend console
const tracks = Array.from(document.querySelectorAll('audio'));
tracks.forEach(t => console.log("Muted?", t.muted));
```

### Fix 2: Browser không cho phép autoplay
Click vào page trước khi test (Chrome yêu cầu user interaction)

### Fix 3: HTTPS required
Nếu dùng LiveKit cloud, frontend phải chạy trên HTTPS hoặc localhost

### Fix 4: Firewall blocking WebSocket
Tắt firewall tạm thời để test:
```powershell
# Windows
netsh advfirewall set allprofiles state off
# Nhớ bật lại sau khi test!
```
