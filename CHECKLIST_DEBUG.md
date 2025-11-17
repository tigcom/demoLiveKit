# Checklist Debug - Backend không nhận frames

## ✅ Đã fix:
- [x] Backend auto-subscribe tracks (kind=1 được nhận diện)
- [x] Token có đủ permissions
- [x] Track được subscribe thành công

## ❌ Vấn đề hiện tại:
Backend log: `→ user1: 1 track(s)` nhưng KHÔNG thấy `▶ Frame received`

→ **track.recv() không trả về frames**

## Các khả năng:

### 1. Frontend không gửi audio data
**Kiểm tra:**
- Mở frontend console
- Phải thấy: `📢 Mic stats → packets=XXX, bytes=YYY` với packets tăng dần
- Nếu packets=0 → mic không hoạt động

**Test mic:**
```
Mở: http://localhost:5173/test-mic.html
Click "Test Mic"
Nói vào mic
Phải thấy RMS > 0.01
```

### 2. LiveKit server không forward audio
**Kiểm tra:**
- Vào LiveKit dashboard: https://cloud.livekit.io
- Xem room "demo-room"
- Kiểm tra có participant "user1" với audio track không
- Kiểm tra bandwidth/bitrate có > 0 không

### 3. track.recv() API sai cách dùng
**Test:**
```powershell
cd backend
python test_track_recv.py
# Sau đó join từ frontend
```

### 4. Browser không cho phép mic
**Kiểm tra:**
- Chrome: Settings → Privacy → Microphone
- Phải thấy localhost:5173 trong "Allowed"
- Thử browser khác (Firefox, Edge)

### 5. Audio track bị muted
**Kiểm tra frontend console:**
```javascript
// Paste vào console
const tracks = Array.from(document.querySelectorAll('audio'));
console.log('Audio elements:', tracks.length);
tracks.forEach((el, i) => {
    console.log(`Audio ${i}:`, {
        muted: el.muted,
        paused: el.paused,
        src: el.src || el.srcObject
    });
});
```

## Các bước debug tiếp theo:

### Bước 1: Kiểm tra frontend stats
Trong frontend console, sau khi join, chạy:
```javascript
setInterval(async () => {
    const room = window.room; // Nếu có export
    if (!room) return;
    
    const localPub = Array.from(room.localParticipant.trackPublications.values())[0];
    if (!localPub || !localPub.track) return;
    
    const stats = await localPub.track.getRTCStatsReport();
    stats.forEach(report => {
        if (report.type === 'outbound-rtp' && report.kind === 'audio') {
            console.log('📊 Outbound:', {
                packets: report.packetsSent,
                bytes: report.bytesSent,
                timestamp: report.timestamp
            });
        }
    });
}, 2000);
```

### Bước 2: Thêm debug vào main.js
Sau dòng `await room.localParticipant.publishTrack(audioTrack);`, thêm:
```javascript
// Export room for debugging
window.room = room;
window.audioTrack = audioTrack;

// Log track state
setInterval(() => {
    console.log('🎤 Track state:', {
        enabled: audioTrack.mediaStreamTrack.enabled,
        muted: audioTrack.isMuted,
        readyState: audioTrack.mediaStreamTrack.readyState
    });
}, 2000);
```

### Bước 3: Kiểm tra LiveKit connection quality
Frontend console:
```javascript
room.on('connectionQualityChanged', (quality, participant) => {
    console.log('📶 Connection quality:', quality, participant.identity);
});
```

### Bước 4: Test với simple audio track
Thay vì mic, thử publish oscillator (test tone):
```javascript
const audioContext = new AudioContext();
const oscillator = audioContext.createOscillator();
const dest = audioContext.createMediaStreamDestination();
oscillator.connect(dest);
oscillator.start();

const testTrack = dest.stream.getAudioTracks()[0];
const lkTrack = new LocalAudioTrack(testTrack);
await room.localParticipant.publishTrack(lkTrack);
```

## Nếu vẫn không hoạt động:

### Option A: Dùng LiveKit Agents SDK
Thay vì manual recv(), dùng LiveKit Agents framework:
```python
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    participant = await ctx.wait_for_participant()
    
    async for event in rtc.RoomEvent(ctx.room):
        if event.track:
            async for frame in event.track:
                # Process frame
                pass
```

### Option B: Dùng track event thay vì polling
```python
async def on_track_subscribed(track, publication, participant):
    if track.kind == rtc.TrackKind.KIND_AUDIO:
        asyncio.create_task(receive_frames(track, participant))

async def receive_frames(track, participant):
    async for frame in track:
        # Process frame
        logger.info(f"Frame: {len(frame.data)} bytes")
```
