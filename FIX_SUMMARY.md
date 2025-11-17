# Tóm tắt các fix cho vấn đề "Backend không nhận audio"

## 🔴 Vấn đề gốc:
Frontend đã publish audio track thành công, nhưng backend không nhận được frames.

## ✅ Các fix đã thực hiện:

### 1. Backend không tự động subscribe tracks
**File:** `backend/agent.py`

**Vấn đề:** Backend chỉ lắng nghe event `track_subscribed` nhưng không chủ động subscribe.

**Fix:** Thêm handler `_on_track_published`:
```python
def _on_track_published(self, publication, participant):
    """Called when a participant publishes a new track"""
    identity = getattr(participant, "identity", "unknown")
    kind = getattr(publication, "kind", "unknown")
    logger.info(f"🎵 Track published by {identity}: kind={kind}, sid={publication.sid}")
    
    # Auto-subscribe to audio tracks
    if kind == "audio" or str(kind).lower() == "audio":
        logger.info(f"🔔 Auto-subscribing to audio track from {identity}")
        publication.set_subscribed(True)
```

Và đăng ký event:
```python
self.room.on("track_published", self._on_track_published)
```

### 2. Không subscribe tracks có sẵn khi agent join
**File:** `backend/agent.py`

**Vấn đề:** Nếu user đã join trước agent, tracks không được subscribe.

**Fix:** Thêm logic subscribe tracks có sẵn:
```python
# Subscribe to existing tracks
track_pubs = getattr(p, "track_publications", {})
for sid, pub in track_pubs.items():
    if pub.subscribed:
        continue
    logger.info(f"🔔 Subscribing to existing track {sid} from {identity}")
    pub.set_subscribed(True)
```

### 3. Token thiếu permissions
**File:** `backend/token_server.py`

**Vấn đề:** Token không có quyền `can_publish`, `can_subscribe`, `can_publish_data`.

**Fix:** Thêm permissions vào VideoGrants:
```python
VideoGrants(
    room_join=True,
    room=req.room,
    can_publish=True,        # ✅ Cho phép publish audio/video
    can_subscribe=True,      # ✅ Cho phép subscribe tracks
    can_publish_data=True,   # ✅ Cho phép gửi data messages
)
```

**File:** `backend/agent.py` (generate_token_directly)

**Fix tương tự:**
```python
vg = VideoGrants(
    room_join=True, 
    room=ROOM, 
    can_publish=True, 
    can_subscribe=True,
    can_publish_data=True
)
```

### 4. Thiếu logging để debug
**File:** `backend/agent.py`

**Fix:** Thêm logging chi tiết:
- Log số participants và tracks mỗi 5 giây
- Log khi tạo buffer cho participant
- Log RMS của audio frames

## 📝 Files mới tạo:

1. **`backend/test_token.py`** - Script test token permissions
2. **`DEBUG_AUDIO_NOT_RECEIVED.md`** - Hướng dẫn debug chi tiết
3. **`QUICK_TEST.md`** - Hướng dẫn test nhanh

## 🧪 Cách test:

Xem file `QUICK_TEST.md` để biết chi tiết.

Tóm tắt:
1. Chạy token server
2. Chạy backend agent
3. Chạy frontend
4. Click "Join & Publish Mic"
5. Nói vào mic
6. Kiểm tra logs

## 🎯 Kết quả mong đợi:

Backend console sẽ hiển thị:
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

## 🔧 Nếu vẫn không hoạt động:

Xem file `DEBUG_AUDIO_NOT_RECEIVED.md` để biết các bước debug chi tiết.
