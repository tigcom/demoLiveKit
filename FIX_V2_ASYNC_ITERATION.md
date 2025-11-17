# Fix V2: Dùng Async Iteration thay vì Polling

## Vấn đề:
Backend đã subscribe track thành công nhưng `track.recv()` trong polling thread không trả về frames.

## Nguyên nhân:
LiveKit Python SDK khuyến nghị dùng **async iteration** (`async for frame in track`) thay vì manual polling với `track.recv()`.

## Fix:
Thay đổi từ:
```python
# ❌ Cách cũ: Polling trong thread
def _audio_receiver_thread(self):
    while self.is_running:
        for track in tracks:
            fut = asyncio.run_coroutine_threadsafe(track.recv(), self.loop)
            frame = fut.result(timeout=0.5)
            # Process frame...
```

Sang:
```python
# ✅ Cách mới: Async iteration
async def _receive_frames_async(self, track, participant_id):
    async for frame in track:
        # Process frame...
```

## Thay đổi:
1. Thêm method `_receive_frames_async()` để nhận frames bằng async iteration
2. Gọi method này trong `_on_track_subscribed()` khi track được subscribe
3. Giữ lại `_audio_receiver_thread()` để log status (có thể xóa sau)

## Test:
```powershell
# Restart backend
python agent.py --livekit

# Join từ frontend và nói vào mic
# Phải thấy:
🎧 Starting async frame receiver for user1
▶ Frame received from user1: len=3840 bytes, rms=1234.56
```

## Nếu vẫn không hoạt động:
Xem file `CHECKLIST_DEBUG.md` để debug từng bước.
