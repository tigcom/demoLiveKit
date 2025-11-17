# Setup Local LiveKit Server

## Vấn đề hiện tại:
LiveKit Cloud project `projectdemo-p4s7u6bw.livekit.cloud` trả về 401 error → project đã hết hạn hoặc bị xóa.

## Solution 1: Tạo LiveKit Cloud project mới (Easiest)

1. Vào: https://cloud.livekit.io
2. Sign in / Sign up
3. Create new project
4. Copy credentials:
   - WebSocket URL: `wss://your-project.livekit.cloud`
   - API Key: `APIxxxxx`
   - API Secret: `xxxxxx`

5. Update `backend/.env`:
```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIxxxxx
LIVEKIT_API_SECRET=xxxxxx
```

6. Update `frontend/.env`:
```env
VITE_LIVEKIT_URL=wss://your-project.livekit.cloud
```

7. Restart backend và frontend

## Solution 2: Run LiveKit Server Locally

### Bước 1: Download LiveKit Server

**Windows:**
```powershell
# Download từ: https://github.com/livekit/livekit/releases
# Hoặc dùng chocolatey:
choco install livekit-server
```

**Hoặc dùng Docker:**
```powershell
docker pull livekit/livekit-server
```

### Bước 2: Tạo config file

Tạo file `livekit-server-config.yaml`:
```yaml
port: 7880
rtc:
  port_range_start: 50000
  port_range_end: 60000
  use_external_ip: false
keys:
  devkey: secret
```

### Bước 3: Chạy server

**Nếu dùng binary:**
```powershell
livekit-server --config livekit-server-config.yaml
```

**Nếu dùng Docker:**
```powershell
docker run --rm -p 7880:7880 -p 50000-60000:50000-60000/udp `
  -v ${PWD}/livekit-server-config.yaml:/livekit.yaml `
  livekit/livekit-server --config /livekit.yaml
```

### Bước 4: Update .env files

**backend/.env:**
```env
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

**frontend/.env:**
```env
VITE_LIVEKIT_URL=ws://localhost:7880
```

### Bước 5: Test connection

```powershell
# Test với curl
curl http://localhost:7880/

# Hoặc mở browser: http://localhost:7880/
```

## Recommended: Dùng LiveKit Cloud

LiveKit Cloud free tier đủ để test và development. Local server phức tạp hơn vì phải config network/firewall.

## Sau khi setup xong:

1. Restart token server:
```powershell
cd backend
uvicorn token_server:app --port 8000
```

2. Restart backend agent:
```powershell
cd backend
python agent.py --livekit
```

3. Restart frontend:
```powershell
cd frontend
npm run dev
```

4. Test lại!
