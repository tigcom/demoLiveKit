"""token_server.py

FastAPI token endpoint for LiveKit access tokens.

Endpoints:
  GET  /healthz              -> {status: ok}
  POST /api/get_token        -> {token: "...", wsUrl: "..."}

ENV vars:
  LIVEKIT_API_KEY            (required)
  LIVEKIT_API_SECRET         (required)
  LIVEKIT_URL                (default: https://cloud.livekit.io)

Import strategy: tries multiple import paths for AccessToken/VideoGrant since
the livekit Python package structure varies across versions.
"""
import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from livekit.api.access_token import AccessToken, VideoGrants
from fastapi.staticfiles import StaticFiles
import pathlib
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()



# Load env vars
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "https://cloud.livekit.io")

app = FastAPI(title="LiveKit Token Server")

# Serve static files (for TTS greetings and similar assets)
static_dir = pathlib.Path(__file__).parent / "static"
greetings_dir = static_dir / "greetings"
static_dir.mkdir(parents=True, exist_ok=True)
greetings_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# CORS: allow localhost:5173 (Vite dev) for development. In production, restrict to your domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",  # common dev port
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class TokenRequest(BaseModel):
    room: str
    identity: str


@app.get("/healthz")
async def healthz():
    """Health check endpoint. Returns 200 if server is up."""
    return {"status": "ok"}


from livekit.api.access_token import AccessToken, VideoGrants

@app.post("/api/get_token")
async def get_token(req: TokenRequest):
    print("🚀 /api/get_token called")
    print(f"Received request: room={req.room}, identity={req.identity}")

    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        print("❌ LIVEKIT_API_KEY or LIVEKIT_API_SECRET not set")
        raise HTTPException(status_code=500, detail="LIVEKIT_API_KEY/SECRET not set")

    try:
        print("1️⃣ Building AccessToken (chained)")
        at = (
            AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
            .with_identity(req.identity)
            .with_name(req.identity)  # bạn có thể đổi thành tên hiển thị khác hoặc bỏ dòng này
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=req.room,
                    can_publish=True,        # ✅ Cho phép publish audio/video
                    can_subscribe=True,      # ✅ Cho phép subscribe tracks từ người khác
                    can_publish_data=True,   # ✅ Cho phép gửi data messages
                )
            )
        )
        print(f"AccessToken built (claims): {getattr(at, 'claims', at)}")

        print("2️⃣ Generating JWT token")
        token = at.to_jwt()
        print(f"Token generated: {token}")

        return {"token": token, "wsUrl": LIVEKIT_URL}

    except Exception as e:
        print(f"❌ Exception during token generation: {e}")
        raise HTTPException(status_code=500, detail=f"Token generation failed: {str(e)}")


