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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Robust import: try multiple paths for AccessToken and VideoGrant
AccessToken = None
VideoGrant = None
IMPORT_ERROR_MSG = None

_import_attempts = [
    ("livekit.api", ["AccessToken", "VideoGrant"]),
    ("livekit", ["AccessToken", "VideoGrant"]),
    ("livekit.jwt", ["AccessToken", "VideoGrant"]),
]

for module_name, classes in _import_attempts:
    try:
        module = __import__(module_name, fromlist=classes)
        AccessToken = getattr(module, "AccessToken", None)
        VideoGrant = getattr(module, "VideoGrant", None)
        if AccessToken and VideoGrant:
            logger.info(f"✓ Imported AccessToken, VideoGrant from {module_name}")
            break
    except (ImportError, AttributeError) as e:
        logger.debug(f"  ✗ Could not import from {module_name}: {e}")

if not (AccessToken and VideoGrant):
    IMPORT_ERROR_MSG = (
        "Could not import AccessToken/VideoGrant from any known livekit module path. "
        "Install 'livekit' package: pip install livekit"
    )
    logger.error(IMPORT_ERROR_MSG)

# Load env vars
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "https://cloud.livekit.io")

app = FastAPI(title="LiveKit Token Server")

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


@app.post("/api/get_token")
async def get_token(req: TokenRequest):
    """
    Generate a LiveKit access token.
    
    Returns:
      {token: "...", wsUrl: "..."}  on success
      
    Errors:
      500 if LIVEKIT_API_KEY/SECRET not set
      500 if livekit SDK not installed
    """
    # Validate env
    if not LIVEKIT_API_KEY:
        logger.error("LIVEKIT_API_KEY not set in environment")
        raise HTTPException(
            status_code=500,
            detail="LIVEKIT_API_KEY must be set in environment (.env or system env)"
        )
    if not LIVEKIT_API_SECRET:
        logger.error("LIVEKIT_API_SECRET not set in environment")
        raise HTTPException(
            status_code=500,
            detail="LIVEKIT_API_SECRET must be set in environment (.env or system env)"
        )

    # Check if SDK is available
    if not (AccessToken and VideoGrant):
        logger.error(IMPORT_ERROR_MSG)
        raise HTTPException(
            status_code=500,
            detail=IMPORT_ERROR_MSG
        )

    try:
        # Create JWT token
        at = AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
        grant = VideoGrant(room=req.room, can_publish=True, can_subscribe=True)
        at.add_grant(grant)
        at.set_identity(req.identity)
        token = at.to_jwt()
        
        logger.info(f"✓ Token issued for identity={req.identity}, room={req.room}")
        return {"token": token, "wsUrl": LIVEKIT_URL}
    except Exception as e:
        logger.error(f"Failed to generate token: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Token generation failed: {str(e)}"
        )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LiveKit Token Server")
    print("="*60)
    if IMPORT_ERROR_MSG:
        print(f"⚠ WARNING: {IMPORT_ERROR_MSG}")
        print("  The server will start but /get_token will fail.")
    else:
        print("✓ LiveKit SDK imported successfully.")
    print(f"✓ LIVEKIT_URL: {LIVEKIT_URL}")
    print(f"✓ LIVEKIT_API_KEY: {'(set)' if LIVEKIT_API_KEY else '(NOT SET)'}")
    print(f"✓ LIVEKIT_API_SECRET: {'(set)' if LIVEKIT_API_SECRET else '(NOT SET)'}")
    print("\nRun with:")
    print("  uvicorn token_server:app --host 0.0.0.0 --port 8000")
    print("\nTest with:")
    print("  curl http://localhost:8000/healthz")
    print("  curl -X POST http://localhost:8000/api/get_token \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"room\": \"demo-room\", \"identity\": \"test1\"}'")
    print("="*60 + "\n")
