#!/usr/bin/env python3
"""
Quick script to verify token permissions
"""
import os
from dotenv import load_dotenv

load_dotenv()

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
ROOM = os.getenv("ROOM", "demo-room")
AGENT_ID = os.getenv("AGENT_ID", "agent-1")

try:
    from livekit.api.access_token import AccessToken, VideoGrants
except ImportError:
    from livekit.api import AccessToken, VideoGrants

def test_token():
    print(f"🔑 Testing token generation for agent: {AGENT_ID}")
    print(f"   Room: {ROOM}")
    
    at = AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
    vg = VideoGrants(
        room_join=True,
        room=ROOM,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True
    )
    
    if hasattr(at, "with_identity"):
        at = at.with_identity(AGENT_ID)
    
    setattr(at, "video_grants", vg)
    token = at.to_jwt()
    
    print(f"\n✅ Token generated successfully!")
    print(f"   can_publish: {vg.can_publish}")
    print(f"   can_subscribe: {vg.can_subscribe}")
    print(f"   can_publish_data: {vg.can_publish_data}")
    print(f"\n🎫 Token (first 50 chars): {token[:50]}...")
    
    return token

if __name__ == "__main__":
    test_token()
