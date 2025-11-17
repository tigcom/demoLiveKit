#!/usr/bin/env python3
"""
Test script to verify track.recv() works correctly
"""
import os
import asyncio
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
ROOM = os.getenv("ROOM", "demo-room")

async def test_recv():
    from livekit import rtc
    from livekit.api.access_token import AccessToken, VideoGrants
    
    # Generate token
    at = AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
    vg = VideoGrants(room_join=True, room=ROOM, can_publish=True, can_subscribe=True, can_publish_data=True)
    at = at.with_identity("test-agent")
    setattr(at, "video_grants", vg)
    token = at.to_jwt()
    
    # Connect
    room = rtc.Room()
    
    tracks_received = []
    
    def on_track_subscribed(track, publication, participant):
        logger.info(f"📢 Track subscribed: {participant.identity}, kind={track.kind}")
        tracks_received.append(track)
    
    def on_track_published(publication, participant):
        logger.info(f"🎵 Track published: {participant.identity}, kind={publication.kind}")
        logger.info(f"   Calling set_subscribed(True)...")
        publication.set_subscribed(True)
    
    room.on("track_subscribed", on_track_subscribed)
    room.on("track_published", on_track_published)
    
    await room.connect(LIVEKIT_URL, token)
    logger.info(f"✅ Connected to {ROOM}")
    
    # Wait for tracks
    logger.info("⏳ Waiting for tracks... (join from frontend now)")
    await asyncio.sleep(10)
    
    if not tracks_received:
        logger.warning("❌ No tracks received!")
        return
    
    logger.info(f"✅ Received {len(tracks_received)} track(s)")
    
    # Try to receive frames
    for track in tracks_received:
        logger.info(f"\n🔊 Testing track.recv() for {track.kind}...")
        try:
            for i in range(10):
                logger.info(f"   Attempt {i+1}/10...")
                frame = await track.recv()
                if frame:
                    data = getattr(frame, "data", b"")
                    logger.info(f"   ✅ Frame received! len={len(data)} bytes")
                else:
                    logger.info(f"   ⚠️ Frame is None")
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"   ❌ Error: {e}", exc_info=True)
    
    await room.disconnect()

if __name__ == "__main__":
    asyncio.run(test_recv())
