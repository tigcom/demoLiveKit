#!/usr/bin/env python3
"""
Minimal test to verify track.recv() works
"""
import os
import asyncio
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
ROOM = os.getenv("ROOM", "demo-room")

async def main():
    from livekit import rtc
    from livekit.api.access_token import AccessToken, VideoGrants
    
    logger.info("🔧 Generating token...")
    at = AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
    vg = VideoGrants(room_join=True, room=ROOM, can_publish=True, can_subscribe=True, can_publish_data=True)
    at = at.with_identity("minimal-agent")
    setattr(at, "video_grants", vg)
    token = at.to_jwt()
    
    logger.info("🔌 Connecting to room...")
    room = rtc.Room()
    
    received_tracks = []
    
    def on_track_published(publication, participant):
        logger.info(f"🎵 Track published: {participant.identity}, kind={publication.kind}, sid={publication.sid}")
        logger.info(f"   Setting subscribed=True...")
        publication.set_subscribed(True)
    
    def on_track_subscribed(track, publication, participant):
        logger.info(f"📢 Track subscribed: {participant.identity}, kind={track.kind}")
        received_tracks.append((track, participant))
        
        # Start receiving in background
        asyncio.create_task(receive_frames(track, participant))
    
    async def receive_frames(track, participant):
        logger.info(f"🎧 Starting frame receiver for {participant.identity}")
        frame_count = 0
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                    if frame:
                        data = getattr(frame, "data", b"")
                        frame_count += 1
                        if frame_count % 50 == 0:  # Log every 50 frames
                            logger.info(f"  ▶ Received {frame_count} frames from {participant.identity}, last frame: {len(data)} bytes")
                except asyncio.TimeoutError:
                    if frame_count > 0:
                        logger.debug(f"  ⏱ Timeout (received {frame_count} frames so far)")
                    continue
                except Exception as e:
                    logger.error(f"  ❌ Error: {e}")
                    break
        except Exception as e:
            logger.error(f"❌ Frame receiver crashed: {e}", exc_info=True)
    
    room.on("track_published", on_track_published)
    room.on("track_subscribed", on_track_subscribed)
    
    await room.connect(LIVEKIT_URL, token)
    logger.info(f"✅ Connected to {ROOM}")
    
    logger.info("\n" + "="*60)
    logger.info("⏳ Waiting for participants to join...")
    logger.info("   1. Open frontend: http://localhost:5173")
    logger.info("   2. Click 'Join & Publish Mic'")
    logger.info("   3. Speak into your microphone")
    logger.info("="*60 + "\n")
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down...")
        await room.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
