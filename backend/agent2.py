"""Simple LiveKit agent for learning/demo purposes.

Behavior:
 - Connects to LiveKit room as a participant
 - Subscribes to audio tracks
 - Logs when audio frames are received (length and basic attrs)

Usage:
  - Set environment variables in .env or shell: LIVEKIT_URL, TOKEN_SERVER_URL, ROOM (optional), AGENT_ID
  - Run: python agent2.py --livekit

This file is intentionally minimal for learning; it does not perform STT/LLM/TTS.
"""
import os
import sys
import asyncio
import logging
import argparse
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "https://cloud.livekit.io")
TOKEN_SERVER_URL = os.getenv("TOKEN_SERVER_URL", "http://localhost:8000")
ROOM = os.getenv("ROOM", "demo-room")
AGENT_ID = os.getenv("AGENT_ID", "agent-2")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("agent2")


async def fetch_token(room: str, identity: str):
    """Fetch token from token server; return token string or None."""
    try:
        import requests
        resp = requests.post(f"{TOKEN_SERVER_URL}/api/get_token", json={"room": room, "identity": identity}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("token")
        logger.warning(f"Token server returned {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.warning(f"Could not fetch token from server: {e}")
    return None


def generate_token_directly():
    """Attempt generate a token using local LiveKit SDK if available (best-effort).
    Returns token string or None.
    """
    try:
        from livekit.api.access_token import AccessToken, VideoGrants
    except Exception:
        try:
            from livekit.api import AccessToken, VideoGrants
        except Exception:
            logger.debug("livekit.api not available for direct token generation")
            return None
    try:
        at = AccessToken(api_key=os.getenv("LIVEKIT_API_KEY"), api_secret=os.getenv("LIVEKIT_API_SECRET"))
        vg = VideoGrants(room_join=True, room=ROOM, can_publish=True, can_subscribe=True)
        if hasattr(at, "with_identity"):
            at = at.with_identity(AGENT_ID)
        setattr(at, "video_grants", vg)
        token = at.to_jwt()
        logger.info(f"Generated token (can_publish={vg.can_publish}, can_subscribe={vg.can_subscribe})")
        return token
    except Exception as e:
        logger.warning(f"Could not generate token directly: {e}")
        return None


class SimpleAgent:
    def __init__(self, room: str, identity: str, token: str):
        self.room_name = room
        self.identity = identity
        self.token = token
        self.room = None
        self.loop = None
        self.is_running = False

    async def connect(self):
        try:
            from livekit import rtc
        except Exception:
            logger.error("livekit rtc SDK not installed. Install 'livekit' python package to connect.")
            return

        self.room = rtc.Room()
        self.loop = asyncio.get_running_loop()

        # register handlers
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("track_subscribed", self._on_track_subscribed)

        logger.info(f"Connecting to {LIVEKIT_URL}...")
        await self.room.connect(LIVEKIT_URL, self.token)
        self.is_running = True
        logger.info(f"Connected to room {self.room_name}")

    def _on_participant_connected(self, participant):
        logger.info(f"Participant connected: {getattr(participant,'identity', participant)}")
        # show published tracks if available
        try:
            pub = getattr(participant, "published_tracks", None)
            tracks = getattr(participant, "tracks", None)
            logger.info(f"  published_tracks={len(pub) if pub is not None else 'n/a'}, tracks={len(tracks) if tracks is not None else 'n/a'}")
        except Exception:
            pass

    def _on_track_subscribed(self, track, publication, participant):
        logger.info(f"Track subscribed: kind={getattr(track,'kind','?')} from {getattr(participant,'identity','?')}")
        if getattr(track, "kind", None) != "audio":
            logger.info("Not an audio track, ignoring")
            return
        # start background receiver
        threading.Thread(target=self._recv_thread, args=(track, participant), daemon=True).start()

    def _recv_thread(self, track, participant):
        """Runs in a background thread; receives frames via run_coroutine_threadsafe and logs sizes."""
        if not self.loop:
            logger.error("No event loop available for receiving frames")
            return

        participant_id = getattr(participant, "identity", "unknown")
        logger.info(f"Starting recv loop for {participant_id}")
        while True:
            try:
                fut = asyncio.run_coroutine_threadsafe(track.recv(), self.loop)
                frame = fut.result(timeout=15)
                if frame is None:
                    logger.debug("recv returned None")
                    continue
                data = getattr(frame, "data", None)
                ln = len(data) if data else 0
                logger.info(f"Received audio frame from {participant_id}: {ln} bytes")
            except Exception as e:
                msg = str(e).lower()
                if "timeout" in msg:
                    # continue waiting
                    continue
                logger.warning(f"Receiver loop error for {participant_id}: {e}")
                break


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--livekit", action="store_true")
    parser.add_argument("--room", default=ROOM)
    parser.add_argument("--identity", default=AGENT_ID)
    args = parser.parse_args()

    token = await fetch_token(args.room, args.identity) or generate_token_directly()
    if not token:
        logger.error("Could not obtain token; set TOKEN_SERVER_URL or LIVEKIT_API_KEY/SECRET")
        return

    agent = SimpleAgent(args.room, args.identity, token)
    await agent.connect()

    try:
        while agent.is_running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Agent stopping")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
