# agent_greeting.py

import os
import sys
import asyncio
import logging
import argparse
import tempfile
import shutil
import uuid
import json
import struct
import threading
from pathlib import Path
from collections import deque
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "https://cloud.livekit.io")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
ROOM = os.getenv("ROOM", "demo-room")
AGENT_ID = os.getenv("AGENT_ID", "agent-1")
MODEL_NAME_LLM = os.getenv("MODEL_NAME_LLM", "gpt2")
DEVICE = os.getenv("DEVICE", "auto")
TOKEN_SERVER_URL = os.getenv("TOKEN_SERVER_URL", "http://localhost:8000")

_stt_model = None
_tts = None
_llm_pipe = None

# -------------------- Lazy-load models --------------------
def get_stt_model():
    global _stt_model
    if _stt_model is None:
        from faster_whisper import WhisperModel
        logger.info("Loading faster-whisper model...")
        _stt_model = WhisperModel("tiny")
    return _stt_model

def get_tts():
    global _tts
    if _tts is None:
        from TTS.api import TTS
        logger.info("Loading Coqui TTS model...")
        _tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)
    return _tts

def get_llm_pipeline():
    global _llm_pipe
    if _llm_pipe is None:
        from transformers import pipeline
        import torch
        use_cuda = (DEVICE.lower() == "cuda" and torch.cuda.is_available()) or (DEVICE.lower()=="auto" and torch.cuda.is_available())
        device = 0 if use_cuda else -1
        logger.info(f"Loading transformers pipeline (model={MODEL_NAME_LLM}) on {'cuda' if use_cuda else 'cpu'}")
        _llm_pipe = pipeline("text-generation", model=MODEL_NAME_LLM, device=device, trust_remote_code=True)
    return _llm_pipe

# -------------------- Core pipeline --------------------
def transcribe_wav(wav_path: str) -> str:
    stt = get_stt_model()
    segments, _ = stt.transcribe(wav_path)
    return " ".join([s.text.strip() for s in segments]).strip()

def query_llm(text: str) -> str:
    pipe = get_llm_pipeline()
    out = pipe(text, max_new_tokens=256, do_sample=False, temperature=0.0)
    if out and "generated_text" in out[0]:
        generated = out[0]["generated_text"]
        return generated[len(text):].strip() if generated.startswith(text) else generated
    return "Sorry, I couldn't understand."

def synthesize_tts(text: str, out_wav_path: str) -> str:
    tts = get_tts()
    tmp_wav = tempfile.mktemp(suffix=".wav")
    tts.tts_to_file(text=text, file_path=tmp_wav)
    shutil.copy(tmp_wav, out_wav_path)
    os.remove(tmp_wav)
    return out_wav_path

# -------------------- Helpers for audio frames --------------------
def frames_to_int16_bytes(frames: list, debug: bool = False) -> bytes:
    """
    Convert list of frame objects (each has .data bytes) into PCM16 bytes.
    Try these strategies in order:
      1) assume data is little-endian int16 PCM
      2) assume data is float32 array -> convert to int16
      3) try chunk-by-chunk fallback
    Return b'' if cannot convert.
    """
    if not frames:
        if debug: logger.debug("frames_to_int16_bytes: empty frames")
        return b""

    # concatenate bytes present
    parts = []
    total_len = 0
    for f in frames:
        d = getattr(f, "data", None)
        if d:
            parts.append(d)
            total_len += len(d)
    combined = b"".join(parts)
    if not combined:
        if debug: logger.debug("frames_to_int16_bytes: combined empty")
        return b""

    # 1) Likely int16: length divisible by 2 and appears noisy (not divisible cleanly by 4 OR heuristic)
    try:
        if len(combined) % 2 == 0:
            # Try interpret as int16 array
            count = len(combined) // 2
            # Quick sanity: count not gigantic
            shorts = struct.unpack(f"<{count}h", combined)
            # quick RMS check: if samples are within int16 range it's ok
            # We'll accept this as PCM16 if not all zeros
            if any(s != 0 for s in shorts):
                if debug: logger.debug(f"frames_to_int16_bytes: interpreted as int16 (count={count})")
                return combined
    except Exception as e:
        if debug: logger.debug(f"frames_to_int16_bytes: int16 attempt failed: {e}")

    # 2) Try float32 -> convert to int16
    try:
        if len(combined) % 4 == 0:
            countf = len(combined) // 4
            floats = struct.unpack(f"<{countf}f", combined)
            int16_list = []
            for x in floats:
                # clip and convert
                v = int(round(x * 32767.0))
                if v > 32767: v = 32767
                if v < -32768: v = -32768
                int16_list.append(v)
            packed = struct.pack(f"<{len(int16_list)}h", *int16_list)
            if debug: logger.debug(f"frames_to_int16_bytes: converted float32->int16 (count={len(int16_list)})")
            return packed
    except Exception as e:
        if debug: logger.debug(f"frames_to_int16_bytes: float32 attempt failed: {e}")

    # 3) Fallback: try chunk by chunk interpreting each chunk as int16 or float32
    out_parts = []
    offset = 0
    while offset < len(combined):
        remaining = len(combined) - offset
        # try small int16 chunk
        if remaining >= 2:
            # attempt to unpack next N bytes as int16 (choose a safe chunk)
            chunk_len = (remaining // 2) * 2
            try:
                count = chunk_len // 2
                shorts = struct.unpack(f"<{count}h", combined[offset:offset+chunk_len])
                out_parts.append(struct.pack(f"<{count}h", *shorts))
                offset += chunk_len
                continue
            except Exception:
                # try float32 fallback for remainder
                break
        else:
            break

    if out_parts:
        return b"".join(out_parts)

    # give up
    if debug: logger.debug("frames_to_int16_bytes: giving up, returning empty")
    return b""

def is_frame_silent(frame, rms_threshold: float = 500.0, debug: bool = False) -> bool:
    """
    Compute RMS on frame.data and return True if below threshold.
    Robust to int16 and float32 encodings.
    """
    data = getattr(frame, "data", None)
    if not data:
        if debug: logger.debug("is_frame_silent: no data -> silent")
        return True
    try:
        # try int16
        if len(data) % 2 == 0:
            count = len(data) // 2
            samples = struct.unpack(f"<{count}h", data)
        else:
            # try float32
            if len(data) % 4 == 0:
                count = len(data) // 4
                floats = struct.unpack(f"<{count}f", data)
                samples = [int(max(-32768, min(32767, int(round(x * 32767.0))))) for x in floats]
            else:
                # unknown format => conservative: not silent
                if debug: logger.debug("is_frame_silent: unknown frame length -> not silent")
                return False

        # compute RMS
        ssum = 0.0
        for s in samples:
            ssum += float(s) * float(s)
        rms = (ssum / max(1, len(samples))) ** 0.5
        if debug: logger.debug(f"is_frame_silent: rms={rms}, threshold={rms_threshold}")
        return rms < rms_threshold
    except Exception as e:
        if debug: logger.debug(f"is_frame_silent: exception {e} -> treat as not silent")
        return False

# -------------------- LiveKit agent with greeting and audio processing --------------------
class LiveKitAgent:
    def __init__(self, room_name: str, agent_id: str, token: str):
        self.room_name = room_name
        self.agent_id = agent_id
        self.token = token
        self.room = None
        self.is_running = False
        self.loop = None
        self.greeted_participants = set()
        self._participant_tracks = {}  # {participant_id: {track_sid: track}}
        logger.info(f"Initialized LiveKitAgent(room={room_name}, id={agent_id})")

    async def connect(self):
        from livekit import rtc
        self.room = rtc.Room()
        self.loop = asyncio.get_running_loop()
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("track_published", self._on_track_published)
        self.room.on("track_subscribed", self._on_track_subscribed)
        await self.room.connect(LIVEKIT_URL, self.token)
        self.is_running = True
        logger.info(f"Connected to room {self.room_name}")

        # Start global audio receiver thread
        threading.Thread(target=self._audio_receiver_thread, daemon=True).start()

        # Greet existing participants AND subscribe to their existing tracks
        participants = getattr(self.room, "participants", {})
        for p in participants.values():
            identity = getattr(p, "identity", "unknown")
            if identity not in self.greeted_participants:
                self.greeted_participants.add(identity)
                asyncio.create_task(self._send_greeting_message(identity))
            
            # Subscribe to existing tracks
            track_pubs = getattr(p, "track_publications", {})
            for sid, pub in track_pubs.items():
                if pub.subscribed:
                    continue
                logger.info(f"🔔 Subscribing to existing track {sid} from {identity}")
                pub.set_subscribed(True)

    async def disconnect(self):
        self.is_running = False
        if self.room:
            await self.room.disconnect()
            logger.info("Disconnected")

    def _on_participant_connected(self, participant):
        identity = getattr(participant, "identity", "unknown")
        logger.info(f"👤 Participant connected: {identity}")
        if identity not in self.greeted_participants:
            self.greeted_participants.add(identity)
            asyncio.create_task(self._send_greeting_message(identity))

    def _on_track_published(self, publication, participant):
        """Called when a participant publishes a new track"""
        identity = getattr(participant, "identity", "unknown")
        kind = getattr(publication, "kind", "unknown")
        logger.info(f"🎵 Track published by {identity}: kind={kind}, sid={publication.sid}")
        
        # Auto-subscribe to audio tracks
        # LiveKit SDK returns: kind=1 for audio, kind=2 for video
        kind_str = str(kind).lower()
        is_audio = (kind == 1 or kind_str == "audio" or kind_str == "1")
        
        if is_audio:
            logger.info(f"🔔 Auto-subscribing to audio track from {identity}")
            publication.set_subscribed(True)
        else:
            logger.info(f"⏭️ Skipping non-audio track: {kind}")

    async def _send_greeting_message(self, user_identity: str):
        greeting_text = f"Hello {user_identity}. Welcome to the room."
        tmp_wav = tempfile.mktemp(suffix="_greeting.wav")
        synthesize_tts(greeting_text, tmp_wav)
        try:
            here = Path(__file__).parent
            static_folder = here / "static" / "greetings"
            static_folder.mkdir(parents=True, exist_ok=True)
            filename = f"greeting_{user_identity}_{uuid.uuid4().hex[:8]}.wav"
            dst = static_folder / filename
            shutil.copy(tmp_wav, dst)
            base_url = TOKEN_SERVER_URL.rstrip("/")
            url = f"{base_url}/static/greetings/{filename}"
            payload = {"type":"greeting","text":greeting_text,"to":user_identity,"url":url}
            await self._publish_data(payload)
            logger.info(f"Published greeting for {user_identity} → {url}")
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

    async def _publish_data(self, payload: dict, reliable: bool = True):
        if not self.room:
            logger.debug("_publish_data: no room")
            return
        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        lp = getattr(self.room, "local_participant", None)
        if lp is None:
            logger.debug("_publish_data: no local_participant")
            return
        try:
            res = lp.publish_data(data_bytes, reliable=reliable)
            # if coroutine returned (depends on SDK), await it
            if asyncio.iscoroutine(res):
                await res
            logger.info(f"📨 Published data: {payload.get('type')}")
        except TypeError:
            # fallback: maybe publish_data signature is different
            try:
                await lp.publish_data(data_bytes)
                logger.info(f"📨 Published data fallback: {payload.get('type')}")
            except Exception as e:
                logger.error(f"Failed publish_data fallback: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to publish data: {e}", exc_info=True)

    # Thread-safe publish helper for background threads
    def _publish_data_sync(self, payload: dict, reliable: bool = True):
        if not self.room:
            logger.debug("_publish_data_sync: no room")
            return
        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        lp = getattr(self.room, "local_participant", None)
        if lp is None:
            logger.debug("_publish_data_sync: no local_participant")
            return

        def _call():
            try:
                res = lp.publish_data(data_bytes, reliable=reliable)
                if asyncio.iscoroutine(res):
                    # schedule awaiting in main loop
                    asyncio.run_coroutine_threadsafe(res, self.loop)
                logger.debug("    ✅ Published data (sync)")
            except Exception as e:
                logger.debug(f"    ⚠️ publish_data sync failed: {e}")

        try:
            # prefer scheduling coroutine in event loop
            asyncio.run_coroutine_threadsafe(self._publish_data(payload, reliable=reliable), self.loop)
        except Exception:
            # fallback to thread runner
            threading.Thread(target=_call, daemon=True).start()

    # ---------------- audio receiving ----------------
    def _on_track_subscribed(self, track, publication, participant):
        pid = participant.identity
        if pid not in self._participant_tracks:
            self._participant_tracks[pid] = {}
        self._participant_tracks[pid][publication.sid] = track
        logger.info(f"📢 Track subscribed from {pid}, kind={getattr(track,'kind','unknown')}")
        
        # Start async frame receiver for this track
        asyncio.run_coroutine_threadsafe(
            self._receive_frames_async(track, pid),
            self.loop
        )

    async def _receive_frames_async(self, track, participant_id: str):
        """
        Async method to receive frames from a track using recv() in a loop.
        """
        logger.info(f"🎧 Starting async frame receiver for {participant_id}")
        
        from collections import deque
        buffer = deque(maxlen=2400)
        silence_counter = 0
        has_speech = False
        
        try:
            while self.is_running:
                try:
                    # Use recv() with proper await
                    frame = await track.recv()
                    
                    if frame is None:
                        await asyncio.sleep(0.01)
                        continue
                    
                    fd = getattr(frame, "data", b"")
                    if fd:
                        rms = (sum((int.from_bytes(fd[i:i+2], 'little', signed=True)**2
                                    for i in range(0, len(fd), 2))) / max(1, len(fd)//2)) ** 0.5
                        logger.info(f"  ▶ Frame received from {participant_id}: len={len(fd)} bytes, rms={rms:.2f}")
                    else:
                        logger.info(f"  ▶ Frame received from {participant_id}: empty data")
                    
                    buffer.append(frame)
                    silent = is_frame_silent(frame, rms_threshold=600.0)
                    
                    if not silent:
                        has_speech = True
                        silence_counter = 0
                    elif has_speech:
                        silence_counter += 1
                    
                    if has_speech and silence_counter >= 10:
                        frames_copy = list(buffer)
                        buffer.clear()
                        has_speech = False
                        silence_counter = 0
                        # Process in background thread
                        threading.Thread(
                            target=self._process_audio_buffer,
                            args=(frames_copy, participant_id),
                            daemon=True
                        ).start()
                
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.01)
                    continue
                except Exception as e:
                    logger.debug(f"Frame recv error: {e}")
                    await asyncio.sleep(0.01)
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error in async frame receiver for {participant_id}: {e}", exc_info=True)

    def _audio_receiver_thread(self):
        import time
        from collections import deque
        logger.info("🔊 Audio receiver thread started (polling all participants)")

        participant_buffers = {}  # {participant_id: {"frames": deque, "silence_counter": int, "has_speech": bool}}
        last_log_time = 0

        while self.is_running:
            participants = getattr(self.room, "participants", {})
            
            # Log participant count every 5 seconds
            current_time = time.time()
            if current_time - last_log_time > 5:
                logger.info(f"🔍 Polling {len(participants)} participants, {len(self._participant_tracks)} have tracks")
                for pid, tracks in self._participant_tracks.items():
                    logger.info(f"  → {pid}: {len(tracks)} track(s)")
                last_log_time = current_time
            
            for p in participants.values():
                pid = p.identity
                tracks = self._participant_tracks.get(pid, {})
                if not tracks:
                    continue
                if pid not in participant_buffers:
                    participant_buffers[pid] = {"frames": deque(maxlen=2400),
                                                "silence_counter": 0,
                                                "has_speech": False}
                    logger.info(f"📦 Created buffer for participant {pid}")

                buf = participant_buffers[pid]

                for sid, track in tracks.items():
                    try:
                        fut = asyncio.run_coroutine_threadsafe(track.recv(), self.loop)
                        frame = fut.result(timeout=0.5)
                        if frame is None:
                            continue

                        fd = getattr(frame, "data", b"")
                        if fd:
                            rms = (sum((int.from_bytes(fd[i:i+2], 'little', signed=True)**2
                                        for i in range(0, len(fd), 2))) / max(1, len(fd)//2)) ** 0.5
                            logger.info(f"  ▶ Frame received from {pid}: len={len(fd)} bytes, rms={rms:.2f}")
                        else:
                            logger.info(f"  ▶ Frame received from {pid}: empty data")

                        buf["frames"].append(frame)
                        silent = is_frame_silent(frame, rms_threshold=600.0)
                        if not silent:
                            buf["has_speech"] = True
                            buf["silence_counter"] = 0
                        elif buf["has_speech"]:
                            buf["silence_counter"] += 1

                        if buf["has_speech"] and buf["silence_counter"] >= 10:
                            frames_copy = list(buf["frames"])
                            buf["frames"].clear()
                            buf["has_speech"] = False
                            buf["silence_counter"] = 0
                            threading.Thread(target=self._process_audio_buffer, args=(frames_copy, pid), daemon=True).start()
                    except TimeoutError:
                        # Normal - no frame available yet
                        continue
                    except Exception as e:
                        logger.error(f"❌ Error receiving frame from {pid}: {e}", exc_info=True)
                        continue
            time.sleep(0.05)
    def _process_audio_buffer(self, frames: list, participant_id: str):
        """
        Convert frames -> WAV -> STT -> LLM -> TTS -> publish URL back to clients.
        Runs in background thread.
        """
        if not frames:
            logger.warning("No frames to process")
            return

        try:
            pcm_bytes = frames_to_int16_bytes(frames, debug=True)
            if not pcm_bytes:
                logger.warning("Converted PCM empty - skip")
                return

            tmp_wav_in = tempfile.mktemp(suffix="_input.wav")
            tmp_wav_out = tempfile.mktemp(suffix="_reply.wav")
            import wave
            with wave.open(tmp_wav_in, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm_bytes)

            # STT
            text = transcribe_wav(tmp_wav_in)
            if not text.strip():
                logger.info(f"No speech detected for {participant_id}")
                os.remove(tmp_wav_in)
                return

            self._publish_data_sync({"type":"transcript","from":participant_id,"text":text})

            # LLM
            reply_text = query_llm(text)

            # TTS
            synthesize_tts(reply_text, tmp_wav_out)

            # Copy to static folder & publish URL
            here = Path(__file__).parent
            static_folder = here / "static" / "greetings"
            static_folder.mkdir(parents=True, exist_ok=True)
            filename = f"reply_{uuid.uuid4().hex[:8]}.wav"
            dst = static_folder / filename
            shutil.copy(tmp_wav_out, dst)
            url = f"{TOKEN_SERVER_URL.rstrip('/')}/static/greetings/{filename}"
            self._publish_data_sync({"type":"greeting","url":url,"from":"agent","to":participant_id,"text":reply_text})

        finally:
            for f in [tmp_wav_in, tmp_wav_out]:
                if os.path.exists(f):
                    os.remove(f)

# -------------------- Token helpers --------------------
async def fetch_token_from_server(room: str, identity: str):
    import requests
    resp = requests.post(f"{TOKEN_SERVER_URL}/api/get_token", json={"room": room, "identity": identity}, timeout=5)
    if resp.status_code == 200:
        return resp.json().get("token")
    return None

def generate_token_directly():
    try:
        from livekit.api.access_token import AccessToken, VideoGrants
    except ImportError:
        from livekit.api import AccessToken, VideoGrants
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
    logger.info(f"✅ Token generated: can_subscribe={vg.can_subscribe}, can_publish={vg.can_publish}, can_publish_data={vg.can_publish_data}")
    return token

# -------------------- Runner --------------------
async def run_livekit_agent():
    token = await fetch_token_from_server(ROOM, AGENT_ID) or generate_token_directly()
    if not token:
        logger.error("Could not obtain token")
        return
    agent = LiveKitAgent(ROOM, AGENT_ID, token)
    await agent.connect()
    try:
        while agent.is_running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await agent.disconnect()

async def main():
    parser = argparse.ArgumentParser(description="LiveKit voice agent")
    parser.add_argument("--livekit", action="store_true")
    parser.add_argument("--local-debug", action="store_true")
    parser.add_argument("--sample", type=str, help="Input WAV")
    parser.add_argument("--output", type=str, help="Output WAV")
    args = parser.parse_args()

    if args.local_debug:
        logger.info("Local debug mode is not using livekit.")
        # you can call process_audio_file manually if needed
    elif args.livekit:
        await run_livekit_agent()
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
