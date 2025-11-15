"""agent.py

LiveKit Voice Assistant Agent

Behavior:
  - Connects to a LiveKit room as a participant (agent-{id}).
  - Subscribes to audio tracks from other participants.
  - For each user audio:
    1. Buffers incoming audio frames.
    2. Transcribes to text (faster-whisper).
    3. Queries LLM (Hugging Face Transformers locally via PyTorch).
    4. Synthesizes reply (Coqui TTS).
    5. Publishes reply audio as a continuous track.

Flows:
  - WebRTC: Full connection to LiveKit with audio I/O (recommended).
  - Local debug: Test STT/LLM/TTS pipeline without LiveKit.

Environment:
  LIVEKIT_URL           (e.g., https://cloud.livekit.io)
  LIVEKIT_API_KEY       (for generating token)
  LIVEKIT_API_SECRET    (for generating token)
  ROOM                  (room name to join, default: demo-room)
  AGENT_ID              (identity for agent participant, default: agent-1)
    MODEL_NAME_LLM        (Hugging Face model id, default: gpt2)
    DEVICE                ('auto'|'cuda'|'cpu') device selection for PyTorch
  TOKEN_SERVER_URL      (default: http://localhost:8000)

CLI Usage:
  python agent.py --livekit         # Connect to LiveKit room
  python agent.py --local-debug     # Test STT/LLM/TTS without LiveKit
"""
import os
import sys
import asyncio
import logging
import argparse
import tempfile
import threading
import queue
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Environment
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "https://cloud.livekit.io")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
ROOM = os.getenv("ROOM", "demo-room")
AGENT_ID = os.getenv("AGENT_ID", "agent-1")
MODEL_NAME_LLM = os.getenv("MODEL_NAME_LLM", "gpt2")  # change to a chat-capable HF model if you want
DEVICE = os.getenv("DEVICE", "auto")  # 'cuda', 'cpu', or 'auto'
TOKEN_SERVER_URL = os.getenv("TOKEN_SERVER_URL", "http://localhost:8000")

# Lazy-loaded models
_stt_model = None
_tts = None
_llm_pipe = None


def get_stt_model():
    """Lazy load faster-whisper model."""
    global _stt_model
    if _stt_model is None:
        try:
            from faster_whisper import WhisperModel
            logger.info("Loading faster-whisper model (tiny)...")
            _stt_model = WhisperModel("tiny")
            logger.info("✓ faster-whisper model loaded")
        except ImportError:
            logger.error("faster-whisper not installed. Install: pip install faster-whisper")
            raise
    return _stt_model


def get_tts():
    """Lazy load Coqui TTS."""
    global _tts
    if _tts is None:
        try:
            from TTS.api import TTS
            logger.info("Loading Coqui TTS model...")
            _tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)
            logger.info("✓ Coqui TTS model loaded")
        except ImportError:
            logger.error("TTS (Coqui) not installed. Install: pip install TTS")
            raise
    return _tts


def transcribe_wav(wav_path: str) -> str:
    """
    Transcribe audio file to text using faster-whisper.
    
    Args:
        wav_path: path to WAV file (should be 48k 16-bit mono)
    
    Returns:
        Transcribed text (empty string if no speech detected)
    """
    try:
        stt = get_stt_model()
        logger.info(f"Transcribing: {wav_path}")
        segments, _ = stt.transcribe(wav_path)
        text = " ".join([s.text.strip() for s in segments]).strip()
        logger.info(f"✓ Transcribed: {text[:100]}")
        return text
    except Exception as e:
        logger.error(f"STT failed: {e}", exc_info=True)
        raise


def query_llm(text: str) -> str:
    """
    Query LLM (using Hugging Face Transformers) with user text and get reply.

    This function lazy-loads a transformers "text-generation" pipeline the first
    time it is called. It will try to use CUDA if available and requested via
    the DEVICE environment variable. The default model is 'gpt2' but you should
    set `MODEL_NAME_LLM` in your .env to a larger/chat-capable model.

    Args:
        text: user message

    Returns:
        LLM reply text
    """
    try:
        pipe = get_llm_pipeline()
        logger.info(f"Querying LLM (model={MODEL_NAME_LLM}): {text[:80]}...")
        # Use the text-generation pipeline to generate a reply.
        # We keep generation deterministic by default (no sampling).
        out = pipe(
            text,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
        )
        # Pipeline returns list of dicts with 'generated_text'
        if isinstance(out, list) and len(out) > 0:
            generated = out[0].get("generated_text", "").strip()
            # If the model echoes the prompt, try to return only the new part
            if generated.startswith(text):
                reply = generated[len(text):].strip()
            else:
                reply = generated
        else:
            reply = "Sorry, I couldn't understand."

        logger.info(f"✓ LLM reply: {reply[:200]}")
        return reply
    except Exception as e:
        logger.error(f"LLM (transformers) query failed: {e}", exc_info=True)
        return "Sorry, I couldn't process that."


def get_llm_pipeline():
    """Lazy-load a Hugging Face text-generation pipeline with torch device handling.

    This will attempt to use CUDA if available and requested. To install a
    CUDA 12.8 compatible PyTorch build, follow instructions in the project
    README or run a command similar to:

      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    (Exact command may vary by torch version; the comment is here to guide
    users. If CUDA is not available, pipeline will use CPU.)
    """
    global _llm_pipe
    if _llm_pipe is not None:
        return _llm_pipe

    try:
        from transformers import pipeline
        import torch
    except ImportError:
        logger.error("transformers or torch not installed. Install: pip install transformers torch torchvision torchaudio")
        raise

    # Detect device
    use_cuda = False
    if DEVICE.lower() == "cuda":
        use_cuda = torch.cuda.is_available()
    elif DEVICE.lower() == "cpu":
        use_cuda = False
    else:  # auto
        use_cuda = torch.cuda.is_available()

    device = 0 if use_cuda else -1
    logger.info(f"Loading transformers pipeline (model={MODEL_NAME_LLM}) on {'cuda' if use_cuda else 'cpu'}")

    # Create pipeline
    _llm_pipe = pipeline(
        "text-generation",
        model=MODEL_NAME_LLM,
        device=device,
        trust_remote_code=True,
    )
    logger.info("✓ Transformers pipeline loaded")
    return _llm_pipe


def synthesize_tts(text: str, out_wav_path: str) -> str:
    """
    Synthesize text to speech using Coqui TTS and save to WAV.
    Ensures output is 48k 16-bit mono.
    
    Args:
        text: text to synthesize
        out_wav_path: output WAV file path
    
    Returns:
        Path to output WAV file (48k 16-bit mono)
    """
    try:
        from audio_utils import ensure_wav_48k16_mono
        tts = get_tts()
        logger.info(f"Synthesizing TTS: {text[:50]}...")
        
        # Write temporary WAV
        tmp_wav = tempfile.mktemp(suffix=".wav")
        tts.tts_to_file(text=text, file_path=tmp_wav)
        
        # Ensure output is 48k 16-bit mono
        ensure_wav_48k16_mono(tmp_wav, out_wav_path)
        logger.info(f"✓ TTS output saved to {out_wav_path}")
        
        # Clean temp file
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        
        return out_wav_path
    except Exception as e:
        logger.error(f"TTS failed: {e}", exc_info=True)
        raise


async def process_audio_file(in_wav_path: str, out_wav_path: str):
    """
    Full pipeline: STT -> LLM -> TTS.
    
    Args:
        in_wav_path: input WAV file
        out_wav_path: output WAV file
    """
    logger.info("=" * 60)
    logger.info("Audio processing pipeline")
    logger.info("=" * 60)
    
    # 1. Transcribe
    try:
        text = transcribe_wav(in_wav_path)
    except Exception as e:
        logger.error(f"STT step failed: {e}")
        return
    
    if not text:
        logger.warning("No speech detected in audio")
        return
    
    # 2. Query LLM
    reply_text = query_llm(text)
    
    # 3. Synthesize
    try:
        synthesize_tts(reply_text, out_wav_path)
    except Exception as e:
        logger.error(f"TTS step failed: {e}")
        return
    
    logger.info("=" * 60)
    logger.info("✓ Pipeline complete")
    logger.info("=" * 60)


# ============================================================================
# LiveKit Agent (WebRTC)
# ============================================================================

class LiveKitAgent:
    """
    Voice assistant agent that connects to LiveKit and processes audio.
    
    Workflow:
    1. Connect to room
    2. Subscribe to participant microphone tracks
    3. Buffer and process audio (STT -> LLM -> TTS)
    4. Publish reply audio as a track
    """
    
    def __init__(self, room_name: str, agent_id: str, token: str):
        self.room_name = room_name
        self.agent_id = agent_id
        self.token = token
        self.room = None
        self.is_running = False
        self.audio_buffer = queue.Queue()  # For buffering incoming audio
        self.process_thread = None
        
        logger.info(f"Initialized LiveKitAgent(room={room_name}, id={agent_id})")
    
    async def connect(self):
        """Connect to LiveKit room."""
        try:
            from livekit import rtc
            
            logger.info(f"Connecting to {LIVEKIT_URL}...")
            
            # Create room
            self.room = rtc.Room()
            
            # Set up event handlers
            self.room.on_participant_connected += self._on_participant_connected
            self.room.on_participant_disconnected += self._on_participant_disconnected
            self.room.on_track_subscribed += self._on_track_subscribed
            self.room.on_track_unsubscribed += self._on_track_unsubscribed
            
            # Connect to room
            await self.room.aconnect(LIVEKIT_URL, self.token)
            
            logger.info(f"✓ Connected to {self.room_name}")
            logger.info(f"✓ Local participant: {self.room.local_participant.identity}")
            
            self.is_running = True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}", exc_info=True)
            raise
    
    def _on_participant_connected(self, participant):
        """Callback when a participant joins the room."""
        logger.info(f"👤 Participant connected: {participant.identity}")
    
    def _on_participant_disconnected(self, participant_id: str):
        """Callback when a participant leaves the room."""
        logger.info(f"👤 Participant disconnected: {participant_id}")
    
    def _on_track_subscribed(self, track, publication, participant):
        """
        Callback when a remote audio track is subscribed.
        This is where we handle incoming audio from users.
        """
        logger.info(f"📢 Track subscribed: {publication.name or 'unknown'} from {participant.identity}")
        
        if track.kind == "audio":
            # Start listening to this audio track in a separate thread
            threading.Thread(
                target=self._audio_receiver_thread,
                args=(track, participant.identity),
                daemon=True
            ).start()
    
    def _on_track_unsubscribed(self, publication, participant):
        """Callback when a remote audio track is unsubscribed."""
        logger.info(f"📢 Track unsubscribed from {participant.identity}")
    
    def _audio_receiver_thread(self, audio_track, participant_id: str):
        """
        Thread that receives audio frames from a remote participant.
        Buffers audio and triggers processing.
        """
        logger.info(f"🎤 Starting audio receiver for {participant_id}")
        
        try:
            audio_buffer = []
            silence_count = 0
            max_silence_frames = 240  # ~5 seconds at 48kHz with 20ms frames
            
            while self.is_running:
                try:
                    # Get audio frame from track
                    frame = audio_track.recv()  # Blocks until frame available
                    
                    if frame is None:
                        silence_count += 1
                        if silence_count > max_silence_frames:
                            # Silence detected, process buffered audio
                            if audio_buffer:
                                logger.info(f"📍 Silence detected, processing {len(audio_buffer)} frames...")
                                self._process_audio_buffer(audio_buffer, participant_id)
                                audio_buffer = []
                            silence_count = 0
                        continue
                    
                    silence_count = 0
                    
                    # Frame is rtc.AudioFrame: has data, sample_rate, num_channels, samples_per_channel
                    audio_buffer.append(frame)
                    
                    # If buffer grows large enough, process
                    if len(audio_buffer) > 240:  # ~5 seconds
                        logger.info(f"📍 Buffer full, processing {len(audio_buffer)} frames...")
                        self._process_audio_buffer(audio_buffer, participant_id)
                        audio_buffer = []
                
                except Exception as e:
                    logger.error(f"Error receiving audio frame: {e}", exc_info=True)
                    break
        
        except Exception as e:
            logger.error(f"Audio receiver thread error: {e}", exc_info=True)
        finally:
            logger.info(f"🎤 Audio receiver for {participant_id} stopped")
    
    def _process_audio_buffer(self, frames: list, participant_id: str):
        """
        Process buffered audio frames.
        Converts to WAV -> STT -> LLM -> TTS -> publish.
        """
        logger.info(f"Processing audio from {participant_id} ({len(frames)} frames)...")
        
        try:
            # Combine frames into single audio data
            # Each frame should be 48kHz 16-bit mono
            import numpy as np
            import wave
            
            audio_data = b''
            for frame in frames:
                # frame.data is bytes of PCM16 audio
                audio_data += frame.data
            
            # Save to temporary WAV
            tmp_wav_in = tempfile.mktemp(suffix="_input.wav")
            tmp_wav_out = tempfile.mktemp(suffix="_reply.wav")
            
            # Write audio data as WAV (48kHz 16-bit mono)
            with wave.open(tmp_wav_in, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(48000)  # 48kHz
                wav_file.writeframes(audio_data)
            
            logger.info(f"📁 Saved input audio to {tmp_wav_in}")
            
            # ===== STT =====
            try:
                text = transcribe_wav(tmp_wav_in)
            except Exception as e:
                logger.error(f"STT failed: {e}")
                return
            
            if not text.strip():
                logger.warning("No speech detected")
                return
            
            logger.info(f"🗣️ User said: {text}")
            
            # ===== LLM =====
            reply_text = query_llm(text)
            logger.info(f"💬 Agent reply: {reply_text}")
            
            # ===== TTS =====
            try:
                synthesize_tts(reply_text, tmp_wav_out)
            except Exception as e:
                logger.error(f"TTS failed: {e}")
                return
            
            # ===== PUBLISH =====
            try:
                self._publish_audio(tmp_wav_out)
            except Exception as e:
                logger.error(f"Failed to publish audio: {e}")
            
            # Cleanup
            for tmp_file in [tmp_wav_in, tmp_wav_out]:
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"Audio processing failed: {e}", exc_info=True)
    
    def _publish_audio(self, wav_path: str):
        """
        Publish audio to the room as a track.
        Uses LiveKit's LocalAudioTrack.
        """
        logger.info(f"Publishing audio: {wav_path}")
        
        try:
            from livekit import rtc
            from audio_source import wav_to_frames
            
            # Load WAV and get frames
            frames = wav_to_frames(wav_path)
            
            if not frames:
                logger.warning("No audio frames to publish")
                return
            
            # Create audio source and track
            # Note: LiveKit's rtc.LocalAudioTrack can publish raw PCM16 frames
            # We create a simple publisher that pushes frames
            
            logger.info(f"Publishing {len(frames)} frames...")
            
            # Create a track source
            track = rtc.LocalAudioTrack.create_audio_track(
                "agent-reply",
                sample_rate=48000,
                num_channels=1,
                audio_source=self._create_frame_source(frames)
            )
            
            # Publish the track
            publication = self.room.local_participant.publish_track(track)
            
            logger.info(f"✓ Published audio track: {publication.name}")
            
        except Exception as e:
            logger.error(f"Publish failed: {e}", exc_info=True)
            raise
    
    def _create_frame_source(self, frames: list):
        """
        Create an audio source that pushes pre-recorded frames.
        This is a workaround for publishing static audio.
        """
        # For now, we'll use a simple approach: create a source that yields frames
        # In a real implementation, you'd subclass rtc.AudioSource
        
        class SimpleFrameSource:
            def __init__(self, frames_list):
                self.frames = frames_list
                self.index = 0
            
            async def get_frame(self):
                if self.index < len(self.frames):
                    frame_data = self.frames[self.index]
                    self.index += 1
                    # Return frame tuple: (data, sample_rate, num_channels, samples_per_channel)
                    return (frame_data, 48000, 1, 960)
                return None
        
        return SimpleFrameSource(frames)
    
    async def disconnect(self):
        """Disconnect from room and cleanup."""
        logger.info("Disconnecting from room...")
        self.is_running = False
        
        if self.room:
            try:
                await self.room.adisconnect()
                logger.info("✓ Disconnected")
            except Exception as e:
                logger.error(f"Disconnect error: {e}")


# ============================================================================
# Helper functions for token management
# ============================================================================

async def fetch_token_from_server(room: str, identity: str) -> Optional[str]:
    """Fetch token from token server."""
    try:
        import requests
        logger.info(f"Fetching token from {TOKEN_SERVER_URL}...")
        resp = requests.post(
            f"{TOKEN_SERVER_URL}/api/get_token",
            json={"room": room, "identity": identity},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            token = data.get("token")
            logger.info(f"✓ Got token")
            return token
        else:
            logger.warning(f"Token server returned {resp.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Could not fetch token from server: {e}")
        return None


def generate_token_directly() -> Optional[str]:
    """Generate token directly using LiveKit SDK."""
    try:
        from livekit.api import AccessToken, VideoGrant
        logger.info("Generating token directly...")
        at = AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
        grant = VideoGrant(room=ROOM, can_publish=True, can_subscribe=True)
        at.add_grant(grant)
        at.set_identity(AGENT_ID)
        token = at.to_jwt()
        logger.info("✓ Token generated")
        return token
    except Exception as e:
        logger.error(f"Failed to generate token: {e}")
        return None


async def run_livekit_agent():
    """Run agent connected to LiveKit."""
    if not (LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        logger.error("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set")
        logger.error("Set them in .env or environment, then run: python agent.py --livekit")
        return
    
    # Get token
    token = await fetch_token_from_server(ROOM, AGENT_ID)
    
    if not token:
        token = generate_token_directly()
    
    if not token:
        logger.error("Could not obtain token")
        return
    
    # Create and run agent
    agent = LiveKitAgent(ROOM, AGENT_ID, token)
    
    try:
        await agent.connect()
        
        # Keep running until interrupted
        while agent.is_running:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Agent interrupted by user")
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
    finally:
        await agent.disconnect()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LiveKit voice agent")
    parser.add_argument(
        "--livekit",
        action="store_true",
        help="Connect to LiveKit room"
    )
    parser.add_argument(
        "--local-debug",
        action="store_true",
        help="Test STT/LLM/TTS pipeline locally (no LiveKit)"
    )
    parser.add_argument(
        "--sample",
        type=str,
        help="Input WAV file for local debug (default: ./sample.wav)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output WAV file for local debug (default: ./sample_reply.wav)"
    )
    
    args = parser.parse_args()
    
    if args.local_debug:
        # Local debug mode
        sample_in = args.sample or "./sample.wav"
        sample_out = args.output or "./sample_reply.wav"
        
        if not os.path.exists(sample_in):
            logger.error(f"Sample file not found: {sample_in}")
            logger.info("Usage: python agent.py --local-debug --sample <input.wav> [--output <output.wav>]")
            return
        
        await process_audio_file(sample_in, sample_out)
    
    elif args.livekit:
        # LiveKit mode
        await run_livekit_agent()
    
    else:
        # Show usage
        parser.print_help()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
