"""audio_source.py

Custom LiveKit AudioSource for publishing TTS-synthesized audio.

Handles:
- Loading WAV file (48kHz 16-bit mono)
- Pushing frames in 20ms chunks
- Proper timing and sample handling
- Audio format conversion if needed
"""
import logging
import struct
import wave
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# LiveKit audio frame parameters
FRAME_DURATION_MS = 20  # 20ms per frame (standard for LiveKit)
SAMPLE_RATE = 48000     # 48kHz (LiveKit standard)
CHANNELS = 1            # Mono
BYTES_PER_SAMPLE = 2    # 16-bit = 2 bytes
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 960 samples at 48kHz


def wav_to_frames(wav_path: str) -> list[bytes]:
    """
    Load a WAV file and split into 20ms frames (960 samples each at 48kHz).
    
    Args:
        wav_path: path to WAV file (should be 48kHz 16-bit mono)
    
    Returns:
        List of byte frames (each ~3840 bytes for 960 samples * 2 bytes)
    """
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            # Read WAV metadata
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            logger.debug(
                f"WAV info: {n_channels}ch, {framerate}Hz, "
                f"{sample_width}B/sample, {n_frames} frames"
            )
            
            # Validate format
            if framerate != SAMPLE_RATE:
                logger.warning(f"WAV sample rate {framerate}Hz != {SAMPLE_RATE}Hz")
            if sample_width != BYTES_PER_SAMPLE:
                logger.warning(f"WAV bit depth {sample_width*8}b != 16b")
            
            # Read all audio data
            audio_data = wav_file.readframes(n_frames)
            
            # Convert to numpy array (int16)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # If multi-channel, mix down to mono
            if n_channels > 1:
                audio_array = audio_array.reshape(-1, n_channels)
                audio_array = np.mean(audio_array, axis=1).astype(np.int16)
            
            # Split into 20ms frames
            frames = []
            for i in range(0, len(audio_array), SAMPLES_PER_FRAME):
                frame_samples = audio_array[i:i+SAMPLES_PER_FRAME]
                
                # Pad if last frame is shorter
                if len(frame_samples) < SAMPLES_PER_FRAME:
                    frame_samples = np.pad(
                        frame_samples,
                        (0, SAMPLES_PER_FRAME - len(frame_samples)),
                        mode='constant'
                    )
                
                # Convert to bytes
                frame_bytes = frame_samples.astype(np.int16).tobytes()
                frames.append(frame_bytes)
            
            logger.info(f"Split WAV into {len(frames)} frames ({len(frames) * 20}ms total)")
            return frames
    except Exception as e:
        logger.error(f"Failed to load WAV {wav_path}: {e}", exc_info=True)
        raise


class TTSAudioSource:
    """
    Simple TTS audio source that manages WAV playback and frame pushing.
    
    NOT a LiveKit AudioSource subclass, but a wrapper that can be used with one.
    To integrate with LiveKit, you'd subclass rtc.AudioSource and call:
        source.push_frame(data, sample_rate, num_channels, samples_per_channel)
    """
    
    def __init__(self):
        """Initialize audio source."""
        self.frames: list[bytes] = []
        self.current_frame_index = 0
        self.is_playing = False
        logger.debug("TTSAudioSource initialized")
    
    def load_wav(self, wav_path: str):
        """Load a WAV file into memory."""
        logger.info(f"Loading WAV: {wav_path}")
        self.frames = wav_to_frames(wav_path)
        self.current_frame_index = 0
        logger.info(f"✓ Loaded {len(self.frames)} frames")
    
    def start_playback(self):
        """Start playback."""
        self.is_playing = True
        self.current_frame_index = 0
        logger.debug("Playback started")
    
    def stop_playback(self):
        """Stop playback."""
        self.is_playing = False
        logger.debug("Playback stopped")
    
    def get_next_frame(self) -> Optional[bytes]:
        """
        Get the next 20ms frame of audio.
        
        Returns:
            Bytes of 960 samples (1920 bytes) at 48kHz 16-bit mono, or None if finished
        """
        if not self.is_playing or not self.frames:
            return None
        
        if self.current_frame_index >= len(self.frames):
            # End of playback
            self.is_playing = False
            return None
        
        frame = self.frames[self.current_frame_index]
        self.current_frame_index += 1
        return frame
    
    def get_duration_ms(self) -> int:
        """Get total duration of loaded audio in milliseconds."""
        return len(self.frames) * FRAME_DURATION_MS
    
    def reset(self):
        """Reset to start."""
        self.current_frame_index = 0
        self.is_playing = False
        logger.debug("Audio source reset")
