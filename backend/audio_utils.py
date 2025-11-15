"""audio_utils.py

Utilities for audio processing: resampling, format conversion, WAV handling.

All audio published to LiveKit must be:
  - Format: PCM 16-bit signed integer (little-endian)
  - Sample rate: 48000 Hz
  - Channels: 1 (mono)
"""
import os
import logging
import tempfile
import wave
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 1
TARGET_DTYPE = np.int16


def resample_audio(
    audio_data: np.ndarray,
    src_sample_rate: int,
    dst_sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """
    Resample audio numpy array from src_sample_rate to dst_sample_rate.
    
    Args:
        audio_data: 1D numpy array of audio samples (mono)
        src_sample_rate: source sample rate (Hz)
        dst_sample_rate: target sample rate (Hz, default 48000)
    
    Returns:
        Resampled audio as numpy array
    """
    if src_sample_rate == dst_sample_rate:
        return audio_data
    
    # Try using librosa/scipy for high-quality resampling if available
    try:
        import resampy
        logger.debug(f"Resampling {src_sample_rate}Hz -> {dst_sample_rate}Hz using resampy")
        return resampy.resample(audio_data, src_sample_rate, dst_sample_rate)
    except ImportError:
        pass
    
    try:
        from scipy import signal
        logger.debug(f"Resampling {src_sample_rate}Hz -> {dst_sample_rate}Hz using scipy.signal")
        num_samples = int(len(audio_data) * dst_sample_rate / src_sample_rate)
        return signal.resample(audio_data, num_samples)
    except ImportError:
        pass
    
    # Fallback: numpy-based simple linear interpolation (lower quality)
    logger.warning(f"Using fallback resample (low quality); install resampy or scipy for better quality")
    ratio = dst_sample_rate / src_sample_rate
    num_samples = int(len(audio_data) * ratio)
    indices = np.arange(num_samples) / ratio
    return np.interp(indices, np.arange(len(audio_data)), audio_data)


def ensure_wav_48k16_mono(in_path: str, out_path: str):
    """
    Read audio from in_path and write to out_path as:
      - 48 kHz sample rate
      - 16-bit signed PCM
      - Mono (1 channel)
    
    Args:
        in_path: input WAV file path
        out_path: output WAV file path (48k 16-bit mono)
    """
    logger.info(f"Converting audio: {in_path} -> {out_path}")
    
    # Read input audio
    try:
        audio_data, sr = sf.read(in_path, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to read audio file {in_path}: {e}")
        raise
    
    # Handle stereo/multichannel -> mono
    if len(audio_data.shape) == 2:
        logger.debug(f"Converting {audio_data.shape[1]} channels to mono")
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample if needed
    if sr != TARGET_SAMPLE_RATE:
        audio_data = resample_audio(audio_data, sr, TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE
    
    # Normalize to [-1, 1] range if needed
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        logger.debug(f"Normalizing audio (max={max_val})")
        audio_data = audio_data / max_val
    
    # Convert to int16
    audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
    
    # Write output
    try:
        sf.write(out_path, audio_int16, sr, subtype='PCM_16')
        logger.info(f"✓ Wrote WAV: {sr}Hz, 16-bit, mono -> {out_path}")
    except Exception as e:
        logger.error(f"Failed to write audio file {out_path}: {e}")
        raise


def load_wav_48k16_mono(wav_path: str) -> tuple[np.ndarray, int]:
    """
    Load a WAV file and ensure it's 48k 16-bit mono.
    
    Args:
        wav_path: path to WAV file
    
    Returns:
        (audio_data, sample_rate) where audio_data is np.int16 array, sample_rate is int
    """
    try:
        audio_data, sr = sf.read(wav_path, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to read {wav_path}: {e}")
        raise
    
    # Ensure mono
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample if needed
    if sr != TARGET_SAMPLE_RATE:
        audio_data = resample_audio(audio_data, sr, TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE
    
    # Convert to int16
    audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
    
    return audio_int16, sr


def verify_wav_format(wav_path: str) -> dict:
    """
    Verify WAV file is 48k 16-bit mono and return metadata.
    
    Args:
        wav_path: path to WAV file
    
    Returns:
        {sample_rate, channels, dtype, num_frames, duration_sec, is_valid}
    """
    try:
        info = sf.info(wav_path)
        data = sf.read(wav_path, dtype=np.int16)
        
        # Check format
        is_valid = (
            info.samplerate == TARGET_SAMPLE_RATE
            and info.channels == TARGET_CHANNELS
            # soundfile reads as int16, so we check the data dtype
        )
        
        result = {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "num_frames": info.frames,
            "duration_sec": info.duration,
            "is_valid": is_valid,
        }
        
        if not is_valid:
            logger.warning(
                f"WAV format mismatch: expected 48k mono, got {info.samplerate}Hz "
                f"{info.channels}ch {info.subtype}"
            )
        
        return result
    except Exception as e:
        logger.error(f"Failed to verify {wav_path}: {e}")
        return {"is_valid": False, "error": str(e)}
