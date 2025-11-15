#!/usr/bin/env python3
"""test_pipeline.py

Standalone CLI to test the STT -> LLM -> TTS pipeline locally without LiveKit.

Usage:
  python test_pipeline.py --sample sample.wav --output reply.wav
  python test_pipeline.py --sample sample.wav --mock-llm   # Use mock LLM instead of ollama
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def transcribe_wav(wav_path: str) -> str:
    """Transcribe WAV file using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
        logger.info(f"Loading faster-whisper model...")
        model = WhisperModel("tiny")
        logger.info(f"Transcribing: {wav_path}")
        segments, _ = model.transcribe(wav_path)
        text = " ".join([s.text.strip() for s in segments]).strip()
        logger.info(f"✓ Transcribed text: {text}")
        return text
    except ImportError:
        logger.error("faster-whisper not installed. Install: pip install faster-whisper")
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise


def query_llm_real(text: str) -> str:
    """Query local LLM via Hugging Face Transformers.

    This will attempt to load a text-generation pipeline using torch. If
    transformers/torch are not installed, an informative error is raised.
    """
    try:
        from transformers import pipeline
        import torch
    except ImportError:
        logger.error("transformers or torch not installed. Install: pip install transformers torch torchvision torchaudio")
        raise

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else -1
    logger.info(f"Querying transformers LLM on {'cuda' if use_cuda else 'cpu'}...")
    try:
        pipe = pipeline("text-generation", model="gpt2", device=device, trust_remote_code=True)
        out = pipe(text, max_new_tokens=128, do_sample=False, temperature=0.0)
        if isinstance(out, list) and len(out) > 0:
            generated = out[0].get("generated_text", "").strip()
            if generated.startswith(text):
                reply = generated[len(text):].strip()
            else:
                reply = generated
        else:
            reply = "Sorry, I couldn't understand."

        logger.info(f"✓ LLM reply: {reply}")
        return reply
    except Exception as e:
        logger.error(f"LLM (transformers) query failed: {e}", exc_info=True)
        return "Sorry, I couldn't process that."


def query_llm_mock(text: str) -> str:
    """Mock LLM response for testing."""
    logger.info(f"Using mock LLM (no ollama required)")
    mock_replies = {
        "hello": "Hello! How are you today?",
        "how are you": "I'm doing great, thanks for asking!",
        "what is your name": "I'm an AI assistant.",
        "thank you": "You're welcome!",
    }
    text_lower = text.lower()
    for key, reply in mock_replies.items():
        if key in text_lower:
            logger.info(f"✓ Mock LLM reply: {reply}")
            return reply
    # Default response
    reply = f"You said: {text}. That's interesting!"
    logger.info(f"✓ Mock LLM reply: {reply}")
    return reply


def synthesize_tts(text: str, out_wav_path: str) -> str:
    """Synthesize text to speech using Coqui TTS."""
    try:
        import tempfile
        from TTS.api import TTS
        from audio_utils import ensure_wav_48k16_mono
        
        logger.info(f"Loading Coqui TTS model...")
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)
        
        logger.info(f"Synthesizing TTS...")
        tmp_wav = tempfile.mktemp(suffix=".wav")
        tts.tts_to_file(text=text, file_path=tmp_wav)
        
        # Ensure output is 48k 16-bit mono
        ensure_wav_48k16_mono(tmp_wav, out_wav_path)
        logger.info(f"✓ TTS output saved to {out_wav_path}")
        
        # Clean up temp file
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        
        return out_wav_path
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install: pip install TTS")
        raise
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}", exc_info=True)
        raise


def verify_output_wav(wav_path: str):
    """Verify output WAV file format."""
    try:
        from audio_utils import verify_wav_format
        logger.info(f"Verifying output WAV format...")
        result = verify_wav_format(wav_path)
        
        logger.info(f"  Sample rate: {result.get('sample_rate')} Hz")
        logger.info(f"  Channels: {result.get('channels')}")
        logger.info(f"  Format: {result.get('format')} {result.get('subtype')}")
        logger.info(f"  Duration: {result.get('duration_sec'):.2f} sec")
        
        if result.get("is_valid"):
            logger.info(f"✓ WAV format is correct (48kHz 16-bit mono)")
        else:
            logger.warning(f"✗ WAV format mismatch; expected 48kHz 16-bit mono")
        
        return result
    except Exception as e:
        logger.error(f"WAV verification failed: {e}", exc_info=True)
        return None


async def main():
    parser = argparse.ArgumentParser(
        description="Test STT -> LLM -> TTS pipeline"
    )
    parser.add_argument(
        "--sample",
        type=str,
        required=True,
        help="Input WAV file to transcribe"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pipeline_reply.wav",
        help="Output WAV file for synthesized reply (default: pipeline_reply.wav)"
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use mock LLM instead of ollama (for testing without LLM running)"
    )
    
    args = parser.parse_args()
    
    # Verify input WAV exists
    if not os.path.exists(args.sample):
        logger.error(f"Sample WAV not found: {args.sample}")
        return 1
    
    try:
        logger.info("=" * 70)
        logger.info("STT -> LLM -> TTS Pipeline Test")
        logger.info("=" * 70)
        
        # 1. Transcribe
        logger.info("\n[1/3] Transcription")
        logger.info("-" * 70)
        text = transcribe_wav(args.sample)
        
        if not text:
            logger.warning("No speech detected in WAV file")
            return 1
        
        # 2. Query LLM
        logger.info("\n[2/3] LLM Query")
        logger.info("-" * 70)
        if args.mock_llm:
            reply_text = query_llm_mock(text)
        else:
            reply_text = query_llm_real(text)
        
        # 3. Synthesize
        logger.info("\n[3/3] Text-to-Speech Synthesis")
        logger.info("-" * 70)
        synthesize_tts(reply_text, args.output)
        
        # Verify output
        logger.info("\n[4] Output Verification")
        logger.info("-" * 70)
        verify_output_wav(args.output)
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Pipeline test complete!")
        logger.info(f"  Reply saved to: {args.output}")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline test failed: {e}")
        logger.error("Check the error messages above for details.")
        return 1


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()) or 0)
