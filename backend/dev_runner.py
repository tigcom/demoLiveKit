#!/usr/bin/env python3
"""dev_runner.py

Simple dev helper to start token server, agent, and display status.

Usage:
  python dev_runner.py --token-server   # Start token server only
  python dev_runner.py --agent          # Start agent only
  python dev_runner.py --test-pipeline  # Run test pipeline
"""
import subprocess
import sys
import argparse
import os
import time
from pathlib import Path

def run_command(cmd, name):
    """Run a command and display its output."""
    print(f"\n{'='*70}")
    print(f"Starting: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print(f"\n✓ {name} stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"✗ {name} failed with exit code {e.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="LiveKit dev helper")
    parser.add_argument("--token-server", action="store_true", help="Run token server")
    parser.add_argument("--agent", action="store_true", help="Run agent")
    parser.add_argument("--test-pipeline", action="store_true", help="Run test pipeline")
    parser.add_argument("--sample", type=str, help="Sample WAV for test pipeline")
    
    args = parser.parse_args()
    
    # Determine which script to run
    if not any([args.token_server, args.agent, args.test_pipeline]):
        # Default: show usage
        print("Dev Helper for LiveKit Voice Demo")
        print()
        print("Usage:")
        print("  python dev_runner.py --token-server         # Start token server")
        print("  python dev_runner.py --agent                # Start agent")
        print("  python dev_runner.py --test-pipeline        # Test STT/LLM/TTS")
        print("  python dev_runner.py --test-pipeline --sample sample.wav")
        print()
        print("Tips:")
        print("  - Ensure .env is configured (copy from .env.example)")
        print("  - Start Ollama in another terminal: ollama serve")
        print("  - In another terminal, run: cd frontend && npm run dev")
        return 0
    
    # Check we're in backend directory
    if not Path("requirements.txt").exists():
        print("✗ Error: requirements.txt not found")
        print("  Run from backend/ directory")
        return 1
    
    # Check venv is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠ Warning: Python virtual environment not activated")
        print("  Activate with: .venv\\Scripts\\Activate.ps1")
    
    # Run selected command
    if args.token_server:
        cmd = [sys.executable, "-m", "uvicorn", "token_server:app", "--host", "0.0.0.0", "--port", "8000"]
        run_command(cmd, "Token Server")
    
    elif args.agent:
        cmd = [sys.executable, "agent.py"]
        run_command(cmd, "Agent")
    
    elif args.test_pipeline:
        if not args.sample:
            print("✗ Error: --sample required for test pipeline")
            return 1
        cmd = [sys.executable, "test_pipeline.py", "--sample", args.sample]
        run_command(cmd, "Test Pipeline")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
