import argparse
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="Record a fan into recordings/<condition>/.")
    parser.add_argument("condition", help="e.g. idle, load, blocked, imbalance")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--device")
    parser.add_argument("--root", default="recordings")
    args = parser.parse_args()

    out = Path(args.root) / args.condition
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d_%H%M%S')}.wav"
    device = int(args.device) if args.device and args.device.isdigit() else args.device

    print(f"recording {args.seconds:.0f}s of '{args.condition}'...")
    audio = sd.rec(int(args.seconds * args.sr), samplerate=args.sr, channels=1, device=device)
    sd.wait()
    sf.write(path, audio, args.sr)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
