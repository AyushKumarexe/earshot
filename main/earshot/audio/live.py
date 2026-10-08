import argparse
import csv
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from earshot.audio.features import Featurizer
from earshot.audio.score import calibration_for, load_bundle


def bar(health, width=30):
    filled = int(round(health * width))
    return "#" * filled + "." * (width - filled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", help="detector trained with --layout own works best for your fan")
    parser.add_argument("--onnx", help="score with this ONNX model instead of torch")
    parser.add_argument("--device", help="input device name or index, see python -m sounddevice")
    parser.add_argument("--machine", default="own")
    parser.add_argument("--smoothing", type=float, default=0.3)
    parser.add_argument("--log", default="runs/live/scores.csv")
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    det = bundle["detector"]
    if args.onnx:
        from earshot.audio.export import OnnxAE
        det = OnnxAE(det, args.onnx)
    cal = calibration_for(bundle, args.machine)
    feat = Featurizer(bundle["config"]["features"], det.input, bundle["config"]["cache_dir"])
    sr = feat.sr
    window, hop = feat.window_samples, feat.window_samples // 2

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["time", "score", "health"])

    chunks = queue.Queue()

    def callback(indata, frames, time_info, status):
        chunks.put(indata[:, 0].copy())

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    buffer = np.zeros(0, np.float32)
    health = None
    print(f"listening at {sr} Hz, ctrl+c to stop")
    try:
        with sd.InputStream(samplerate=sr, channels=1, blocksize=hop, dtype="float32",
                            device=device, callback=callback):
            while True:
                buffer = np.concatenate([buffer, chunks.get()])[-window:]
                if len(buffer) < window:
                    continue
                t0 = time.perf_counter()
                score = det.score(feat.from_audio(buffer))
                h = float(cal.health(score))
                health = h if health is None else (1 - args.smoothing) * health + args.smoothing * h
                ms = (time.perf_counter() - t0) * 1000
                log.writerow([time.time(), score, health])
                log_file.flush()
                flag = "  CHECK FAN" if health < 0.5 else ""
                print(f"\rhealth {health:4.2f} [{bar(health)}] score {score:8.3f} ({ms:4.1f} ms){flag}   ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
