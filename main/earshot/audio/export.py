import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from earshot.audio.features import Featurizer
from earshot.audio.models import AEDetector
from earshot.audio.score import load_bundle


class OnnxAE:
    def __init__(self, detector, path):
        self.det = detector
        self.input = detector.input
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def score(self, spec):
        x = self.det.vectors(spec)
        y = self.session.run(None, {"x": x})[0]
        return float(((y - x) ** 2).mean())


def export(det, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / f"{stem}.onnx"
    int8 = out_dir / f"{stem}.int8.onnx"
    example = torch.zeros(2, *det.input_shape)
    torch.onnx.export(det.model.eval(), example, str(fp32), input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "n"}, "y": {0: "n"}}, dynamo=False)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    return fp32, int8


def benchmark(scorer, specs, repeats=3):
    scorer.score(specs[0])
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        scores = [scorer.score(s) for s in specs]
        best = min(best, (time.perf_counter() - start) / len(specs))
    return best * 1000, np.array(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--out", default="runs/onnx")
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    det = bundle["detector"]
    if not isinstance(det, AEDetector):
        raise SystemExit("only the autoencoder detectors (ae, conv) export to ONNX")

    fp32, int8 = export(det, args.out, Path(args.bundle).stem)
    n_mels = len(det.mean)
    rng = np.random.default_rng(0)
    frames = Featurizer(bundle["config"]["features"], cache_dir=bundle["config"]["cache_dir"]).window_frames
    specs = [(det.mean[:, None] + det.std[:, None] * rng.normal(size=(n_mels, frames))).astype(np.float32)
             for _ in range(50)]

    ms_torch, ref = benchmark(det, specs)
    print(f"torch fp32  {ms_torch:6.2f} ms / window")
    for label, path in [("onnx fp32", fp32), ("onnx int8", int8)]:
        ms, scores = benchmark(OnnxAE(det, path), specs)
        corr = np.corrcoef(ref, scores)[0, 1]
        print(f"{label}   {ms:6.2f} ms / window   corr with torch {corr:.4f}   {path.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
