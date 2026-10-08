import hashlib
from pathlib import Path

import librosa
import numpy as np

SR = 16000


def load_audio(path, sr=SR, offset=0.0, duration=None, channel=0):
    y, _ = librosa.load(path, sr=sr, mono=False, offset=offset, duration=duration)
    if y.ndim > 1:
        y = y[channel]
    return y


def log_mel(y, sr=SR, n_mels=128, n_fft=1024, hop=512):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
    return librosa.power_to_db(mel, ref=1.0, amin=1e-10).astype(np.float32)


def frame_stack(spec, context=5):
    n_mels, n_frames = spec.shape
    n = n_frames - context + 1
    if n < 1:
        return np.empty((0, n_mels * context), np.float32)
    idx = np.arange(n)[:, None] + np.arange(context)[None, :]
    return spec.T[idx].reshape(n, -1)


def spec_windows(spec, frames, hop):
    return [spec[:, i:i + frames] for i in range(0, spec.shape[1] - frames + 1, hop)]


class PannsEmbedder:
    def __init__(self, device="cpu"):
        from panns_inference import AudioTagging
        self.model = AudioTagging(checkpoint_path=None, device=device)

    def __call__(self, y, sr=SR):
        y = librosa.resample(y, orig_sr=sr, target_sr=32000)
        _, emb = self.model.inference(y[None, :])
        return emb[0]


class Featurizer:
    """Log-mel spectrograms, or PANNs embeddings for the kNN detector, cached on disk per clip."""

    def __init__(self, cfg, kind="spec", cache_dir="data/cache"):
        self.cfg = cfg
        self.kind = kind
        self.sr = cfg["sr"]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.window_samples = int(cfg["window_s"] * self.sr)
        self.window_frames = 1 + self.window_samples // cfg["hop"]
        self._panns = None

    def spec(self, y):
        c = self.cfg
        return log_mel(y, self.sr, c["n_mels"], c["n_fft"], c["hop"])

    def from_audio(self, y):
        if self.kind == "panns":
            if self._panns is None:
                self._panns = PannsEmbedder()
            return self._panns(y, self.sr)
        return self.spec(y)

    def _key(self, clip):
        c = self.cfg
        raw = f"{self.kind}|{clip.path}|{clip.offset}|{clip.duration}|{c['n_mels']}|{c['n_fft']}|{c['hop']}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def clip(self, clip):
        path = self.cache_dir / f"{self._key(clip)}.npy"
        if path.exists():
            return np.load(path)
        x = self.from_audio(load_audio(clip.path, self.sr, clip.offset, clip.duration))
        np.save(path, x)
        return x

    def windows(self, clip, max_windows=None):
        if self.kind == "panns":
            y = load_audio(clip.path, self.sr, clip.offset, clip.duration)
            step = self.window_samples // 2
            chunks = [y[i:i + self.window_samples] for i in range(0, len(y) - self.window_samples + 1, step)]
            return [self.from_audio(ch) for ch in chunks[:max_windows]]
        wins = spec_windows(self.clip(clip), self.window_frames, self.window_frames // 2)
        return wins[:max_windows]
