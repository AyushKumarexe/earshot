import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from earshot.audio.features import frame_stack


class GMMDetector:
    input = "spec"

    def __init__(self, n_components=16, n_mfcc=20, max_frames=200_000, seed=0):
        self.n_components = n_components
        self.n_mfcc = n_mfcc
        self.max_frames = max_frames
        self.seed = seed

    def _frames(self, spec):
        mfcc = librosa.feature.mfcc(S=spec, n_mfcc=self.n_mfcc)
        delta = librosa.feature.delta(mfcc, width=9)
        return np.vstack([mfcc, delta]).T

    def fit(self, specs, log=print):
        X = np.concatenate([self._frames(s) for s in specs])
        if len(X) > self.max_frames:
            X = X[np.random.default_rng(self.seed).choice(len(X), self.max_frames, replace=False)]
        log(f"fitting GMM on {len(X)} frames")
        self.gmm = GaussianMixture(self.n_components, covariance_type="diag", random_state=self.seed).fit(X)
        return self

    def score(self, spec):
        return float(-self.gmm.score(self._frames(spec)))


class DenseAE(nn.Module):
    """The DCASE 2020 Task 2 baseline: 640 -> 128 x4 -> 8 -> 128 x4 -> 640."""

    def __init__(self, n_in=640, hidden=128, bottleneck=8):
        super().__init__()
        sizes = [n_in, hidden, hidden, hidden, hidden, bottleneck, hidden, hidden, hidden, hidden]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.Linear(hidden, n_in))

    def forward(self, x):
        return self.net(x)


class ConvAE(nn.Module):
    def __init__(self, n_mels=128, frames=32, latent=64):
        super().__init__()
        self.shape = (64, n_mels // 8, frames // 8)
        flat = int(np.prod(self.shape))

        def down(a, b):
            return [nn.Conv2d(a, b, 3, 2, 1), nn.BatchNorm2d(b), nn.ReLU()]

        def up(a, b):
            return [nn.ConvTranspose2d(a, b, 4, 2, 1), nn.BatchNorm2d(b), nn.ReLU()]

        self.encoder = nn.Sequential(*down(1, 16), *down(16, 32), *down(32, 64), nn.Flatten(), nn.Linear(flat, latent))
        self.expand = nn.Linear(latent, flat)
        self.decoder = nn.Sequential(*up(64, 32), *up(32, 16), nn.ConvTranspose2d(16, 1, 4, 2, 1))

    def forward(self, x):
        z = self.expand(self.encoder(x)).view(-1, *self.shape)
        return self.decoder(z)


class AEDetector:
    input = "spec"

    def __init__(self, kind="dense", context=5, patch=32, epochs=100, batch_size=512, lr=1e-3,
                 samples_per_clip=64, seed=0):
        self.kind = kind
        self.context = context
        self.patch = patch
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.samples_per_clip = samples_per_clip
        self.seed = seed

    def vectors(self, spec):
        x = (spec - self.mean[:, None]) / self.std[:, None]
        if self.kind == "dense":
            return frame_stack(x, self.context).astype(np.float32)
        hop = self.patch // 2
        starts = range(0, x.shape[1] - self.patch + 1, hop)
        return np.stack([x[:, s:s + self.patch] for s in starts])[:, None].astype(np.float32)

    @property
    def input_shape(self):
        if self.kind == "dense":
            return (len(self.mean) * self.context,)
        return (1, len(self.mean), self.patch)

    def fit(self, specs, log=print):
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        n_frames = sum(s.shape[1] for s in specs)
        self.mean = sum(s.sum(axis=1) for s in specs) / n_frames
        self.std = np.sqrt(sum(((s - self.mean[:, None]) ** 2).sum(axis=1) for s in specs) / n_frames) + 1e-6

        parts = []
        for s in specs:
            v = self.vectors(s)
            n = min(len(v), self.samples_per_clip)
            parts.append(v[rng.choice(len(v), n, replace=False)])
        X = torch.as_tensor(np.concatenate(parts))
        log(f"training {self.kind} autoencoder on {len(X)} samples")

        n_mels = len(self.mean)
        self.model = DenseAE(n_mels * self.context) if self.kind == "dense" else ConvAE(n_mels, self.patch)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            self.model.train()
            total = 0.0
            for idx in torch.randperm(len(X)).split(self.batch_size):
                batch = X[idx].to(device)
                loss = ((self.model(batch) - batch) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                log(f"  epoch {epoch + 1:3d}  loss {total / len(X):.4f}")
        self.model.cpu().eval()
        return self

    @torch.no_grad()
    def reconstruct(self, x):
        return self.model(torch.as_tensor(x)).numpy()

    def score(self, spec):
        x = self.vectors(spec)
        return float(((self.reconstruct(x) - x) ** 2).mean())


class KNNDetector:
    def __init__(self, embedding="stats", k=5):
        self.embedding = embedding
        self.input = "panns" if embedding == "panns" else "spec"
        self.k = k

    def embed(self, x):
        if self.embedding == "stats":
            return np.concatenate([x.mean(axis=1), x.std(axis=1)])
        return np.asarray(x)

    def fit(self, xs, log=print):
        E = np.stack([self.embed(x) for x in xs])
        self.mean, self.std = E.mean(0), E.std(0) + 1e-6
        log(f"kNN bank of {len(E)} normal clips, {E.shape[1]}-d")
        self.nn = NearestNeighbors(n_neighbors=self.k).fit((E - self.mean) / self.std)
        return self

    def score(self, x):
        e = (self.embed(x) - self.mean) / self.std
        dist, _ = self.nn.kneighbors(e[None])
        return float(dist.mean())


def build_detector(name, cfg):
    params = cfg["models"].get(name, {})
    if name == "gmm":
        return GMMDetector(**params)
    if name in ("ae", "conv"):
        return AEDetector(kind="dense" if name == "ae" else "conv", **params)
    if name == "knn":
        return KNNDetector(**params)
    raise ValueError(f"unknown detector {name!r}")
