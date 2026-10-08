import random
import re
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


@dataclass(frozen=True)
class Clip:
    path: Path
    machine_id: str
    label: int
    offset: float = 0.0
    duration: float | None = None
    condition: str = ""


def dcase_clips(root, split):
    clips = []
    for p in sorted((Path(root) / split).glob("*.wav")):
        m = re.match(r"(normal|anomaly)_(id_\d+)_", p.name)
        if m:
            clips.append(Clip(p, m.group(2), int(m.group(1) == "anomaly")))
    return clips


def mimii_clips(root, snr):
    clips = []
    for p in sorted(Path(root).glob(f"{snr}_dB_fan/**/id_*/*/*.wav")):
        condition = p.parent.name
        clips.append(Clip(p, p.parent.parent.name, int(condition == "abnormal"), condition=condition))
    return clips


def own_clips(root, normal_conditions, clip_seconds=10.0):
    clips = []
    for p in sorted(Path(root).glob("*/*.wav")):
        condition = p.parent.name
        length = sf.info(p).duration
        for start in range(0, int(length // clip_seconds)):
            clips.append(Clip(p, "own", int(condition not in normal_conditions),
                              start * clip_seconds, clip_seconds, condition))
    return clips


def holdout_split(clips, seed=0):
    """Hold out as many normal clips as there are anomalies per machine, like the DCASE test sets."""
    rng = random.Random(seed)
    train, test = [], []
    for mid in sorted({c.machine_id for c in clips}):
        normal = [c for c in clips if c.machine_id == mid and not c.label]
        anomalous = [c for c in clips if c.machine_id == mid and c.label]
        rng.shuffle(normal)
        test += normal[:len(anomalous)] + anomalous
        train += normal[len(anomalous):]
    return train, test


def load_splits(data_cfg, seed=0):
    layout = data_cfg["layout"]
    if layout == "dcase":
        train, test = dcase_clips(data_cfg["root"], "train"), dcase_clips(data_cfg["root"], "test")
    elif layout == "mimii":
        train, test = holdout_split(mimii_clips(data_cfg["mimii_root"], data_cfg["snr"]), seed)
    elif layout == "own":
        clips = own_clips(data_cfg["own_root"], data_cfg["normal_conditions"])
        train, test = holdout_split(clips, seed)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    if not train:
        raise FileNotFoundError(f"no training clips found for layout {layout!r}, check the paths in the config")

    normal = [c for c in train if not c.label]
    rng = random.Random(seed)
    rng.shuffle(normal)
    n_val = max(1, int(len(normal) * data_cfg["val_fraction"]))
    return normal[n_val:], normal[:n_val], test
