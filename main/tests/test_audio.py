import numpy as np
import pytest
import soundfile as sf

from earshot import load_config
from earshot.audio.data import load_splits
from earshot.audio.features import Featurizer, frame_stack, log_mel
from earshot.audio.models import build_detector
from earshot.audio.train import calibrate, evaluate

SR = 16000


def fan_tone(rng, seconds, faulty=False):
    t = np.arange(int(seconds * SR)) / SR
    hum = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.1 * np.sin(2 * np.pi * 240 * t)
    if faulty:
        hum += 0.15 * np.sin(2 * np.pi * 1730 * t) * (1 + np.sin(2 * np.pi * 7 * t))
    return (hum + 0.05 * rng.normal(size=len(t))).astype(np.float32)


@pytest.fixture
def recordings(tmp_path):
    rng = np.random.default_rng(0)
    for condition, faulty, seconds in [("idle", False, 240), ("imbalance", True, 60)]:
        (tmp_path / condition).mkdir()
        sf.write(tmp_path / condition / "a.wav", fan_tone(rng, seconds, faulty), SR)
    return tmp_path


def test_frame_stack_shape():
    spec = np.arange(128 * 10, dtype=np.float32).reshape(128, 10)
    stacked = frame_stack(spec, 5)
    assert stacked.shape == (6, 640)
    assert np.allclose(stacked[1, :128], spec[:, 1])


def test_log_mel_one_second_is_32_frames():
    assert log_mel(np.zeros(SR, np.float32)).shape == (128, 32)


@pytest.mark.parametrize("model", ["gmm", "ae", "conv", "knn"])
def test_detectors_separate_a_faulty_tone(recordings, tmp_path, model):
    cfg = load_config("configs/detector.yaml")
    cfg["data"].update(layout="own", own_root=str(recordings), normal_conditions=["idle"])
    cfg["cache_dir"] = str(tmp_path / "cache")
    for name in ("ae", "conv"):
        cfg["models"][name]["epochs"] = 5

    fit, val, test = load_splits(cfg["data"], seed=0)
    det = build_detector(model, cfg)
    feat = Featurizer(cfg["features"], det.input, cfg["cache_dir"])
    det.fit([feat.clip(c) for c in fit], log=lambda *_: None)
    cals = calibrate(det, feat, val, max_windows=5)
    rows, _ = evaluate(det, feat, test)

    assert rows[-1]["auc"] > 0.9
    healthy = cals["own"].health([det.score(w) for w in feat.windows(val[0])])
    assert healthy.mean() > 0.5
