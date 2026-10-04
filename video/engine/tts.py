"""离线中文配音（Kokoro v1.1-zh，经 sherpa-onnx 运行）与语音识别校对（SenseVoice）。"""
import hashlib
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from .core import ASSETS

VOICE_SID = 62          # 选定的男声音色，固定不变
SAMPLE_RATE = 24000

_tts = None
_asr = None


def _engine():
    global _tts
    if _tts is None:
        import sherpa_onnx
        k = ASSETS / "kokoro-multi-lang-v1_1"
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(k / "model.onnx"), voices=str(k / "voices.bin"), tokens=str(k / "tokens.txt"),
                    data_dir=str(k / "espeak-ng-data"), dict_dir=str(k / "dict"),
                    lexicon=f"{k / 'lexicon-us-en.txt'},{k / 'lexicon-zh.txt'}"),
                num_threads=4),
            rule_fsts=f"{k / 'phone-zh.fst'},{k / 'date-zh.fst'},{k / 'number-zh.fst'}",
            max_num_sentences=1)
        _tts = sherpa_onnx.OfflineTts(cfg)
    return _tts


def _recognizer():
    global _asr
    if _asr is None:
        import sherpa_onnx
        a = ASSETS / "sense-voice"
        _asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(a / "model.int8.onnx"), tokens=str(a / "tokens.txt"),
            use_itn=False, language="zh", num_threads=4)
    return _asr


def trim_silence(x, sr, thr=0.008, keep=0.03):
    """去掉首尾静音，各保留 keep 秒。"""
    env = np.abs(x)
    idx = np.where(env > thr)[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(keep * sr))
    b = min(len(x), idx[-1] + int(keep * sr))
    return x[a:b]


def synth(text, cache_dir, speed=1.0, sid=VOICE_SID):
    """合成一句话，带磁盘缓存。返回 (samples float32, sample_rate)。"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{sid}|{speed}|{text}".encode()).hexdigest()[:16]
    path = cache_dir / f"{key}.wav"
    if path.exists():
        x, sr = sf.read(path, dtype="float32")
        return x, sr
    audio = _engine().generate(text, sid=sid, speed=speed)
    x = trim_silence(np.asarray(audio.samples, np.float32), audio.sample_rate)
    sf.write(path, x, audio.sample_rate)
    return x, audio.sample_rate


def transcribe(x, sr):
    s = _recognizer().create_stream()
    s.accept_waveform(sr, x)
    _recognizer().decode_stream(s)
    return s.result.text


def han(text):
    return re.sub(r"[^一-鿿]", "", text)


def cer(ref, hyp):
    """字错误率（只比较汉字）。"""
    r, h = han(ref), han(hyp)
    if not r:
        return 0.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[:], i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)


def silences(x, sr, min_len=0.06, thr_db=-38):
    """返回句内停顿区间列表 [(start, end), ...]（秒）。"""
    hop = int(0.01 * sr)
    frames = len(x) // hop
    if frames == 0:
        return []
    rms = np.sqrt(np.mean(x[:frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-9))
    quiet = db < thr_db
    out, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * 0.01 >= min_len:
                out.append((start * 0.01, i * 0.01))
            start = None
    return out
