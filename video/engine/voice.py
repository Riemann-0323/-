"""配音后端：MiniMax（正式）与 Kokoro（离线兜底，只用于预览）。

每个后端都实现 synth_scene(lines, cache_dir) -> (audio, sr, [(句开始, 句结束), ...])，
时间都是相对这段场景音频的秒数。
"""
import json
import re
from pathlib import Path

import numpy as np

from . import tts, tts_minimax

ROOT = Path(__file__).resolve().parents[1]
VOICE_CONFIG = ROOT / "voice.json"
_TOKEN = re.compile(r"[一-鿿]|\d+(?:\.\d+)?|[A-Za-z]+")


def weight(text):
    """估算一段文字的"朗读长度"：汉字 1，数字按位数折算，英文单词 2。"""
    w = 0.0
    for tok in _TOKEN.findall(text):
        if tok.isdigit() or tok.replace(".", "", 1).isdigit():
            w += min(len(tok.replace(".", "")), 6) * 1.1 + (1 if "." in tok else 0)
        elif tok.isascii():
            w += 2
        else:
            w += 1
    return max(w, 1.0)


def strip_quotes(text):
    return re.sub(r"[“”「」\"]", "", text)


def _normalize(x, peak=0.85):
    m = np.abs(x).max()
    return x / m * peak if m > 0 else x


def _concat(parts, pauses, sr):
    """把逐句音频按停顿拼接，返回 (audio, times)。"""
    out, times, t = [], [], 0.0
    for i, x in enumerate(parts):
        times.append((t, t + len(x) / sr))
        out.append(x)
        t += len(x) / sr
        if i < len(parts) - 1:
            gap = np.zeros(int(pauses[i] * sr), np.float32)
            out.append(gap)
            t += len(gap) / sr
    return np.concatenate(out).astype(np.float32), times


class KokoroVoice:
    name = "kokoro"
    default_speed = 1.15

    def synth_scene(self, lines, cache_dir):
        parts, sr = [], 24000
        for ln in lines:
            x, sr = tts.synth(ln.tts or strip_quotes(ln.text), cache_dir, speed=ln.speed or self.default_speed)
            parts.append(_normalize(x, 0.8))
        audio, times = _concat(parts, [ln.pause for ln in lines], sr)
        return audio, sr, times


class MiniMaxVoice:
    name = "minimax"

    def __init__(self, voice_id, speed=1.1, tones=(), emotion=None):
        self.voice_id, self.speed, self.tones, self.emotion = voice_id, speed, tuple(tones), emotion

    def _text(self, ln):
        return ln.mm or strip_quotes(ln.text)

    def synth_scene(self, lines, cache_dir):
        texts = [self._text(ln) for ln in lines]
        joined = ""
        for i, (tx, ln) in enumerate(zip(texts, lines)):
            joined += tx
            if i < len(lines) - 1 and ln.pause >= 0.05:
                joined += f"<#{ln.pause:.2f}#>"
        x, sr, subs = tts_minimax.synthesize(joined, self.voice_id, cache_dir, self.speed, self.tones, self.emotion)
        if len(lines) == 1:
            times = [(0.0, len(x) / sr)]
        elif subs and len(subs) >= len(lines):
            times = align(texts, [ln.pause for ln in lines], x, sr, subs)
        else:
            times = None
        if times is None:
            # 拿不到可靠的句级时间戳时，逐句合成再拼接（每句时间精确，句间衔接略逊）
            parts = []
            for tx in texts:
                y, sr, _ = tts_minimax.synthesize(tx, self.voice_id, cache_dir, self.speed, self.tones, self.emotion,
                                                  subtitles=False)
                parts.append(y)
            x, times = _concat(parts, [ln.pause for ln in lines], sr)
        return _normalize(x), sr, times


def align(texts, pauses, x, sr, subs, snap=0.35):
    """把每句话对齐到整段音频上：先按 MiniMax 句级时间戳定位，再吸附到最近的停顿。"""
    dur = len(x) / sr
    gaps = tts.silences(x, sr, min_len=0.08)
    w = np.array([weight(t) for t in texts])
    cum = np.cumsum(w) / w.sum()
    if subs:
        sw = np.array([weight(s[0]) for s in subs])
        scum = np.cumsum(sw) / max(sw.sum(), 1e-6)
        ends = [s[2] for s in subs]
        starts = [s[1] for s in subs]

        def time_at(p):
            j = int(np.searchsorted(scum, p - 1e-9))
            j = min(j, len(subs) - 1)
            p0 = scum[j - 1] if j > 0 else 0.0
            frac = (p - p0) / max(scum[j] - p0, 1e-6)
            return starts[j] + (ends[j] - starts[j]) * min(max(frac, 0), 1)
        bounds = [time_at(p) for p in cum[:-1]]
    else:
        if not gaps:
            return None
        bounds = list(cum[:-1] * dur)
    # 吸附到停顿：句间插入了明确的停顿，通常是附近最长的那段静音
    times, prev = [], 0.0
    for i, b in enumerate(bounds):
        cand = [(g1 - g0, g0, g1) for g0, g1 in gaps if abs((g0 + g1) / 2 - b) < snap + pauses[i] and g0 > prev]
        if cand:
            _, g0, g1 = max(cand)
            end_i, start_next = g0 + 0.03, g1 - 0.03
        elif subs:
            end_i = start_next = b
        else:
            return None
        times.append((prev, max(end_i, prev + 0.2)))
        prev = max(start_next, prev + 0.2)
    times.append((prev, dur))
    return times


def load_config():
    if VOICE_CONFIG.exists():
        return json.loads(VOICE_CONFIG.read_text())
    return {}


def choose(kind="auto", tones=(), log=print):
    """auto：配置了 MiniMax Key 和音色就用 MiniMax，否则退回 Kokoro（只适合预览）。"""
    cfg = load_config()
    if kind in ("auto", "minimax") and tts_minimax.available() and cfg.get("voice_id"):
        log(f"  配音：MiniMax {tts_minimax.model()} · 音色 {cfg['voice_id']} · 语速 {cfg.get('speed', 1.1)}")
        return MiniMaxVoice(cfg["voice_id"], cfg.get("speed", 1.1), tones, cfg.get("emotion"))
    if kind == "minimax":
        missing = "MINIMAX_API_KEY 环境变量" if not tts_minimax.available() else f"{VOICE_CONFIG.name} 里的 voice_id"
        raise SystemExit(f"无法使用 MiniMax：缺少 {missing}。先运行 python3 -m tools.voice_samples 选定音色。")
    log("  ⚠ 配音：使用离线 Kokoro 临时配音（仅供预览，正式成片请配置 MiniMax）")
    return KokoroVoice()
