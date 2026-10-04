"""把逐字稿变成时间轴：每句配音的起止时间、场景区间、字幕条目。"""
import re
from dataclasses import dataclass, field

import numpy as np

from . import tts


@dataclass
class Line:
    """逐字稿里的一句话。

    text  : 字幕显示的文字
    tts   : 离线 Kokoro 读的文字（数字写汉字、多音字换同音字）；也作为语音识别校对的参照
    mm    : 给 MiniMax 读的文字（默认用 text 去掉引号；MiniMax 自己会读数字，多音字用发音词典处理）
    pause : 这句话之后的停顿（秒）
    speed : 只对 Kokoro 生效的语速
    """
    text: str
    tts: str = None
    pause: float = 0.15
    speed: float = None
    mm: str = None

    @property
    def reference(self):
        return self.tts or self.text


@dataclass
class Spoken:
    scene: str
    text: str
    spoken: str
    start: float
    dur: float
    audio: np.ndarray = field(repr=False)
    sr: int = 24000

    @property
    def end(self):
        return self.start + self.dur


@dataclass
class SceneSpan:
    name: str
    start: float
    end: float
    lines: list          # 本场景内各句的 (相对开始, 相对结束)


def build(scenes, voice, cache_dir, lead=0.35, scene_gap=0.03, tail=0.3, visual_lead=0.15, log=print):
    """逐场景合成配音并排出时间轴。返回 (逐句信息, 场景区间, 总时长, 场景音频列表)。"""
    t = lead
    spoken, raw, scene_audio = [], [], []
    for name, lines in scenes:
        audio, sr, times = voice.synth_scene(lines, cache_dir)
        scene_audio.append((t, audio, sr))
        rel = []
        for ln, (a, b) in zip(lines, times):
            seg = audio[int(a * sr):int(b * sr)]
            spoken.append(Spoken(name, ln.text, ln.reference, t + a, b - a, seg, sr))
            rel.append((t + a, t + b))
        first = t
        t += len(audio) / sr + lines[-1].pause + scene_gap
        raw.append([name, first, t, rel])
        log(f"  {name:10s} {first:6.2f}s – {t:6.2f}s  ({len(lines)} 句)")
    total = t + tail
    spans = []
    for i, (name, first, end, rel) in enumerate(raw):
        s = 0.0 if i == 0 else first - visual_lead
        e = raw[i + 1][1] - visual_lead if i + 1 < len(raw) else total
        spans.append(SceneSpan(name, s, e, [(a - s, b - s) for a, b in rel]))
    return spoken, spans, total, scene_audio


_SPLIT = re.compile(r"([，。？！：；])")


def _clauses(text):
    parts = _SPLIT.split(text)
    out, cur = [], ""
    for p in parts:
        if not p:
            continue
        cur += p
        if _SPLIT.fullmatch(p):
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    # 太短的分句并到下一句（避免字幕一闪而过）
    merged = []
    for c in out:
        if merged and len(tts.han(merged[-1])) < 4:
            merged[-1] += c
        else:
            merged.append(c)
    if len(merged) > 1 and len(tts.han(merged[-1])) < 4:
        last = merged.pop()
        merged[-1] += last
    return merged


def _display(c):
    c = c.strip().replace("。”", "”").replace("，”", "”")
    while c and c[-1] in "，。：；":
        c = c[:-1]
    return c


def subtitles(spoken, max_chars=24):
    """把每句话切成字幕条目，切分点吸附到配音中的停顿处。返回 [(start, end, text)]。"""
    cues = []
    for s in spoken:
        parts = _clauses(s.text)
        # 一个分句太长时再按字数对半拆
        fixed = []
        for p in parts:
            if len(tts.han(p)) > max_chars and "，" not in p[:-1]:
                h = len(p) // 2
                fixed += [p[:h], p[h:]]
            else:
                fixed.append(p)
        parts = fixed
        weights = np.array([max(1, len(tts.han(p)) + len(re.findall(r"\d", p)) * 0.8) for p in parts], float)
        bounds = np.cumsum(weights)[:-1] / weights.sum() * s.dur
        gaps = tts.silences(s.audio, s.sr)
        mids = np.array([(a + b) / 2 for a, b in gaps]) if gaps else np.array([])
        times = [0.0]
        for b in bounds:
            if len(mids):
                j = np.argmin(np.abs(mids - b))
                if abs(mids[j] - b) < 0.7 and mids[j] > times[-1] + 0.4:
                    b = mids[j]
            times.append(max(b, times[-1] + 0.4))
        times.append(s.dur)
        for p, a, b in zip(parts, times[:-1], times[1:]):
            cues.append((s.start + a, s.start + b + 0.12, _display(p)))
    # 相邻字幕之间不留太短的空隙
    fixed = []
    for i, (a, b, t) in enumerate(cues):
        if i + 1 < len(cues) and cues[i + 1][0] - b < 0.35:
            b = cues[i + 1][0]
        fixed.append((a, b, t))
    return fixed


def srt(cues):
    def ts(x):
        h, r = divmod(x, 3600)
        m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"
    return "\n".join(f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n" for i, (a, b, t) in enumerate(cues, 1))
