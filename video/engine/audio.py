"""背景音乐与音效全部用代码合成（无版权问题），再与配音混音。"""
import numpy as np
from scipy import signal

SR = 48000


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def note_hz(n):
    """MIDI 音高转频率。"""
    return 440.0 * 2 ** ((n - 69) / 12)


def _env(n, attack, release):
    e = np.ones(n, np.float32)
    a, r = int(attack * SR), int(release * SR)
    if a:
        e[:a] = np.linspace(0, 1, a) ** 2
    if r:
        e[-r:] *= np.linspace(1, 0, r) ** 2
    return e


def _lowpass(x, cutoff, order=2):
    b, a = signal.butter(order, cutoff / (SR / 2), "low")
    return signal.lfilter(b, a, x).astype(np.float32)


def _bandpass(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], "band")
    return signal.lfilter(b, a, x).astype(np.float32)


def reverb(stereo, decay=2.8, mix=0.35, seed=7):
    """用指数衰减的噪声做卷积混响。"""
    rng = np.random.default_rng(seed)
    n = int(decay * SR)
    env = np.exp(-np.linspace(0, 7, n)).astype(np.float32)
    out = np.zeros_like(stereo)
    for ch in range(2):
        ir = rng.standard_normal(n).astype(np.float32) * env
        ir = _lowpass(ir, 6000)
        ir /= np.sqrt((ir ** 2).sum())
        wet = signal.fftconvolve(stereo[:, ch], ir)[:len(stereo)]
        out[:, ch] = stereo[:, ch] * (1 - mix) + wet * mix
    return out


# ---------- 背景音乐 ----------

CHORDS = {
    "calm": [[57, 64, 71, 72], [53, 60, 64, 69], [48, 55, 64, 67], [55, 62, 66, 71]],      # Am9 Fmaj7 C G
    "wonder": [[53, 60, 64, 69], [55, 62, 67, 71], [57, 64, 67, 72], [52, 59, 64, 67]],    # F G Am Em
}


def _pad_chord(notes, dur, rng, bright=0.5):
    t = _t(dur)
    out = np.zeros((len(t), 2), np.float32)
    for i, n in enumerate(notes):
        f = note_hz(n)
        for det, pan in ((-0.004, 0.25), (0.004, 0.75)):
            ph = rng.random() * 6.28
            w = (np.sin(2 * np.pi * f * (1 + det) * t + ph)
                 + bright * 0.35 * np.sin(2 * np.pi * 2 * f * (1 + det) * t + ph)
                 + bright * 0.12 * np.sin(2 * np.pi * 3 * f * (1 + det) * t))
            lfo = 0.85 + 0.15 * np.sin(2 * np.pi * (0.07 + 0.03 * i) * t + rng.random() * 6)
            out[:, 0] += w * lfo * (1 - pan)
            out[:, 1] += w * lfo * pan
    return out / (len(notes) * 2)


def _bell(freq, dur=2.5):
    t = _t(dur)
    x = (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2.01 * t)
         + 0.12 * np.sin(2 * np.pi * freq * 3.98 * t)) * np.exp(-t * 2.2)
    return (x * _env(len(t), 0.004, 0.2)).astype(np.float32)


def bgm(total, sections, mutes=(), seed=3):
    """sections: [(t0, t1, mood)]，mood 为 calm / wonder / tension。mutes: [(t0, t1)] 静音区。"""
    rng = np.random.default_rng(seed)
    out = np.zeros((int(total * SR) + SR, 2), np.float32)
    chord_len = 6.0
    for t0, t1, mood in sections:
        if mood == "tension":
            t = _t(t1 - t0)
            drone = (np.sin(2 * np.pi * note_hz(33) * t) * 0.9 + np.sin(2 * np.pi * note_hz(40) * t) * 0.5
                     + 0.3 * np.sin(2 * np.pi * note_hz(45) * 1.003 * t))
            drone *= 0.75 + 0.25 * np.sin(2 * np.pi * 0.25 * t)
            noise = _lowpass(rng.standard_normal(len(t)).astype(np.float32), 300) * 0.6
            pulse = np.zeros(len(t), np.float32)
            for k in np.arange(0, t1 - t0, 1.6):           # 心跳般的低频脉冲
                i = int(k * SR)
                seg = _t(0.5)
                p = np.sin(2 * np.pi * 55 * seg) * np.exp(-seg * 9)
                pulse[i:i + len(p)] += p[:len(pulse) - i] * 0.9
            mono = (drone * 0.3 + noise * 0.25 + pulse * 0.5) * _env(len(t), 2.0, 1.5)
            a = int(t0 * SR)
            out[a:a + len(mono)] += np.stack([mono, mono], 1) * 0.9
            continue
        chords = CHORDS[mood]
        k, t = 0, t0
        while t < t1:
            d = min(chord_len + 2.0, t1 - t + 2.0)
            pad = _pad_chord(chords[k % len(chords)], d, rng, bright=0.6 if mood == "wonder" else 0.45)
            pad *= _env(len(pad), 2.0, 2.0)[:, None]
            a = int(t * SR)
            out[a:a + len(pad)] += pad[:len(out) - a] * 0.55
            # 稀疏的铃音琶音
            notes = chords[k % len(chords)]
            for j, bt in enumerate(np.arange(0.4, chord_len, 1.5)):
                if t + bt >= t1:
                    break
                n = notes[(j * 2 + k) % len(notes)] + 12
                b = _bell(note_hz(n)) * 0.22
                bi = int((t + bt) * SR)
                pan = 0.3 + 0.4 * rng.random()
                seg = out[bi:bi + len(b)]
                seg[:, 0] += b[:len(seg)] * (1 - pan)
                seg[:, 1] += b[:len(seg)] * pan
            t += chord_len
            k += 1
    for m0, m1 in mutes:
        a, b = int(m0 * SR), int(m1 * SR)
        fade = int(0.25 * SR)
        g = np.ones(len(out), np.float32)
        g[a:b] = 0
        g[max(0, a - fade):a] = np.linspace(1, 0, min(fade, a))
        g[b:b + fade] = np.linspace(0, 1, len(g[b:b + fade]))
        out *= g[:, None]
    out = reverb(out, decay=3.5, mix=0.4)
    return out[:int(total * SR)]


# ---------- 音效 ----------

def sfx_ding(freq=1318.5):
    t = _t(1.2)
    x = (np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * freq * 2.76 * t) * np.exp(-t * 6)) * np.exp(-t * 4)
    return (x * _env(len(t), 0.002, 0.05) * 0.5).astype(np.float32)


def sfx_whoosh(dur=0.7, seed=1):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    x = rng.standard_normal(n).astype(np.float32)
    x = _bandpass(x, 300, 5000)
    e = np.sin(np.linspace(0, np.pi, n)) ** 2
    return (x * e * 0.35).astype(np.float32)


def sfx_boom(dur=3.5, seed=2):
    rng = np.random.default_rng(seed)
    t = _t(dur)
    f = 70 * np.exp(-t * 1.2) + 28
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t * 1.1)
    noise = _lowpass(rng.standard_normal(len(t)).astype(np.float32), 900) * np.exp(-t * 2.5) * 0.8
    return ((x + noise) * _env(len(t), 0.005, 0.5) * 0.9).astype(np.float32)


def sfx_click(seed=4):
    rng = np.random.default_rng(seed)
    t = _t(0.12)
    x = rng.standard_normal(len(t)) * np.exp(-t * 90) + 0.5 * np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 120)
    return (_bandpass(x.astype(np.float32), 800, 8000) * 0.6).astype(np.float32)


def sfx_swish(seed=5):
    rng = np.random.default_rng(seed)
    n = int(0.35 * SR)
    x = _bandpass(rng.standard_normal(n).astype(np.float32), 1500, 9000)
    return (x * np.linspace(1, 0, n) ** 2 * 0.4).astype(np.float32)


def sfx_shimmer(dur=2.5):
    """空灵的上行音，用于转折。"""
    out = np.zeros(int(dur * SR) + SR * 3, np.float32)
    for i, n in enumerate([76, 79, 83, 88]):
        b = _bell(note_hz(n), 2.5) * 0.25
        a = int(i * 0.12 * SR)
        out[a:a + len(b)] += b
    return out


SFX = {"ding": sfx_ding, "whoosh": sfx_whoosh, "boom": sfx_boom, "click": sfx_click,
       "swish": sfx_swish, "shimmer": sfx_shimmer, "low_ding": lambda: sfx_ding(659.3)}


# ---------- 混音 ----------

def resample(x, sr):
    if sr == SR:
        return x
    from math import gcd
    g = gcd(SR, sr)
    return signal.resample_poly(x, SR // g, sr // g).astype(np.float32)


def place(track, clip, t, gain=1.0, pan=0.5):
    a = int(t * SR)
    if a >= len(track):
        return
    clip = clip[:len(track) - a]
    if clip.ndim == 1:
        track[a:a + len(clip), 0] += clip * gain * (1 - pan) * 2 ** 0.5
        track[a:a + len(clip), 1] += clip * gain * pan * 2 ** 0.5
    else:
        track[a:a + len(clip)] += clip * gain


def duck_gain(voice_mono, depth_db=-9, attack=0.08, release=0.5):
    """根据配音音量生成背景音乐的闪避增益曲线。"""
    hop = int(0.01 * SR)
    n = len(voice_mono) // hop + 1
    pad = np.zeros(n * hop, np.float32)
    pad[:len(voice_mono)] = np.abs(voice_mono)
    env = pad.reshape(n, hop).max(1)
    active = (env > 0.02).astype(np.float32)
    g = np.ones(n, np.float32)
    low = 10 ** (depth_db / 20)
    cur = 1.0
    for i in range(n):
        target = low if active[i] else 1.0
        k = 0.01 / (attack if target < cur else release)
        cur += (target - cur) * min(1.0, k)
        g[i] = cur
    return np.repeat(g, hop)[:len(voice_mono)]
