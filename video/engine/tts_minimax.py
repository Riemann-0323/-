"""MiniMax 语音合成（国内版 api.minimaxi.com，模型 speech-2.6-hd）。

需要的环境变量：
    MINIMAX_API_KEY    必填
    MINIMAX_GROUP_ID   选填（控制台显示了 Group ID 就填上）
    MINIMAX_API_HOST   选填，默认 https://api.minimaxi.com
    MINIMAX_TTS_MODEL  选填，默认 speech-2.6-hd
"""
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 32000


def available():
    return bool(os.environ.get("MINIMAX_API_KEY"))


def _host():
    return os.environ.get("MINIMAX_API_HOST", "https://api.minimaxi.com").rstrip("/")


def model():
    return os.environ.get("MINIMAX_TTS_MODEL", "speech-2.6-hd")


def _request(url, payload=None, timeout=180):
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        headers["Authorization"] = f"Bearer {os.environ['MINIMAX_API_KEY']}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def post(path, payload, retries=4):
    url = _host() + path
    gid = os.environ.get("MINIMAX_GROUP_ID")
    if gid:
        url += ("&" if "?" in url else "?") + "GroupId=" + gid
    last = None
    for attempt in range(retries):
        try:
            resp = json.loads(_request(url, payload))
            base = resp.get("base_resp") or {}
            code = base.get("status_code", 0)
            if code == 0:
                return resp
            msg = f"MiniMax 返回错误 {code}：{base.get('status_msg')}"
            if code in (1004, 2049, 2013):      # 鉴权失败 / Key 无效 / 参数错误：重试没有意义
                raise RuntimeError(msg)
            last = RuntimeError(msg)
        except urllib.error.HTTPError as e:
            last = RuntimeError(f"MiniMax HTTP {e.code}：{e.read()[:300]!r}")
            if e.code in (401, 403):
                raise last
        except (urllib.error.URLError, TimeoutError) as e:
            last = RuntimeError(f"无法连接 {_host()}：{e}（检查环境的网络白名单）")
        time.sleep(2 ** (attempt + 1))
    raise last


def decode_mp3(raw):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE),
                        "pipe:1"], input=raw, capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).copy()


def parse_subtitles(raw):
    """把 MiniMax 的字幕 JSON 统一成 [(文字, 开始秒, 结束秒)]。"""
    items = raw
    if isinstance(raw, dict):
        for k in ("subtitles", "data", "sentences", "list"):
            if isinstance(raw.get(k), list):
                items = raw[k]
                break
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        text = it.get("text") or it.get("sentence") or ""
        if "time_begin" in it:
            a, b = float(it["time_begin"]) / 1000, float(it["time_end"]) / 1000
        elif "begin_time" in it:
            a, b = float(it["begin_time"]) / 1000, float(it["end_time"]) / 1000
        elif "start" in it:
            a, b = float(it["start"]), float(it["end"])
            if b > 1000:          # 毫秒
                a, b = a / 1000, b / 1000
        else:
            continue
        out.append((text, a, b))
    return out


def synthesize(text, voice_id, cache_dir, speed=1.1, tones=(), emotion=None, subtitles=True):
    """合成一段文字。返回 (samples, sample_rate, subtitles 或 None)。结果按内容缓存。"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(json.dumps([model(), voice_id, speed, text, list(tones), emotion]).encode()).hexdigest()[:16]
    wav, meta = cache_dir / f"{key}.wav", cache_dir / f"{key}.json"
    if wav.exists() and meta.exists():
        x, sr = sf.read(wav, dtype="float32")
        subs = json.loads(meta.read_text()).get("subtitles")
        return x, sr, ([tuple(s) for s in subs] if subs is not None else None)
    voice = {"voice_id": voice_id, "speed": speed, "vol": 1.0, "pitch": 0}
    if emotion:
        voice["emotion"] = emotion
    payload = {
        "model": model(), "text": text, "stream": False, "voice_setting": voice,
        "audio_setting": {"sample_rate": SAMPLE_RATE, "bitrate": 128000, "format": "mp3", "channel": 1},
        "language_boost": "Chinese", "output_format": "hex", "subtitle_enable": bool(subtitles),
    }
    if tones:
        payload["pronunciation_dict"] = {"tone": list(tones)}
    resp = post("/v1/t2a_v2", payload)
    data = resp.get("data") or {}
    x = decode_mp3(bytes.fromhex(data["audio"]))
    subs = None
    url = data.get("subtitle_file")
    if subtitles and url:
        try:
            subs = parse_subtitles(json.loads(_request(url)))
        except Exception as e:  # 字幕文件所在域名可能不在网络白名单里，交给上层按句合成兜底
            print(f"  ⚠ 字幕时间戳下载失败（{e}），改为逐句合成对齐")
    sf.write(wav, x, SAMPLE_RATE)
    meta.write_text(json.dumps({"text": text, "voice_id": voice_id, "speed": speed, "subtitles": subs},
                               ensure_ascii=False))
    return x, SAMPLE_RATE, subs


def list_voices(voice_type="system"):
    resp = post("/v1/get_voice", {"voice_type": voice_type})
    out = []
    for k in ("system_voice", "voice_cloning", "voice_generation"):
        for v in resp.get(k) or []:
            out.append({"voice_id": v.get("voice_id"), "name": v.get("voice_name", ""),
                        "description": " ".join(v.get("description") or []) if isinstance(v.get("description"), list)
                        else (v.get("description") or "")})
    return out
