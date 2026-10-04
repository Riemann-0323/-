"""MiniMax 音色试听与选定。

    python3 -m tools.voice_samples --list                 列出全部系统音色
    python3 -m tools.voice_samples                        自动挑 6 个适合科普解说的中文音色，生成样音
    python3 -m tools.voice_samples --voices A,B,C         指定音色生成样音
    python3 -m tools.voice_samples --pick <voice_id> [--speed 1.1]   选定音色，写入 video/voice.json
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import tts_minimax  # noqa: E402
from engine.voice import VOICE_CONFIG  # noqa: E402

OUT = ROOT / "out" / "voice_samples"

SAMPLE = ("你今晚看到的星星，可能早就不存在了。这句话，你一定听过。<#0.4#>可惜，它只对了一半。"
          "这里是反直觉研究所，我是所长。今天我们来拆一句被转发了无数次的浪漫科普。"
          "参宿四离我们五百多光年，它迟早会变成超新星，亮得像半个月亮，白天都能看见。")
TONES = ["参宿四/(shen1)(xiu4)(si4)"]

KEYWORDS = re.compile(r"narrat|announc|anchor|news|documentar|reliable|steady|calm|sincere|gentle|magnetic|"
                      r"解说|播音|主持|纪录|沉稳|磁性|知性|讲述|旁白", re.I)
FALLBACK = ["Chinese (Mandarin)_Reliable_Executive", "Chinese (Mandarin)_Male_Announcer",
            "Chinese (Mandarin)_News_Anchor", "Chinese (Mandarin)_Gentleman", "presenter_male",
            "male-qn-jingying", "audiobook_male_1", "presenter_female"]


def pick_candidates(voices, n):
    zh = [v for v in voices if v["voice_id"] and (v["voice_id"].startswith("Chinese (Mandarin)")
                                                  or re.search(r"[一-鿿]", v["name"] + v["description"])
                                                  or v["voice_id"] in FALLBACK)]
    scored = sorted(zh, key=lambda v: -len(KEYWORDS.findall(v["voice_id"] + " " + v["name"] + " " + v["description"])))
    chosen = [v for v in scored if KEYWORDS.search(v["voice_id"] + v["name"] + v["description"])][:n]
    ids = {v["voice_id"] for v in chosen}
    for f in FALLBACK:
        if len(chosen) >= n:
            break
        hit = next((v for v in voices if v["voice_id"] == f), None)
        if hit and f not in ids:
            chosen.append(hit)
            ids.add(f)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--voices", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--speed", type=float, default=1.1)
    ap.add_argument("--pick", default=None)
    args = ap.parse_args()

    if args.pick:
        cfg = {"voice_id": args.pick, "speed": args.speed}
        VOICE_CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
        print(f"已选定音色：{args.pick}（语速 {args.speed}），写入 {VOICE_CONFIG}")
        return

    if not tts_minimax.available():
        raise SystemExit("缺少 MINIMAX_API_KEY 环境变量。")

    try:
        voices = tts_minimax.list_voices()
    except Exception as e:
        print(f"⚠ 获取音色列表失败（{e}），使用内置候选。")
        voices = [{"voice_id": v, "name": "", "description": ""} for v in FALLBACK]

    if args.list:
        for v in voices:
            print(f"{v['voice_id']:45s} {v['name']:16s} {v['description'][:80]}")
        print(f"共 {len(voices)} 个")
        return

    if args.voices:
        ids = [s.strip() for s in args.voices.split(",") if s.strip()]
        chosen = [next((v for v in voices if v["voice_id"] == i), {"voice_id": i, "name": "", "description": ""})
                  for i in ids]
    else:
        chosen = pick_candidates(voices, args.n)

    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for k, v in enumerate(chosen, 1):
        vid = v["voice_id"]
        try:
            x, sr, _ = tts_minimax.synthesize(SAMPLE, vid, OUT / "cache", args.speed, TONES, subtitles=False)
        except Exception as e:
            print(f"  ✗ {vid}：{e}")
            continue
        safe = re.sub(r"[^\w\-]+", "_", vid).strip("_")
        mp3 = OUT / f"{k:02d}_{safe}.mp3"
        import soundfile as sf
        wav = OUT / "tmp.wav"
        sf.write(wav, x, sr)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-b:a", "128k", str(mp3)], check=True)
        wav.unlink()
        index.append(f"{k:02d}. {vid}  {v['name']}  {v['description'][:60]}  → {mp3.name}")
        print(f"  ✓ {index[-1]}")
    (OUT / "README.txt").write_text("\n".join(index), encoding="utf-8")
    print(f"样音已保存到 {OUT}")


if __name__ == "__main__":
    main()
