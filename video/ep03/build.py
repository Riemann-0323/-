"""EP03 一键成片。

用法（在 video/ 目录下）：
    python3 -m ep03.build            # 全流程：配音 → 校对 → 混音 → 渲染 → 合成
    python3 -m ep03.build --preview  # 只渲染若干关键帧 PNG 用于检查
    python3 -m ep03.build --check    # 只做配音和语音识别校对
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import audio, core, render, timeline, tts  # noqa: E402
from ep03.script import SCENES  # noqa: E402

OUT = ROOT / "out" / "ep03"
TITLE = "EP03-你看到的星星还在吗"


def log(*a):
    print(*a, flush=True)


def voice_and_timeline():
    log("① 配音与时间轴")
    spoken, spans, total = timeline.build(SCENES, OUT / "tts_cache", log=log)
    cues = timeline.subtitles(spoken)
    log(f"  总时长 {total:.1f}s，{len(spoken)} 句，{len(cues)} 条字幕")
    return spoken, spans, total, cues


def check(spoken):
    log("② 语音识别校对（SenseVoice 反向转写，比较汉字错误率）")
    rows, worst = [], 0
    for s in spoken:
        hyp = tts.transcribe(s.audio, s.sr)
        e = tts.cer(s.spoken, hyp)
        worst = max(worst, e)
        rows.append({"text": s.text, "tts": s.spoken, "asr": hyp, "cer": round(e, 3)})
        flag = "  ⚠" if e > 0.12 else ""
        log(f"  {e:5.2f}{flag}  {s.spoken}\n         → {hyp}")
    mean = float(np.mean([r["cer"] for r in rows]))
    log(f"  平均字错误率 {mean:.3f}，最高 {worst:.3f}")
    (OUT / "asr_report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return mean


def build_audio(spoken, spans, total, scene_objs):
    log("③ 混音：配音 + 背景音乐 + 音效")
    n = int(total * audio.SR) + audio.SR
    voice = np.zeros((n, 2), np.float32)
    for s in spoken:
        x = audio.resample(s.audio, s.sr)
        x = x / (np.abs(x).max() + 1e-9) * 0.8
        audio.place(voice, x, s.start)
    span = {sp.name: sp for sp in spans}
    sections = [
        (0, span["orion"].start, "calm"),
        (span["orion"].start, span["wave"].end, "tension"),
        (span["andromeda"].start - 0.5, total, "wonder"),
    ]
    sup = span["supernova"]
    mute = (sup.start + sup.lines[1][1] + 0.1, span["wave"].start + 0.6)
    music = audio.bgm(total + 1, sections, mutes=[mute])[:n]
    if len(music) < n:
        music = np.pad(music, ((0, n - len(music)), (0, 0)))
    g = audio.duck_gain(voice[:, 0], depth_db=-8)
    music *= g[:, None]
    fx = np.zeros((n, 2), np.float32)
    for sp, obj in zip(spans, scene_objs):
        for t_local, name, gain in obj.sfx():
            audio.place(fx, audio.SFX[name](), sp.start + t_local, gain)
    mix = voice * 1.0 + music * 0.16 + fx * 0.5
    peak = np.abs(mix).max()
    mix = mix / peak * 0.89
    path = OUT / "audio.wav"
    sf.write(path, mix[:int(total * audio.SR)], audio.SR)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scene", default=None, help="只预览某个场景")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    spoken, spans, total, cues = voice_and_timeline()
    (OUT / f"{TITLE}.srt").write_text(timeline.srt(cues), encoding="utf-8")
    (OUT / "timeline.json").write_text(json.dumps(
        {"total": total, "scenes": [[s.name, round(s.start, 3), round(s.end, 3), s.lines] for s in spans],
         "subtitles": cues}, ensure_ascii=False, indent=1))
    if args.check:
        check(spoken)
        return

    from ep03 import scenes as S
    comp = S.Compositor(spans, cues, total)

    if args.preview:
        import cv2
        pdir = OUT / "preview"
        pdir.mkdir(exist_ok=True)
        for sp in spans:
            if args.scene and sp.name != args.scene:
                continue
            for k, frac in enumerate((0.15, 0.55, 0.92)):
                T = sp.start + (sp.end - sp.start) * frac
                img = comp.frame(int(T * core.FPS))
                cv2.imwrite(str(pdir / f"{sp.name}_{k}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
        log(f"预览帧已保存到 {pdir}")
        return

    check(spoken)
    wav = build_audio(spoken, spans, total, comp.scene_objects())
    log("④ 渲染画面")
    n_frames = int(total * core.FPS)
    video = OUT / "video_only.mp4"
    render.render_frames(comp.frame, n_frames, video, workers=args.workers, log=log)
    log("⑤ 合成")
    final = OUT / f"{TITLE}.mp4"
    render.mux(video, wav, final)
    log(f"完成：{final}（总用时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
