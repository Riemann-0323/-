"""通用的一键成片流程。每一期只需要提供一个配置字典 EP：

    name     输出目录名，如 "ep01"
    title    成片文件名，如 "EP01-你看到的星星还在吗"
    tag      左上角期号角标，如 "反直觉 #01"
    scenes   逐字稿 [(场景名, [Line, ...]), ...]
    classes  {场景名: 场景类}
    music    函数 (spans, total) -> (sections, mutes)
    tones    MiniMax 发音词典，如 ["参宿四/(shen1)(xiu4)(si4)"]

命令行（在 video/ 目录下）：
    python3 -m ep01.build             完整成片
    python3 -m ep01.build --check     只配音并做语音识别校对
    python3 -m ep01.build --preview   每个场景出 3 张预览图
    选项 --voice kokoro|minimax|auto（默认 auto）
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio, core, render, timeline, tts
from . import voice as voices
from .ui import Compositor

ROOT = Path(__file__).resolve().parents[1]


def log(*a):
    print(*a, flush=True)


def check(spoken, out):
    log("② 语音识别校对（SenseVoice 反向转写，比较汉字错误率）")
    rows = []
    for s in spoken:
        hyp = tts.transcribe(s.audio, s.sr)
        e = tts.cer(s.spoken, hyp)
        rows.append({"text": s.text, "reference": s.spoken, "asr": hyp, "cer": round(e, 3)})
        flag = "  ⚠" if e > 0.12 else ""
        log(f"  {e:5.2f}{flag}  {s.text}\n         → {hyp}")
    mean = float(np.mean([r["cer"] for r in rows]))
    log(f"  平均字错误率 {mean:.3f}，最高 {max(r['cer'] for r in rows):.3f}")
    (out / "asr_report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return mean


def build_audio(scene_audio, spans, total, scene_objs, music, out):
    log("③ 混音：配音 + 背景音乐 + 音效")
    n = int(total * audio.SR) + audio.SR
    voice = np.zeros((n, 2), np.float32)
    for t, x, sr in scene_audio:
        audio.place(voice, audio.resample(x, sr), t)
    peak = np.abs(voice).max()
    voice *= 0.85 / max(peak, 1e-6)
    sections, mutes = music(spans, total)
    bgm = audio.bgm(total + 1, sections, mutes=mutes)[:n]
    if len(bgm) < n:
        bgm = np.pad(bgm, ((0, n - len(bgm)), (0, 0)))
    bgm *= audio.duck_gain(voice[:, 0], depth_db=-8)[:, None]
    fx = np.zeros((n, 2), np.float32)
    for sp, obj in zip(spans, scene_objs):
        for t_local, name, gain in obj.sfx():
            audio.place(fx, audio.SFX[name](), sp.start + t_local, gain)
    mix = voice + bgm * 0.16 + fx * 0.5
    mix *= 0.89 / np.abs(mix).max()
    path = out / "audio.wav"
    sf.write(path, mix[:int(total * audio.SR)], audio.SR)
    return path


def run(ep):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--scene", default=None, help="只预览某个场景")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--voice", default="auto", choices=["auto", "minimax", "kokoro"])
    args = ap.parse_args()
    out = ROOT / "out" / ep["name"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("① 配音与时间轴")
    v = voices.choose(args.voice, ep.get("tones", ()), log)
    spoken, spans, total, scene_audio = timeline.build(ep["scenes"], v, out / f"tts_{v.name}", log=log)
    cues = timeline.subtitles(spoken)
    log(f"  总时长 {total:.1f}s，{len(spoken)} 句，{len(cues)} 条字幕")
    (out / f"{ep['title']}.srt").write_text(timeline.srt(cues), encoding="utf-8")
    (out / "timeline.json").write_text(json.dumps(
        {"voice": v.name, "total": total,
         "scenes": [[s.name, round(s.start, 3), round(s.end, 3), s.lines] for s in spans],
         "subtitles": cues}, ensure_ascii=False, indent=1))
    if args.check:
        check(spoken, out)
        return

    comp = Compositor(spans, cues, total, ep["classes"], ep["tag"])
    if args.preview:
        import cv2
        pdir = out / "preview"
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

    check(spoken, out)
    wav = build_audio(scene_audio, spans, total, comp.scene_objects(), ep["music"], out)
    log("④ 渲染画面")
    video = out / "video_only.mp4"
    render.render_frames(comp.frame, int(total * core.FPS), video, workers=args.workers, log=log)
    log("⑤ 合成")
    final = out / f"{ep['title']}.mp4"
    render.mux(video, wav, final)
    log(f"完成：{final}（配音：{v.name}，总用时 {time.time() - t0:.0f}s）")
