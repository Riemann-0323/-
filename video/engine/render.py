"""多进程逐帧渲染，交给 ffmpeg 编码，最后与音频合成 MP4。"""
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

from .core import FPS, H, W

_frame_fn = None


def _encode_chunk(args):
    idx, a, b, path = args
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-g", str(FPS * 2), str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(a, b):
        p.stdin.write(_frame_fn(i).tobytes())
    p.stdin.close()
    p.wait()
    return idx, b - a, time.time() - t0


def render_frames(frame_fn, n_frames, out_path, workers=4, chunks=16, log=print):
    """frame_fn(i) -> (H, W, 3) uint8。用 fork 方式共享已加载的时间轴数据。"""
    global _frame_fn
    _frame_fn = frame_fn
    out_path = Path(out_path)
    seg_dir = out_path.parent / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    step = -(-n_frames // chunks)
    jobs = [(k, k * step, min(n_frames, (k + 1) * step), seg_dir / f"seg{k:03d}.mp4")
            for k in range(chunks) if k * step < n_frames]
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for idx, n, dt in pool.imap_unordered(_encode_chunk, jobs):
            log(f"    片段 {idx + 1}/{len(jobs)}：{n} 帧，{dt:.0f}s（{n / dt:.1f} fps）")
    lst = seg_dir / "list.txt"
    lst.write_text("".join(f"file '{j[3].name}'\n" for j in jobs))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out_path)], check=True)
    log(f"  视频渲染完成：{n_frames} 帧，用时 {time.time() - t0:.0f}s")


def mux(video, audio, out, subtitles_srt=None):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio)]
    cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-shortest", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
