"""第 2 期一键成片。用法（在 video/ 目录下）：python3 -m ep02.build [--check | --preview]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.episode import run  # noqa: E402
from ep02.scenes import SCENE_CLASSES, music  # noqa: E402
from ep02.script import SCENES, TONES  # noqa: E402

EP = {"name": "ep02", "title": "EP02-公交车10分钟一班为什么你总要等更久", "tag": "反直觉 #02",
      "scenes": SCENES, "classes": SCENE_CLASSES, "music": music, "tones": TONES}

if __name__ == "__main__":
    run(EP)
