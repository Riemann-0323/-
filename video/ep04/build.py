"""第 4 期一键成片。用法（在 video/ 目录下）：python3 -m ep04.build [--check | --preview]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.episode import run  # noqa: E402
from ep04.scenes import SCENE_CLASSES, music  # noqa: E402
from ep04.script import SCENES, TONES  # noqa: E402

EP = {"name": "ep04", "title": "EP04-此刻地球上一定有一个地方没有风", "tag": "反直觉 #04",
      "scenes": SCENES, "classes": SCENE_CLASSES, "music": music, "tones": TONES}

if __name__ == "__main__":
    run(EP)
