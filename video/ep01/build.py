"""第 1 期一键成片。用法（在 video/ 目录下）：python3 -m ep01.build [--check | --preview]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.episode import run  # noqa: E402
from ep01.scenes import SCENE_CLASSES, music  # noqa: E402
from ep01.script import SCENES, TONES  # noqa: E402

EP = {"name": "ep01", "title": "EP01-你看到的星星还在吗", "tag": "反直觉 #01",
      "scenes": SCENES, "classes": SCENE_CLASSES, "music": music, "tones": TONES}

if __name__ == "__main__":
    run(EP)
