# video/ · 代码生成完整视频

这里的代码把一期文案直接变成一条可以发布的 **横屏 1080p MP4**，全程不需要剪映、即梦等任何外部工具。

| 环节 | 实现方式 |
|---|---|
| 配音 | 离线中文 TTS：Kokoro v1.1-zh（sherpa-onnx 运行），固定男声音色 #62 |
| 配音校对 | 离线语音识别 SenseVoice 把配音转回文字，自动比对字错误率，抓多音字、读错的数字 |
| 画面 | Python 程序化绘制：星空、银河、旋涡星系、行星、恒星、超新星、剪影、图表、文字动画 |
| 字幕 | 按配音实际停顿自动切分并烧录进画面，同时导出 `.srt` |
| 音乐/音效 | 代码合成的氛围音乐和音效（无版权问题），配音时自动压低音乐 |
| AI 标识 | 画面右上角常驻「AI 生成」，片尾有「本视频画面与配音由 AI 生成」 |

## 目录

```
video/
  engine/          通用引擎（以后每一期都复用）
    core.py        画布、缓动、文字、形状、辉光、噪声
    cosmos.py      星空、银河、星系、行星、恒星、剪影
    tts.py         配音合成与语音识别校对
    timeline.py    时间轴与字幕切分
    audio.py       背景音乐、音效、混音
    render.py      多进程渲染 + ffmpeg 编码
  ep03/            第三期：你看到的星星还在吗
    script.py      逐字稿（字幕文字 + 配音文字 + 停顿）
    scenes.py      24 个场景的画面
    build.py       一键成片
    cover.py       封面（16:9 与 3:4）
  setup_assets.sh  下载模型和字体（约 1.2 GB，只需一次）
```

## 自己重新生成

需要 Python 3.10+ 和 ffmpeg。

```bash
cd video
pip install -r requirements.txt
bash setup_assets.sh
python3 -m ep03.build --check     # 只生成配音并校对（约 1 分钟）
python3 -m ep03.build --preview   # 每个场景出 3 张预览图
python3 -m ep03.build             # 完整成片（4 核约 10 分钟）
python3 -m ep03.cover             # 封面
```

输出在 `video/out/ep03/`：成片 MP4、字幕 SRT、封面、配音校对报告 `asr_report.json`。

## 改文案怎么办

只改 `ep03/script.py` 里对应那一行：
- `text` 是字幕显示的文字；
- `tts` 是给配音模型读的文字。数字写成汉字，多音字用同音字替换，比如"参宿四"写成"深秀四"、"长得惊人"写成"常得惊人"；
- `pause` 是这句之后的停顿秒数。

重新运行 `build`，时间轴、字幕、画面节奏都会自动跟着配音长度调整。
