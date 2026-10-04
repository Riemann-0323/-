# video/ · 代码生成完整视频

这里的代码把一期文案直接变成一条可以发布的 **横屏 1080p MP4**，全程不需要剪映、即梦等任何外部工具。

| 环节 | 实现方式 |
|---|---|
| 配音 | MiniMax 语音合成（`speech-2.6-hd`），每个场景整段合成，音色由样音选定；未配置 MiniMax 时退回离线 Kokoro，仅供预览 |
| 对齐 | MiniMax 返回句级时间戳，再吸附到配音的静音处，字幕和画面动画都跟着真实配音时间走 |
| 配音校对 | 离线语音识别 SenseVoice 把配音转回文字，自动比对字错误率，抓多音字、读错的数字 |
| 画面 | Python 程序化绘制：星空、银河、星系、行星、地球风场、甜甜圈环面、剪影、图表、文字动画 |
| 字幕 | 按配音实际停顿自动切分并烧录进画面，同时导出 `.srt` |
| 音乐/音效 | 代码合成的氛围音乐和音效（无版权问题），配音时自动压低音乐 |
| 画面标识 | 左上角只有期号角标「反直觉 #0x」，不放「AI 生成」水印（AI 声明在平台发布时勾选） |

## 目录

```
video/
  engine/            通用引擎（每一期都复用）
    core.py          画布、缓动、文字、形状、辉光、噪声
    cosmos.py        星空、银河、星系、行星、恒星、剪影
    ui.py            场景基类、文字动画、通用夜空、片头片尾、合成器（转场 + 字幕 + 角标）
    episode.py       一键成片流程：配音 → 校对 → 混音 → 预览/渲染 → 合成
    voice.py         配音选择（MiniMax / Kokoro）与句级对齐
    tts_minimax.py   MiniMax 语音合成接口
    tts.py           Kokoro 离线配音、SenseVoice 识别校对
    timeline.py      时间轴与字幕切分
    audio.py         背景音乐、音效、混音
    render.py        多进程渲染 + ffmpeg 编码
    props.py         公交车、站牌、长椅、钟表
    fields.py        球面向量场（风、毛球）
    torus.py         光线步进渲染的甜甜圈环面
  ep01/ … ep05/      第一季五期，每期都有：
    script.py        逐字稿（字幕文字 + 可选的配音文字 + 停顿）和多音字读音表
    scenes.py        各场景画面
    build.py         一键成片
    cover.py         封面（16:9 与 3:4）
  tools/
    voice_samples.py 生成 MiniMax 候选音色样音、选定音色
  voice.json         选定的 MiniMax 音色（选音色后生成）
  setup_assets.sh    下载识别模型、离线配音模型和字体（约 1.2 GB，只需一次）
```

## 准备

需要 Python 3.10+ 和 ffmpeg。

```bash
cd video
pip install -r requirements.txt
bash setup_assets.sh
```

MiniMax 配音需要两个环境变量（在环境设置里配置，不要写进代码或提交到仓库）：

| 变量 | 说明 |
|---|---|
| `MINIMAX_API_KEY` | MiniMax 开放平台（minimaxi.com）的接口密钥 |
| `MINIMAX_GROUP_ID` | 可选，控制台显示了 GroupId 时再加 |

网络需要能访问 `api.minimaxi.com`。可选：`MINIMAX_TTS_MODEL` 换模型，`MINIMAX_API_HOST` 换接口地址。

## 选音色（只做一次）

```bash
python3 -m tools.voice_samples --list            # 列出账号可用的系统音色
python3 -m tools.voice_samples --n 6             # 自动挑 6 个适合解说的中文音色，各生成一段样音
python3 -m tools.voice_samples --voices A,B,C    # 或者指定音色 ID
python3 -m tools.voice_samples --pick <voice_id> # 选定音色，写入 voice.json
```

样音输出在 `out/voice_samples/*.mp3`。

## 生成一期

以第 1 期为例（其他期把 `ep01` 换成 `ep02`…`ep05`）：

```bash
python3 -m ep01.build --check            # 只生成配音并校对
python3 -m ep01.build --preview          # 每个场景出 3 张预览图
python3 -m ep01.build --preview --scene hook   # 只预览某个场景
python3 -m ep01.build                    # 完整成片
python3 -m ep01.cover                    # 封面
```

`--voice minimax|kokoro` 可以强制指定配音来源；默认 `auto`：配好了 MiniMax 密钥和 `voice.json` 就用 MiniMax，否则用 Kokoro 并在日志里提示"仅供预览"。

输出在 `video/out/ep01/`：成片 MP4、字幕 SRT、`timeline.json`、封面、配音校对报告 `asr_report.json`。

## 改文案怎么办

只改 `epXX/script.py` 里对应那一行：
- 第一个参数是字幕显示的文字；
- 第二个参数（可选）是给配音读的文字，比如把年份写成汉字；
- `pause` 是这句之后的停顿秒数；
- 多音字写进文件末尾的 `TONES`，格式如 `"参宿四/(shen1)(xiu4)(si4)"`，MiniMax 会按这个读音读，字幕不变。

重新运行 `build`，时间轴、字幕、画面节奏都会自动跟着配音长度调整。
