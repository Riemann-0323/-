#!/usr/bin/env bash
# 下载成片所需的离线模型与字体到 video/.assets（约 1.2 GB，只需运行一次）
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .assets/fonts
cd .assets

REL=https://github.com/k2-fsa/sherpa-onnx/releases/download

if [ ! -d kokoro-multi-lang-v1_1 ]; then
  echo "下载配音模型 Kokoro v1.1-zh ..."
  curl -sSLO "$REL/tts-models/kokoro-multi-lang-v1_1.tar.bz2"
  tar xjf kokoro-multi-lang-v1_1.tar.bz2 && rm kokoro-multi-lang-v1_1.tar.bz2
fi

if [ ! -d sense-voice ]; then
  echo "下载语音识别模型 SenseVoice（用于配音校对）..."
  curl -sSLO "$REL/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
  tar xjf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  mv sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 sense-voice
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
fi

cd fonts
if [ ! -f SmileySans-Oblique.otf ]; then
  echo "下载字体：得意黑 ..."
  curl -sSL -o smiley.zip https://github.com/atelier-anchor/smiley-sans/releases/download/v2.0.1/smiley-sans-v2.0.1.zip
  python3 -c "import zipfile; zipfile.ZipFile('smiley.zip').extract('SmileySans-Oblique.otf', '.')"
  rm smiley.zip
fi
if [ ! -f SourceHanSansSC-Bold.otf ]; then
  echo "下载字体：思源黑体 ..."
  curl -sSL -o shs.zip https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansSC.zip
  python3 - <<'PY'
import zipfile
z = zipfile.ZipFile("shs.zip")
for w in ("Regular", "Medium", "Bold", "Heavy"):
    data = z.read(f"OTF/SimplifiedChinese/SourceHanSansSC-{w}.otf")
    open(f"SourceHanSansSC-{w}.otf", "wb").write(data)
PY
  rm shs.zip
fi
echo "素材准备完成。"
