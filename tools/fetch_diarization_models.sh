#!/usr/bin/env bash
# 用途: 下载说话人分离用的两个 ONNX 模型（pyannote 分割 + 3D-Speaker 声纹）到 ./models
# 用法: ./tools/fetch_diarization_models.sh [目标目录]
# 入参: $1 目标目录（默认 <项目>/models）
# 出参: 目录下 segmentation.onnx / embedding.onnx；echo 进度；退出码 0 成功 / 1 失败
# 依赖: curl；网络（优先 hf-mirror 镜像，GitHub releases 在国内常被墙）
# 关键词: 说话人分离, 模型下载, diarization, pyannote, segmentation, 3dspeaker, embedding, sherpa-onnx, hf-mirror
# 更新: 2026-10-16
set -euo pipefail

DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/models}"
mkdir -p "$DIR"

SEG_URLS=(
  "https://hf-mirror.com/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx"
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMB_URLS=(
  "https://hf-mirror.com/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)

# 期望大小（字节）：下完必须对上，否则就是被截断了
SEG_SIZE=5992913
EMB_SIZE=39593761

fetch() {
  local out="$1" want="$2"; shift 2
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$want" ]; then
    echo "已存在且大小正确：$out"; return 0
  fi
  for url in "$@"; do
    echo "下载 $url"
    if curl -L --fail --retry 2 --max-time 900 -o "$out.part" "$url" 2>/dev/null; then
      local size; size=$(stat -c%s "$out.part")
      if [ "$size" = "$want" ]; then mv "$out.part" "$out"; echo "  ✓ $(basename "$out") ($size 字节)"; return 0; fi
      echo "  ✗ 大小不对（$size，应为 $want），换个源重试"; rm -f "$out.part"
    else
      echo "  ✗ 失败"; rm -f "$out.part"
    fi
  done
  return 1
}

fetch "$DIR/segmentation.onnx" "$SEG_SIZE" "${SEG_URLS[@]}"
fetch "$DIR/embedding.onnx"    "$EMB_SIZE" "${EMB_URLS[@]}"
echo "完成。把这两份模型放在 $DIR，服务端会按需加载（DOUBAO_DIAR_DIR 可改路径）。"
