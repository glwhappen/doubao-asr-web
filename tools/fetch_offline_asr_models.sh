#!/usr/bin/env bash
# 用途: 下载「离线识别引擎」要用的字节跳动 ASR 模型（公开 CDN，无需鉴权），解包到 models/offline-asr/
# 用法: ./tools/fetch_offline_asr_models.sh [--model <名字>] [--sdk-version <面板版本>] [--list] [--force] [--help]
# 入参: --model      模型名（默认 lv_asr_support，是唯一一份「明文 ONNX」的完整 ASR 流水线）
#       --sdk-version 面板版本，越大条目越多（默认 25.0.0，上限就是它，约 1128 个模型）
#       --list       只列出语音相关模型（名字/大小/链接），不下载
#       --force      目标已存在也重新下载
# 出参: models/offline-asr/ 下的 encoder/decoder/classifier/punc onnx + token 词表 + strategy.json
# 依赖: curl、unzip、python3（只用标准库）
# 关键词: 离线识别, 离线引擎, 端侧, 豆包, 豆包输入法, 字节, ByteDance, SAMI, Flute, ONNX, onnxruntime,
#         ASR, 语音识别, 语音转文字, 模型下载, 模型分发, effect.snssdk.com, arithmetics, 算法列表,
#         lv_asr_support, lv_grow_voice_local_asr, encoder, decoder, classifier, punc, token, 模型拉取
# 更新: 2026-10-17
#
# 背景（怎么找到这个接口的，别重复踩）：
#   豆包输入法 macOS 版走 TTNet 请求 https://effect.snssdk.com/model/api/arithmetics
#   （TTNet 服务名 com.wave.offline-model.metainfo-service），参数必须有 sdk_version（小写下划线），
#   只能 GET（POST 返回 404）。返回值里 data.arithmetics.cakeCook[] 每项带 file_url.url_list 直链。
#   ld_asr_support 是 zip，解开就是明文 ONNX；同目录的 lv_grow_voice_local_asr 是 7z 套一个
#   SAMI 私有容器（加密载荷），需要闭源 Flute 引擎，本脚本不取那个。
set -euo pipefail

MODEL="lv_asr_support"
SDK_VERSION="25.0.0"
LIST_ONLY=0
FORCE=0
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models/offline-asr"
BASE="https://effect.snssdk.com/model/api/arithmetics"
UA="DoubaoIME/1.0.2 (Macintosh; Intel Mac OS X 10_15_7)"

usage() { sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="${2:?--model 需要值}"; shift 2 ;;
    --sdk-version) SDK_VERSION="${2:?--sdk-version 需要值}"; shift 2 ;;
    --list) LIST_ONLY=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage ;;
    *) echo "未知参数：$1（用 --help 看用法）" >&2; exit 2 ;;
  esac
done

fetch_index() {
  python3 - "$BASE" "$SDK_VERSION" "$UA" <<'PY'
import json, sys, urllib.parse, urllib.request
base, ver, ua = sys.argv[1], sys.argv[2], sys.argv[3]
url = base + "?" + urllib.parse.urlencode({"sdk_version": ver})
req = urllib.request.Request(url, headers={"User-Agent": ua})
with urllib.request.urlopen(req, timeout=60) as f:
    data = json.loads(f.read().decode())
if data.get("status_code") != 0:
    sys.exit(f"接口返回异常：{data}")
out = {}
for group, items in (data.get("data") or {}).get("arithmetics", {}).items():
    if isinstance(items, list):
        for it in items:
            out[it["name"]] = it
json.dump(out, open("/tmp/.offline-asr-index.json", "w"))
print(f"面板 sdk_version={ver} 共 {len(out)} 个模型", file=sys.stderr)
PY
  INDEX=/tmp/.offline-asr-index.json
}

fetch_index
INDEX=/tmp/.offline-asr-index.json

if [ "$LIST_ONLY" = "1" ]; then
  python3 - "$INDEX" <<'PY'
import json, re, sys
idx = json.load(open(sys.argv[1]))
pat = re.compile(r"asr|voice|speech|denoise|kws", re.I)
rows = [(n, it) for n, it in idx.items() if pat.search(n)]
print(f"语音相关模型 {len(rows)} 个：")
for n, it in sorted(rows):
    urls = (it.get("file_url") or {}).get("url_list") or [""]
    print(f"  {n}@{it.get('version'):8s} {urls[0]}")
PY
  exit 0
fi

python3 - "$INDEX" "$MODEL" <<'PY' > /tmp/.offline-asr-pick.txt
import json, sys
idx = json.load(open(sys.argv[1]))
name = sys.argv[2]
if name not in idx:
    sys.exit(f"面板里没有模型 {name}（用 --list 看有哪些）")
it = idx[name]
fu = it.get("file_url") or {}
urls = fu.get("zip_url_list") or fu.get("url_list") or []
if not urls:
    sys.exit(f"{name} 没有下载链接")
print(urls[0]); print(name.replace("/", "_") + "_v" + str(it["version"]))
PY
URL="$(sed -n 1p /tmp/.offline-asr-pick.txt)"
STEM="$(sed -n 2p /tmp/.offline-asr-pick.txt)"

if [ -f "$DEST/asr-model-encoder.onnx" ] && [ "$FORCE" != "1" ]; then
  echo "已存在 $DEST/asr-model-encoder.onnx（要重下加 --force）"; exit 0
fi

mkdir -p "$DEST"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
echo "下载 $MODEL ← $URL"
curl -fL --retry 3 --connect-timeout 20 -A "$UA" -o "$TMP/pkg.zip" "$URL"
echo "下载完成：$(du -h "$TMP/pkg.zip" | cut -f1)"

# zip 里可能是 <名字>.model（SAMI 私有容器）或一堆 onnx；两种情况都给用户明确交代
if unzip -l "$TMP/pkg.zip" | grep -qi '\.onnx'; then
  unzip -q -o "$TMP/pkg.zip" -x '__MACOSX/*' -d "$TMP/x"
  # 平铺到目标目录（zip 里常带一层目录）
  find "$TMP/x" -type f ! -name '._*' -exec cp -f {} "$DEST/" \;
  echo "已解包到 $DEST："
  ls -la "$DEST" | awk 'NR>3 {printf "  %8.1fMB  %s\n", $5/1048576, $9}'
else
  cp -f "$TMP/pkg.zip" "$DEST/$STEM.zip"
  echo "注意：$MODEL 是 SAMI 私有容器（加密载荷），已存到 $DEST/$STEM.zip。"
  echo "      这种需要闭源 Flute/SAMI 引擎，本项目的离线引擎用不了；请用 --model lv_asr_support。"
fi
