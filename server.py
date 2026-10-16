"""豆包输入法语音识别 · 网页实时转写 Demo

用途: 浏览器录音 → 服务端通过豆包输入法（PC 版）同款接口做流式识别 → 实时回显带时间戳的文字。
      链路: log.snssdk.com 匿名设备注册 → is.snssdk.com settings 拿设备级 app_key
            → wss://frontier-audio-ime-ws.doubao.com/ocean/api/v1/ws（protobuf 帧，送 raw PCM16）
用法: uvicorn server:app --host 172.17.0.1 --port 4416   （或 systemctl start doubao-asr-web）
入参: 无。环境变量可选 — DOUBAO_AID / DOUBAO_CREDS / DOUBAO_MAX_SESSIONS。
出参: GET / 录音页面；GET /health 健康检查；WS /ws 事件流：
      {type:status|update|final|error, lines:[{text,start,end,ts,final}], text, message}
      update 每次带全量 lines（前端整块重渲染），所以历史句不会丢。
依赖: fastapi, uvicorn, websockets（见 requirements.txt）
关键词: 豆包输入法, 豆包语音, ASR, 语音转文字, 实时转写, 时间戳, 字幕, SRT, 导出, 流式识别,
        网页录音, doubao ime, frontier-audio-ime, snssdk, protobuf, PCM16, 16k, 麦克风, 听写,
        纠正, 纠错, 词典, 热词, hotword, 上下文, context, 多遍解码, threepass, 个性化
更新: 2026-10-16
"""
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import time
import uuid
from pathlib import Path

import urllib.parse
import urllib.request

import httpx
import websockets
from diarization import FILL_GAP_SECONDS, DIARIZER, SpeakerTracker, assign_speakers
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

log = logging.getLogger("doubao-asr")
logging.basicConfig(level=os.environ.get("DOUBAO_LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
CREDS_PATH = Path(os.environ.get("DOUBAO_CREDS", BASE / ".creds.json"))

AID = int(os.environ.get("DOUBAO_AID", "685343"))          # 取自 ImeService.exe
APP_NAME = "oime"
VERSION_CODE = 100102018
VERSION_NAME = "1.1.2"
UA = ("com.bytedance.android.doubaoime/100102018 (Linux; U; Android 16; en_US; "
      "Pixel 7 Pro; Build/BP2A.250605.031.A2)")
REGISTER_URL = "https://log.snssdk.com/service/2/device_register/"
SETTINGS_URL = "https://is.snssdk.com/service/settings/v3/"
WS_URL = os.environ.get("DOUBAO_CLOUD_WS", "wss://frontier-audio-ime-ws.doubao.com/ocean/api/v1/ws")

# 本地兜底引擎（复用本机 FunASR asr-engine）：云端不可用/报错时自动接管
LOCAL_ENGINE_URL = os.environ.get("DOUBAO_LOCAL_ENGINE", "").rstrip("/")
LOCAL_ENGINE_TOKEN = os.environ.get("DOUBAO_LOCAL_ENGINE_TOKEN", "")
LOCAL_CHUNK = 3200                 # 100ms 一批发给本地引擎
USE_CONTEXT = os.environ.get("DOUBAO_CONTEXT", "1") != "0"   # 是否把热词上下文带给上游

SAMPLE_RATE = 16000
FRAME_BYTES = 640                  # 20ms / 16kHz / mono / PCM16
HISTORY_BYTES = 5 * 60 * SAMPLE_RATE * 2   # 降级补喂用的 PCM 缓冲上限（5 分钟）
MAX_SESSIONS = int(os.environ.get("DOUBAO_MAX_SESSIONS", "3"))
CLOUD_MAX_RETRY = int(os.environ.get("DOUBAO_CLOUD_RETRY", "2"))   # 云端抖动时的重连次数
DEBUG_WIRE = os.environ.get("DOUBAO_DEBUG_WIRE", "0") == "1"
# 说话人标签怎么算：online=每句定稿立刻算（实时出标签，默认）；batch=收尾整段算；both=先实时再整段校正
DIAR_MODE = os.environ.get("DOUBAO_DIAR_MODE", "online").lower()
GITHUB_URL = os.environ.get("DOUBAO_GITHUB_URL", "https://github.com/glwhappen/doubao-asr-web")
TOKEN_TTL = 12 * 3600
PUNCT = "。！？!?；;…\n"

app = FastAPI(title="豆包输入法语音转写 Demo")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
slots = asyncio.Semaphore(MAX_SESSIONS)


# ---------------------------------------------------------------- 凭据（设备级）

def _post_json(url, params, body, headers):
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _device_register():
    """匿名注册一台设备，拿到 device_id（不需要任何账号）。"""
    cdid, openudid, clientudid = str(uuid.uuid4()), secrets.token_hex(8), str(uuid.uuid4())
    header = {
        "device_id": 0, "install_id": 0, "aid": AID, "app_name": APP_NAME,
        "version_code": VERSION_CODE, "version_name": VERSION_NAME,
        "manifest_version_code": VERSION_CODE, "update_version_code": VERSION_CODE,
        "channel": "official", "device_platform": "android", "os": "android", "os_api": "34",
        "os_version": "16", "device_type": "Pixel 7 Pro", "device_brand": "google",
        "device_model": "Pixel 7 Pro", "resolution": "1080*2400", "dpi": "420",
        "language": "zh", "timezone": 8, "access": "wifi", "region": "CN",
        "tz_name": "Asia/Shanghai", "tz_offset": 28800, "openudid": openudid,
        "clientudid": clientudid, "cdid": cdid,
    }
    body = json.dumps({"magic_tag": "ss_app_log", "header": header,
                       "_gen_time": int(time.time() * 1000)}).encode()
    params = {
        "device_platform": "android", "os": "android", "ssmix": "a",
        "_rticket": str(int(time.time() * 1000)), "cdid": cdid, "channel": "official",
        "aid": str(AID), "app_name": APP_NAME, "version_code": str(VERSION_CODE),
        "version_name": VERSION_NAME, "manifest_version_code": str(VERSION_CODE),
        "update_version_code": str(VERSION_CODE), "resolution": "1080*2400", "dpi": "420",
        "device_type": "Pixel 7 Pro", "device_brand": "google", "language": "zh",
        "os_api": "34", "os_version": "16", "ac": "wifi",
    }
    out = _post_json(REGISTER_URL, params, body,
                     {"Content-Type": "application/json", "User-Agent": UA})
    did = str(out.get("device_id_str") or out.get("device_id") or "")
    if not did or did == "0":
        raise RuntimeError("设备注册失败：响应中没有 device_id")
    return did, cdid


def _fetch_app_key(device_id, cdid):
    """settings 接口下发该设备可用的 ASR app_key（这就是识别用的 token）。"""
    body = b"body=null"
    params = {
        "device_platform": "android", "os": "android", "ssmix": "a", "channel": "official",
        "aid": str(AID), "app_name": APP_NAME, "version_code": str(VERSION_CODE),
        "version_name": VERSION_NAME, "device_id": device_id, "cdid": cdid,
        "_rticket": str(int(time.time() * 1000)),
    }
    out = _post_json(SETTINGS_URL, params, body, {
        "User-Agent": UA,
        "x-ss-stub": hashlib.md5(body).hexdigest().upper(),
        "Content-Type": "application/x-www-form-urlencoded",
    })
    key = (out.get("data", {}).get("settings", {}).get("asr_config", {}) or {}).get("app_key")
    if not key:
        raise RuntimeError("settings 接口未返回 asr_config.app_key")
    return key


class Credentials:
    """设备凭据缓存：没有就注册，app_key 过期就刷新。"""

    def __init__(self):
        self.data = {}
        self.lock = asyncio.Lock()
        self._load()

    def _load(self):
        try:
            self.data = json.loads(CREDS_PATH.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}

    def _save(self):
        try:
            CREDS_PATH.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            CREDS_PATH.chmod(0o600)
        except Exception:
            log.warning("凭据写入失败（不影响运行）", exc_info=True)

    async def get(self, force=False):
        async with self.lock:
            fresh = (self.data.get("app_key")
                     and time.time() - self.data.get("updated_at", 0) < TOKEN_TTL)
            if fresh and not force:
                return self.data
            if force or not self.data.get("device_id"):
                did, cdid = await asyncio.to_thread(_device_register)
                self.data.update(device_id=did, cdid=cdid)
            key = await asyncio.to_thread(_fetch_app_key, self.data["device_id"], self.data["cdid"])
            self.data.update(app_key=key, updated_at=time.time())
            self._save()
            log.info("凭据就绪 device_id=%s", self.data["device_id"])
            return self.data


CREDS = Credentials()


# ---------------------------------------------------------------- 纠正词典 / 热词

HOTWORD_LIMIT = int(os.environ.get("DOUBAO_HOTWORD_LIMIT", "50"))


class Lexicon:
    """纠正词典 / 热词，做两件事：

    1. 每句识别结果先过一遍本地替换（对应豆包输入法的 asr_user_correct_dict.dat）
    2. 把改后词当热词通过会话 context 发给上游（对应 AsrContextBuilder 的 hotwords）

    词库由**浏览器本地**保存（每人一份、互不覆盖）：服务端在每个会话开始时从客户端收一份，
    用完即弃，服务端不保存任何词库。
    """

    def __init__(self):
        self.data = {"corrections": [], "hotwords": []}
        self.lock = asyncio.Lock()

    @classmethod
    def from_payload(cls, payload):
        """用客户端传来的词库建一个会话级实例（不落盘）。"""
        lex = cls()
        for item in (payload.get("corrections") or []):
            wrong = str(item.get("wrong") or "").strip()
            right = str(item.get("right") or "").strip()
            if wrong and right and wrong != right:
                lex.data["corrections"].append({
                    "wrong": wrong, "right": right,
                    "phrase": cls.corrected_phrase(wrong, right),
                    "count": int(item.get("count") or 1),
                })
        lex.data["hotwords"] = [str(w).strip() for w in (payload.get("hotwords") or []) if str(w).strip()]
        return lex

    @staticmethod
    def corrected_phrase(wrong, right):
        """从「原句 → 改后句」里抽出真正被改掉的那个词，用它当热词才准。

        例如「我用微信语音接口。」→「我用微信语音识别接口。」抽出「识别」；
        「接口测试」→「接口验证」抽出「验证」。长度不在 2~12 字就不要（整句改写学不到词）。
        """
        i = 0
        while i < len(wrong) and i < len(right) and wrong[i] == right[i]:
            i += 1
        j = 0
        while (j < len(wrong) - i and j < len(right) - i
               and wrong[-1 - j] == right[-1 - j]):
            j += 1
        phrase = right[i:len(right) - j]
        return phrase if 2 <= len(phrase) <= 12 else None

    def hotwords(self):
        """热词 = 手加的 + 每次纠正里被改掉的那个词。"""
        words = list(self.data["hotwords"])
        for c in self.data["corrections"]:
            phrase = self.corrected_phrase(c["wrong"], c["right"])
            if phrase:
                words.append(phrase)
        seen, out = set(), []
        for w in words:
            w = (w or "").strip().strip("。！？!?，,、；;")
            if len(w) >= 2 and w not in seen:
                seen.add(w)
                out.append(w)
        return sorted(out, key=len, reverse=True)[:HOTWORD_LIMIT]

    def apply(self, text):
        """把学到的纠正对应用到一句文本上，返回 (纠正后文本, 命中的 [原文, 改后] 列表)。"""
        hits = []
        for item in sorted(self.data["corrections"], key=lambda c: len(c["wrong"]), reverse=True):
            wrong, right = item["wrong"], item["right"]
            if wrong in text:
                text = text.replace(wrong, right)
                hits.append([wrong, right])
        return text, hits




def build_context(app_name, hotwords):
    """拼会话上下文（base64）：chat + ime_info + hotwords。

    格式来自公开逆向（voxgate 的 buildContextInfo）+ 本机二进制字符串里的字段名，
    实测上游对 hotwords 的两种写法（字符串数组 / 对象数组）都能接受。
    """
    info = {
        "chat": [],
        "ime_info": {"app_apk_name": app_name, "input_type": ""},
    }
    if hotwords:
        info["hotwords"] = [{"word": w, "class": "user", "frequency": 1} for w in hotwords]
    return base64.b64encode(json.dumps(info, ensure_ascii=False).encode()).decode()


# ---------------------------------------------------------------- 文本 → 带时间戳的句子

def _align_times(text, words):
    """把上游的逐字时间铺到 text 的每个字符上。

    上游 words 是「不含标点」的逐字列表，且数量可能少于 text 的汉字数，
    所以先按非标点字符逐一对应，对不上就按顺序退化对应，标点则继承前一字的时间。
    返回 [(start, end), ...]，长度与 text 相同。
    """
    times = [None] * len(text)
    if not words or not text:
        return times

    # 展开成逐字：(字, start, end)
    flat = [(ch, w.get("start_time"), w.get("end_time"))
            for w in words for ch in (w.get("word") or "")]
    positions = [i for i, ch in enumerate(text) if ch not in PUNCT]
    for k, pos in enumerate(positions):
        if k >= len(flat):
            break
        char, start, end = flat[k]
        if char == text[pos] or k < len(flat):        # 字对不上也按位置退化
            times[pos] = (start, end)
    # 标点继承前一个非标点字符的时间
    last = None
    for i in range(len(text)):
        if times[i] is not None:
            last = times[i]
        elif last is not None:
            times[i] = last
    return times


def _build_lines(text, words, base_ts, start_hint=None, end_hint=None):
    """把上游「整段累积文本 + 逐字时间」切成带时间戳的句子。

    上游 word 的 end_time 偏大（带前瞻窗口），所以句子结束时间优先用
    「下一句首字时间」，末句用上游整段的 end_time，最后再夹逼一次。
    """
    if start_hint is not None and start_hint < 0:      # 上游偶尔回 -0.001 表示"未知"
        start_hint = None
    if end_hint is not None and end_hint < 0:
        end_hint = None
    times = _align_times(text, words)
    lines = []
    cur = None
    for i, ch in enumerate(text):
        if cur is None:
            cur = {"text": "", "start": None, "end": None}
        cur["text"] += ch
        ts = times[i]
        if ts is not None:
            if cur["start"] is None or (ts[0] is not None and ts[0] < cur["start"]):
                cur["start"] = ts[0]
            if ts[1] is not None:
                cur["end"] = ts[1] if cur["end"] is None else max(cur["end"], ts[1])
        if ch in PUNCT:
            lines.append(cur)
            cur = None
    if cur:
        lines.append(cur)

    # 先用整段 hint 兜底（上游没给逐字时间时，逐句只能退化成整段范围），
    # 再校正：句尾取下一句起点，末句取整段结束时间
    for i, line in enumerate(lines):
        if line.get("start") is None:
            line["start"] = start_hint if start_hint is not None else 0.0
        if line.get("end") is None:
            line["end"] = end_hint if end_hint is not None else line["start"] + 0.1
        if i + 1 < len(lines):
            nxt = lines[i + 1].get("start")
            if nxt is not None and nxt > line["start"]:
                line["end"] = nxt
        elif end_hint is not None and end_hint > line["start"]:
            line["end"] = end_hint
        line["end"] = max(line["end"], line["start"] + 0.1)

    out = []
    for line in lines:
        start = line.get("start")
        end = line.get("end")
        if start is not None and start < 0:          # 上游偶发 -0.001，按 0 处理
            start = 0.0
        if end is not None and end < 0:
            end = start
        out.append({
            "text": line["text"],
            "start": round(start, 2) if start is not None else None,
            "end": round(end, 2) if end is not None else None,
            "ts": int((base_ts + (start or 0)) * 1000),        # 墙钟毫秒：说话时刻
            "final": False,
        })
    return out


class Transcript:
    """维护带时间戳的句子列表，永远不会丢已经说过的内容。

    上游每帧回的是「本段累积文本」，所以这里按前缀关系判断是同一句的修订/增长，
    还是已经翻篇的新一段；翻篇时把上一段锁住，不再被覆盖。
    """

    def __init__(self, base_ts, lexicon=None):
        self.lexicon = lexicon or Lexicon()
        self.base_ts = base_ts
        self.locked = []       # 已定稿并锁住的行
        self.live = []         # 当前这一段的实时行
        self.last_text = ""
        self.last_settled = False
        self.last_vad_finished = False
        self.seg_end = None    # 当前这一段已知的结束时间（秒，相对会话起点）
        self.seg_start = None  # 当前这一段的起点（判断“改字”还是“新一段”的关键）
        self.serial = 0        # 上游消息序号：用来区分“同一句改字”与“同文本的不同句”

    def _is_new_segment(self, text, start_hint, vad_finished=False):
        """判断上游是不是已经翻到新的一段发言。

    实测坑（2026-10-16）：短语气词会被上游反复改字（唉→哎→唉），三条消息
    **start_time 完全相同、只有 end_time 在涨**，只是文本在摇摆。
    以前只看文本是否“增长/缩短”，就会把改字当成新的一段，一句话变成三行。
    """
        if not (self.live or self.last_text):
            return False
        # 0) 起点没动 → 同一句的修订（改字），无论文本多不一致都不新开一段
        if (start_hint is not None and self.seg_start is not None
                and abs(start_hint - self.seg_start) <= 0.35):
            return False
        # 1) 起点跳到上一段结束之后了 → 新的一段
        if (start_hint is not None and start_hint >= 0 and self.seg_end is not None
                and start_hint >= self.seg_end - 0.2):
            return True
        # 2) 上一轮已经定稿（或已 VAD 结束），这次却变短了 → 是新的一段
        if (self.last_settled or self.last_vad_finished) and len(text) < len(self.last_text):
            return True
        # 3) 文本对不上（既不是增长也不是缩短）→ 新的一段
        if self.last_text and not self._related(text, self.last_text):
            return True
        return False

    def apply(self, text, interim, words, start_hint=None, end_hint=None, vad_finished=False, cap=None):
        if not text:
            return
        self.serial += 1
        if self._is_new_segment(text, start_hint, vad_finished):
            self.locked += [dict(line, final=True) for line in self.live]
            self.live = []
            self.last_text = ""
            self.seg_end = None
            self.seg_start = None

        self.live = _build_lines(text, words, self.base_ts, start_hint, end_hint)
        # 上游偶尔把新一段的起点丢掉（回 -0.001，被归成 0），用上一段的结束时间兜底，
        # 免得时间轴跳回开头
        if (self.locked and self.live and self.locked[-1].get("end") is not None
                and self.live[0].get("start") is not None
                and self.live[0]["start"] < self.locked[-1]["end"]):
            shift = self.locked[-1]["end"] - self.live[0]["start"]
            for line in self.live:
                if line.get("start") is not None:
                    line["start"] = round(line["start"] + shift, 2)
                if line.get("end") is not None:
                    line["end"] = round(line["end"] + shift, 2)
                line["ts"] = int((self.base_ts + (line.get("start") or 0)) * 1000)
        if cap:                                      # 时间不能超过已经送出去的音频长度
            for line in self.live:
                if line.get("start") is not None and line["start"] > cap:
                    line["start"] = round(cap, 2)
                if line.get("end") is not None and line["end"] > cap:
                    line["end"] = round(cap, 2)
                if line.get("start") is not None and line.get("end") is not None:
                    line["end"] = max(line["end"], line["start"] + 0.1)
                line["ts"] = int((self.base_ts + (line.get("start") or 0)) * 1000)
        for line in self.live:                       # 本地纠正词典：先改字再上屏
            fixed, hits = self.lexicon.apply(line["text"])
            if hits:
                line["orig"], line["text"], line["fixes"] = line["text"], fixed, hits
        if self.live:
            self.live[-1]["final"] = not interim
            self.seg_end = self.live[-1]["end"] if self.live[-1]["end"] is not None else self.seg_end
            if self.seg_start is None and self.live[0]["start"] is not None:
                self.seg_start = self.live[0]["start"]
            for line in self.live:                     # 标上来自哪条上游消息
                line["_serial"] = self.serial
        self.last_text = text
        self.last_settled = not interim
        self.last_vad_finished = vad_finished

    @staticmethod
    def _related(a, b):
        """两段文本是否是同一次发言的修订关系。"""
        if not a or not b:
            return True
        if a.startswith(b) or b.startswith(a):
            return True
        return a[:2] == b[:2]        # 至少开头两个字一样，视为同一句的重写

    def close(self):
        """收尾：把实时行也锁成定稿。"""
        self.locked += [dict(line, final=True) for line in self.live]
        self.live = []

    @property
    def lines(self):
        return self._collapse(self.locked + self.live)

    @staticmethod
    def _mergeable(prev, line):
        """两行是不是同一个短句在反复改字（时间紧邻 + 内容相近/太短）。

    必须来自**不同的上游消息**：同一条消息里按标点切出来的多句（哪怕是重复的句子）
    是真实的多句，不能合并——这就是之前把“连说三句相同的话”合并成一句的原因。
    """
        ta = (prev.get("text") or "").strip()
        tb = (line.get("text") or "").strip()
        if not ta or not tb:
            return False
        if prev.get("_serial") is not None and prev.get("_serial") == line.get("_serial"):
            return False
        ea, sb = prev.get("end"), line.get("start")
        if ea is None or sb is None or sb - ea > 0.6:      # 中间有停顿就不算同一句
            return False
        if ta == tb:
            return True
        if len(ta) <= 4 and len(tb) <= 4:                  # "唉/哎" 这种短语气词
            return True
        return ta in tb or tb in ta

    def _collapse(self, lines):
        """将时间重叠/紧邻的重复行合并成一行（防止一句话被上游切成好几行）。"""
        out = []
        for line in lines:
            if out and self._mergeable(out[-1], line):
                prev = out[-1]
                if line.get("final") and not prev.get("final"):
                    keep = line
                elif prev.get("final") and not line.get("final"):
                    keep = prev
                elif len(line.get("text") or "") >= len(prev.get("text") or ""):
                    keep = line
                else:
                    keep = prev
                merged = dict(prev)
                merged["text"] = keep.get("text")
                merged["end"] = max(prev.get("end") or 0, line.get("end") or 0)
                merged["final"] = bool(prev.get("final") or line.get("final"))
                for key in ("orig", "fixes"):
                    if keep.get(key):
                        merged[key] = keep[key]
                    elif not merged.get(key):
                        merged.pop(key, None)
                out[-1] = merged
                continue
            out.append(dict(line))
        return out

    @property
    def text(self):
        return "\n".join(line["text"] for line in self.lines if line["text"])


# ---------------------------------------------------------------- 上游协议编解码

def _varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | 0x80 if v else b)
        if not v:
            return bytes(out)


def encode_request(token="", service="ASR", method="", payload="", audio=b"", request_id="", state=0):
    """编码上游 protobuf 请求帧（字段号来自实测/公开逆向）。"""
    body = bytearray()
    for num, val in ((2, token), (3, service), (5, method), (6, payload)):
        if val:
            raw = val.encode()
            body += _varint(num << 3 | 2) + _varint(len(raw)) + raw
    if audio:
        body += _varint(7 << 3 | 2) + _varint(len(audio)) + audio
    if request_id:
        raw = request_id.encode()
        body += _varint(8 << 3 | 2) + _varint(len(raw)) + raw
    if state:
        body += _varint(9 << 3) + _varint(state)
    return bytes(body)


def decode_message(data: bytes):
    """解析上游响应帧，返回 {字段号: [值, ...]}。"""
    out, off = {}, 0

    def read_varint(buf, pos):
        val = shift = 0
        while True:
            byte = buf[pos]
            pos += 1
            val |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return val, pos

    while off < len(data):
        tag, off = read_varint(data, off)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, off = read_varint(data, off)
            out.setdefault(field, []).append(val)
        elif wire == 2:
            length, off = read_varint(data, off)
            out.setdefault(field, []).append(data[off:off + length])
            off += length
        elif wire == 1:
            off += 8
        elif wire == 5:
            off += 4
        else:
            break
    return out


def field_str(msg, num):
    val = msg.get(num)
    if not val:
        return None
    return val[0].decode("utf-8", "replace") if isinstance(val[0], bytes) else val[0]


class LocalSession:
    """本地 FunASR 引擎会话（浏览器 PCM → asr-engine /stream）。

    为什么要有它：豆包云端不可用、被限流、或干脆断网时，页面至少还能转写。
    代价：本地引擎在 CPU 上约 1.85x 实时，且流式结果没有真时间戳（按字数估算）。
    """

    def __init__(self, on_update, lexicon=None):
        self.on_update = on_update
        self.lexicon = lexicon or Lexicon()
        self.client = None
        self.sid = None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.sender = None
        self.text = ""
        self.base_ts = time.time()
        self.buf = bytearray()
        self.bytes_in = 0

    @property
    def engine(self):
        return "local"

    @property
    def headers(self):
        return {"x-engine-token": LOCAL_ENGINE_TOKEN} if LOCAL_ENGINE_TOKEN else {}

    async def open(self, hotwords=()):
        if not LOCAL_ENGINE_URL:
            raise RuntimeError("未配置本地引擎（DOUBAO_LOCAL_ENGINE）")
        self.client = httpx.AsyncClient(base_url=LOCAL_ENGINE_URL, timeout=180)
        resp = await self.client.post("/stream", headers=self.headers, json={
            "language": "zh", "hotwords": " ".join(hotwords or []),
        })
        resp.raise_for_status()
        self.sid = resp.json()["session_id"]
        self.sender = asyncio.create_task(self._send_loop())
        log.info("本地引擎会话已建立 sid=%s", self.sid)

    async def _send_loop(self):
        while True:
            chunk = await self.queue.get()
            if chunk is None:
                return
            try:
                resp = await self.client.post(
                    f"/stream/{self.sid}/chunk", headers={**self.headers, "content-type": "application/octet-stream"},
                    content=chunk)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:                 # noqa: BLE001
                log.warning("本地引擎喂数据失败：%r", exc)
                continue
            text = (data.get("text") or "").strip()
            if text and text != self.text:
                self.text = text
                await self.on_update(self._live_lines(text))

    async def push(self, frame: bytes, state: int = 0):
        self.bytes_in += len(frame)
        self.buf += frame
        while len(self.buf) >= LOCAL_CHUNK:
            await self.queue.put(bytes(self.buf[:LOCAL_CHUNK]))
            del self.buf[:LOCAL_CHUNK]

    async def replay(self, pcm: bytes):
        """切换引擎时把已缓冲的音频补喂进来（保证不丢已经说过的话）。"""
        for i in range(0, len(pcm), LOCAL_CHUNK):
            await self.queue.put(pcm[i:i + LOCAL_CHUNK])

    def _fix(self, text):
        fixed, hits = self.lexicon.apply(text)
        return fixed, hits

    def _live_lines(self, text):
        """流式阶段：本地引擎不给标点，先当一整句显示。"""
        fixed, hits = self._fix(text)
        line = {"text": fixed, "start": 0.0, "end": round(self.bytes_in / (SAMPLE_RATE * 2), 2),
                "ts": int(self.base_ts * 1000), "final": False}
        if hits:
            line.update(orig=text, fixes=hits)
        return [line]

    def _final_lines(self, data):
        sentences = data.get("sentences") or []
        text = (data.get("text") or "").strip()
        if not sentences and text:
            sentences = [{"start": 0, "end": int(self.bytes_in / (SAMPLE_RATE * 2) * 1000), "text": text}]
        lines = []
        for item in sentences:
            raw = (item.get("text") or "").strip()
            if not raw:
                continue
            fixed, hits = self._fix(raw)
            start = (item.get("start") or 0) / 1000
            line = {"text": fixed, "start": round(start, 2),
                    "end": round((item.get("end") or 0) / 1000, 2),
                    "ts": int((self.base_ts + start) * 1000), "final": True}
            if hits:
                line.update(orig=raw, fixes=hits)
            lines.append(line)
        self.text = "\n".join(l["text"] for l in lines)
        return lines

    async def finish(self, timeout=240):
        if self.buf:
            await self.queue.put(bytes(self.buf))
            self.buf.clear()
        await self.queue.put(None)                 # 排干队列
        if self.sender:
            try:
                await asyncio.wait_for(self.sender, timeout=timeout)
            except asyncio.TimeoutError:
                self.sender.cancel()
        try:
            resp = await self.client.post(f"/stream/{self.sid}/finish", headers=self.headers)
            resp.raise_for_status()
            return self._final_lines(resp.json()), None
        except Exception as exc:                     # noqa: BLE001
            log.warning("本地引擎收尾失败：%r", exc)
            return self._live_lines(self.text), f"本地引擎收尾失败：{exc}"

    async def close(self):
        if self.sender and not self.sender.done():
            await self.queue.put(None)
        if self.client:
            try:
                if self.sid:
                    await self.client.post(f"/stream/{self.sid}/close", headers=self.headers)
            except Exception:                        # noqa: BLE001
                pass
            await self.client.aclose()

    @property
    def dead(self):
        return False


class UpstreamError(Exception):
    pass


class UpstreamSession:
    """一次识别会话：StartTask → StartSession → 若干 TaskRequest → FinishSession。"""

    def __init__(self, creds, on_update, lexicon=None):
        self.creds = creds
        self.lexicon = lexicon or Lexicon()
        self.on_update = on_update            # async callback(lines)
        self.ws = None
        self.task_id = None
        self.request_id = str(uuid.uuid4())
        self.finished = asyncio.Event()
        self.error = None
        self._reader = None
        self.audio_seconds = 0.0
        self.transcript = Transcript(time.time(), self.lexicon)

    async def open(self):
        url = f"{WS_URL}?aid={AID}&device_id={urllib.parse.quote(str(self.creds['device_id']))}"
        headers = {"User-Agent": UA, "proto-version": "v2", "x-custom-keepalive": "true"}
        self.ws = await websockets.connect(url, additional_headers=headers,
                                          open_timeout=15, max_size=None, ping_interval=20)
        await self._handshake()

    async def _send(self, **kw):
        data = encode_request(token=self.creds["app_key"], request_id=self.request_id, **kw)
        if DEBUG_WIRE:
            log.info("WIRE> method=%s payload=%s bytes=%s", kw.get("method"), kw.get("payload"), data.hex()[:160])
        await self.ws.send(data)

    async def _recv(self, timeout=15):
        raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
        if isinstance(raw, str):
            return {"json": raw}
        return decode_message(raw)

    async def _handshake(self):
        for attempt in range(3):                       # 上游偶发 50700000（service discovery failure），重试几次
            await self._send(method="StartTask")
            msg = await self._recv()
            if field_str(msg, 4) == "TaskStarted":
                break
            code = (msg.get(5) or [0])[0]
            if attempt < 2 and code and code >= 50000000:
                log.info("上游 StartTask 5xx（%s），重试 %d", code, attempt + 1)
                await asyncio.sleep(0.6)
                continue
            raise UpstreamError(self._fail_reason(msg, "StartTask"))

        payload = json.dumps({
            "audio_info": {"channel": 1, "format": "raw", "sample_rate": SAMPLE_RATE},
            "enable_punctuation": True,
            "enable_speech_rejection": False,
            "extra": {
                "did": str(self.creds["device_id"]),
                "input_mode": "tool",
                # 以下四个是豆包输入法自己的多遍解码/纠错开关（与客户端一致）
                "strong_ddc": True,
                "use_twopass_retry": True,
                "enable_asr_twopass": True,
                "enable_asr_threepass": True,
                "disable_user_words": False,
                "cell_compress_rate": 8,
                **({"context": build_context("com.android.chrome", self.lexicon.hotwords())} if USE_CONTEXT else {}),
            },
        }, ensure_ascii=False)
        for attempt in range(3):
            await self._send(method="StartSession", payload=payload)
            msg = await self._recv()
            if field_str(msg, 4) == "SessionStarted":
                break
            code = (msg.get(5) or [0])[0]
            if attempt < 2 and code and code >= 50000000:
                log.info("上游 StartSession 5xx（%s），重试 %d", code, attempt + 1)
                await asyncio.sleep(0.6)
                continue
            raise UpstreamError(self._fail_reason(msg, "StartSession"))
        self.task_id = field_str(msg, 2)
        self._reader = asyncio.create_task(self._read_loop())

    @staticmethod
    def _fail_reason(msg, stage):
        code = (msg.get(5) or [None])[0]
        text = field_str(msg, 6)
        if code == 40200011:
            return "服务端并发配额已满（共享 app_key 限流），请稍后重试"
        if code in (40200001, 40100003):
            return f"上游鉴权失败（{code}）：{text}"
        return f"{stage} 失败（{code}）：{text}"

    async def _read_loop(self):
        try:
            while True:
                msg = await self._recv(timeout=120)
                if "json" in msg:
                    continue
                kind = field_str(msg, 4)
                result = field_str(msg, 7)
                if result:
                    try:
                        data = json.loads(result)
                    except ValueError:
                        data = None
                    if data is not None:
                        await self._consume(data)
                if kind in ("SessionFinished", "TaskFinished", "SessionFailed", "TaskFailed"):
                    if kind.endswith("Failed"):
                        self.error = self._fail_reason(msg, kind)
                    self.finished.set()
                    return
        except Exception as exc:                     # noqa: BLE001
            log.info("上游读取结束: %r", exc)
            self.error = self.error or f"上游连接中断：{exc}"
            self.finished.set()

    @property
    def engine(self):
        return "cloud"

    @property
    def dead(self):
        return bool(self.error) or (self.ws is not None and getattr(self.ws, "state", None) is not None
                                    and self.ws.state.name != "OPEN")

    async def _consume(self, data):
        items = [it for it in (data.get("results") or []) if it.get("text")]
        if not items:
            return
        item = max(items, key=lambda it: it.get("index", 0))
        words = None
        for alt in (item.get("alternatives") or []):
            if alt.get("words"):
                words = alt["words"]
                break
        self.transcript.apply(item.get("text") or "", bool(item.get("is_interim")), words,
                              start_hint=item.get("start_time"), end_hint=item.get("end_time"),
                              vad_finished=bool(item.get("is_vad_finished")),
                              cap=self.audio_seconds or None)
        await self.on_update(self.transcript.lines)

    async def push(self, frame: bytes, state: int):
        self.audio_seconds += len(frame) / (SAMPLE_RATE * 2)
        await self._send(method="TaskRequest",
                         payload=json.dumps({"timestamp_ms": int(time.time() * 1000) % 100000}),
                         audio=frame, state=state)

    async def replay(self, pcm: bytes):
        """重连后把缓存的音频补喂回去（首帧 1 / 其余 3，末帧留给正常收尾）。"""
        frames = [pcm[i:i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
        self.audio_seconds = 0.0
        for i, frame in enumerate(frames):
            await self.push(frame, 1 if i == 0 else 3)

    async def finish(self, timeout=25):
        try:
            await self._send(method="FinishSession")
        except Exception:                            # noqa: BLE001
            self.finished.set()
        try:
            await asyncio.wait_for(self.finished.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self.error = self.error or "等待最终结果超时"
        self.transcript.close()
        return self.transcript.lines, self.error

    async def close(self):
        if self._reader and not self._reader.done():
            self._reader.cancel()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:                        # noqa: BLE001
                pass


# ---------------------------------------------------------------- WebSocket 端点

def _retryable(reason):
    """上游抖动（50700000 service discovery failure、连接断开等）值得重连。"""
    text = str(reason or "")
    if "service discovery" in text or "连接中断" in text:
        return True
    match = re.search(r"（(\d{6,})\)", text)
    return bool(match and int(match.group(1)) >= 50000000)


async def _open_upstream(on_update, lexicon=None):
    """建立上游会话；app_key 失效时刷新一次重试。"""
    last = None
    for attempt in range(2):
        creds = await CREDS.get(force=attempt > 0)
        session = UpstreamSession(creds, on_update, lexicon)
        try:
            await session.open()
            return session
        except Exception as exc:                     # noqa: BLE001
            last = exc
            await session.close()
            log.warning("上游握手失败（第 %d 次）：%s", attempt + 1, exc)
    raise last if isinstance(last, Exception) else RuntimeError("上游不可用")


@app.websocket("/ws")
async def transcribe_socket(ws: WebSocket):
    """?engine=auto|cloud|local；auto 时云端失败会自动降级到本地引擎。"""
    await ws.accept()
    mode = (ws.query_params.get("engine") or "auto").lower()
    if mode not in ("auto", "cloud", "local"):
        mode = "auto"
    if slots.locked() and slots._value == 0:         # noqa: SLF001
        await ws.send_json({"type": "status", "message": f"当前会话已满（上限 {MAX_SESSIONS}），排队等待…"})

    async with slots:
        session = None
        engine = None
        buf = bytearray()
        history = bytearray()                        # 已收到的 PCM，重连/降级时补喂
        pending = None
        sent_first = False
        finished = False
        cloud_retries = 0

        async def feed_pcm(data: bytes):
            """把浏览器 PCM 切成 20ms 帧往下送（云端压一帧，用于标末帧）。"""
            nonlocal buf, pending, sent_first
            history.extend(data)
            if len(history) > HISTORY_BYTES:
                del history[:-HISTORY_BYTES]
            buf += data
            while len(buf) >= FRAME_BYTES:
                frame = bytes(buf[:FRAME_BYTES])
                del buf[:FRAME_BYTES]
                try:
                    if engine == "cloud":
                        if session.dead:            # 上游挂了，交给外层重连/降级
                            return
                        if pending is not None:
                            await session.push(pending, 1 if not sent_first else 3)
                            sent_first = True
                        pending = frame
                    else:
                        await session.push(frame)
                except Exception as exc:            # noqa: BLE001
                    log.info("推送失败（交给外层重连）：%r", exc)
                    return

        async def reopen_cloud(reason):
            """云端抖动：重连并把已说过的音频补喂回去。"""
            nonlocal session, sent_first, pending
            await session.close()
            session = await _open_upstream(emit, session_lexicon)
            sent_first = True
            pending = None
            if history:
                await session.replay(bytes(history))
            log.info("云端已重连并补喂 %.1fs 音频（%s）", len(history) / (SAMPLE_RATE * 2), reason)

        def public(lines):
            return [{k: v for k, v in line.items() if not k.startswith("_")} for line in lines]

        async def label_speakers(lines):
            """在线追踪：句子一定稿就立刻算声纹、跟已有说话人比，实时把标签贴上。"""
            if tracker is None or not nonlocal_diarize[0]:
                return
            for line in lines:
                if line.get("final") and not line.get("spk"):
                    spk = await asyncio.to_thread(tracker.assign, bytes(history), line.get("start"), line.get("end"))
                    if spk:
                        line["spk"] = spk
                        log.info("在线说话人：%.1f-%.1fs → 说话人 %s", line.get("start") or 0, line.get("end") or 0, spk)

        async def emit(lines):
            await label_speakers(lines)
            try:
                await ws.send_json({"type": "update", "engine": engine, "lines": public(lines),
                                    "text": "\n".join(l["text"] for l in lines if l.get("text"))})
            except Exception:                        # noqa: BLE001
                pass

        # 词库（纠正 + 热词）由**浏览器本地**保存、每个用户一份；会话开始时客户端发过来，
        # 服务端只在本次会话里用，用完即弃 —— 这样多人同时用不会互相覆盖。
        session_lexicon = Lexicon()
        early_audio = []
        tracker = SpeakerTracker() if DIAR_MODE in ("online", "both") and DIARIZER.available() else None

        nonlocal_diarize = [False]          # read_config 里改，主流程里读

        async def read_config():
            """等客户端先发词库（最多 1s）；如果先收到音频就留着，交给后面正常流程。"""
            deadline = time.time() + 1.0
            while True:
                try:
                    message = await asyncio.wait_for(ws.receive(), timeout=max(0.1, deadline - time.time()))
                except asyncio.TimeoutError:
                    return
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect()
                if (text := message.get("text")) is not None:
                    try:
                        command = json.loads(text)
                    except ValueError:
                        continue
                    if command.get("type") == "config":
                        nonlocal_diarize[0] = bool(command.get("diarize"))
                        session_lexicon.data = Lexicon.from_payload(command.get("lexicon") or {}).data
                        log.info("会话词库：%d 条纠正 / %d 个热词", len(session_lexicon.data["corrections"]),
                                 len(session_lexicon.data["hotwords"]))
                        return
                    if command.get("type") == "stop":
                        return
                elif (data := message.get("bytes")) is not None:
                    early_audio.append(data)
                    return

        async def start_local(reason):
            opened = LocalSession(emit, session_lexicon)
            await opened.open(session_lexicon.hotwords())
            await ws.send_json({"type": "status", "engine": "local", "message": reason})
            return opened

        try:
            await read_config()
            if mode in ("auto", "cloud"):
                await ws.send_json({"type": "status", "message": "正在连接识别服务…"})
                try:
                    session = await _open_upstream(emit, session_lexicon)
                    engine = "cloud"
                except Exception as exc:             # noqa: BLE001
                    log.warning("云端不可用：%r", exc)
                    if mode == "cloud":
                        raise
                    session = await start_local(f"云端识别不可用（{exc}），已切到本地引擎")
                    engine = "local"
            else:
                session = await start_local("使用本地识别引擎（离线兜底）")
                engine = "local"
            await ws.send_json({"type": "status", "engine": engine,
                                "message": "本地引擎已就绪，开始说话" if engine == "local" else "已就绪，开始说话"})
            if nonlocal_diarize[0] and tracker is None and DIAR_MODE in ("online", "both"):
                await ws.send_json({"type": "status", "message": "说话人分离模型不可用（见 tools/fetch_diarization_models.sh）"})
            for chunk in early_audio:                # 比 config 先到的那点音频
                await feed_pcm(chunk)

            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (data := message.get("bytes")) is not None:
                    if engine == "cloud" and session.dead:
                        # 上游抖动：先重连补喂，实在不行再降级/报错
                        reason = session.error or "云端连接中断"
                        if _retryable(reason) and cloud_retries < CLOUD_MAX_RETRY:
                            cloud_retries += 1
                            await ws.send_json({"type": "status", "engine": "cloud",
                                                "message": f"云端抖动，重连中（{cloud_retries}/{CLOUD_MAX_RETRY}）…"})
                            try:
                                await reopen_cloud(reason)
                            except Exception as exc:             # noqa: BLE001
                                log.warning("云端重连失败：%r", exc)
                                reason = f"{reason}；重连失败：{exc}"
                                if mode != "auto":
                                    raise UpstreamError(reason) from exc
                                session = await start_local("云端重连失败，已切到本地引擎")
                                engine = "local"
                                cloud_retries = CLOUD_MAX_RETRY
                                await session.replay(bytes(history))
                        elif mode == "auto":
                            log.warning("云端会话中断，降级本地：%s", reason)
                            await session.close()
                            session = await start_local("云端连接中断，已切到本地引擎")
                            engine = "local"
                            await session.replay(bytes(history))
                        else:
                            raise UpstreamError(reason)
                    await feed_pcm(data)
                elif (text := message.get("text")) is not None:
                    try:
                        command = json.loads(text)
                    except ValueError:
                        continue
                    if command.get("type") == "stop" and not finished:
                        finished = True
                        await ws.send_json({"type": "status", "message": "正在整理结果…"})
                        if engine == "cloud" and pending:      # 末帧
                            await session.push(pending, 1 if not sent_first else 9)
                            pending = None
                        lines, error = await session.finish()
                        # finish() 会重建行对象，实时阶段贴的标签会丢，这里按时间区间重新贴一遍
                        # （追踪器按 (start,end) 缓存，不会重复算声纹）
                        await label_speakers(lines)
                        received = len(history) / (SAMPLE_RATE * 2)
                        last_end = max((l.get("end") or 0) for l in lines) if lines else 0
                        log.info("会话收尾：收到 %.1fs 音频，上游认到 %.1fs（%d 句，跨度差 %.1fs）",
                                 received, last_end, len(lines), received - last_end)
                        speakers = tracker.count if tracker else 0
                        if nonlocal_diarize[0] and DIAR_MODE in ("batch", "both"):
                            if DIARIZER.available():
                                await ws.send_json({"type": "status", "message": "正在区分说话人（本地模型）…"})
                                segments = await asyncio.to_thread(DIARIZER.diarize, bytes(history))
                                speakers = assign_speakers(lines, segments, FILL_GAP_SECONDS)
                                log.info("说话人分离：%d 段 → %d 人", len(segments), speakers)
                                if not speakers:
                                    error = error or "没听出多个人（或音频太短），本次未区分说话人"
                            else:
                                error = error or "未安装说话人分离模型，跳过（见 tools/fetch_diarization_models.sh）"
                        elif nonlocal_diarize[0] and not DIARIZER.available():
                            error = error or "未安装说话人分离模型，跳过（见 tools/fetch_diarization_models.sh）"
                        await ws.send_json({"type": "final", "engine": engine, "lines": public(lines),
                                            "text": "\n".join(l["text"] for l in lines if l.get("text")),
                                            "speakers": speakers, "error": error,
                                            # 诊断用：浏览器送来多少秒音频 / 上游认到第几秒。
                                            # 两者差得多说明「音频到了但没认出来」，不是链路丢包。
                                            "audio_seconds": round(received, 1),
                                            "recognized_seconds": round(last_end, 1)})
                        break
                    if command.get("type") == "cancel":
                        break
        except WebSocketDisconnect:
            pass
        except Exception as exc:                     # noqa: BLE001
            log.warning("会话异常: %r", exc)
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:                        # noqa: BLE001
                pass
        finally:
            if session is not None:
                await session.close()
            try:
                await ws.close()
            except Exception:                        # noqa: BLE001
                pass


# ---------------------------------------------------------------- 静态页面

@app.get("/health")
async def health():
    creds = CREDS.data
    return {
        "status": "ok",
        "device_id": creds.get("device_id"),
        "app_key_ready": bool(creds.get("app_key")),
        "max_sessions": MAX_SESSIONS,
        "aid": AID,
        "diarization": {**DIARIZER.status(), "mode": DIAR_MODE},
    }


# ---------------------------------------------------------------- 纠正词典接口

class Correction(BaseModel):
    wrong: str
    right: str


class Hotword(BaseModel):
    word: str


@app.post("/api/corrections")
async def add_correction(body: Correction):
    return JSONResponse({"ok": False, "message": "词库已改为浏览器本地保存，请在页面上编辑"},
                        status_code=410)


@app.delete("/api/corrections")
async def drop_correction(wrong: str):
    return JSONResponse({"ok": False, "message": "词库已改为浏览器本地保存，请在页面上删除"},
                        status_code=410)


@app.post("/api/hotwords")
async def add_hotword(body: Hotword):
    return JSONResponse({"ok": False, "message": "热词已改为浏览器本地保存，请在页面上编辑"},
                        status_code=410)


@app.delete("/api/hotwords")
async def drop_hotword(word: str):
    return JSONResponse({"ok": False, "message": "热词已改为浏览器本地保存，请在页面上删除"},
                        status_code=410)


# ---------------------------------------------------------------- 静态页面

def render_page(path: Path) -> HTMLResponse:
    """把静态页里的 __GITHUB_URL__ 换成配置的仓库地址（改地址只需改一个环境变量）。"""
    html = path.read_text(encoding="utf-8").replace("__GITHUB_URL__", GITHUB_URL)
    return HTMLResponse(html)


@app.get("/")
async def index():
    return render_page(STATIC / "index.html")


@app.get("/how")
async def how_it_works():
    """原理与复刻指南（由 tools/build_how_page.py 生成）。"""
    page = STATIC / "how.html"
    if not page.exists():
        return JSONResponse({"error": "请先运行 tools/build_how_page.py"}, status_code=404)
    return render_page(page)


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse({}, status_code=204)
