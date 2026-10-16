#!/usr/bin/env python3
"""豆包输入法语音识别 · 最小可用客户端（Python，单文件）

原理：客户端内置 aid + 匿名设备注册 → 服务端下发 ASR app_key → 用它连 WebSocket，
      按 20ms 一帧送 16k/单声道/PCM16，边送边收识别结果。

用法: python3 asr_min.py <16k-mono-16bit.wav>
依赖: pip install websockets
"""
import asyncio
import hashlib
import json
import secrets
import sys
import time
import urllib.parse
import urllib.request
import uuid
import wave

import websockets

AID = 685343                     # 豆包输入法的 app id（取自客户端 ImeService.exe）
APP_NAME = "oime"
VERSION_CODE = 100102018
VERSION_NAME = "1.1.2"
UA = ("com.bytedance.android.doubaoime/100102018 (Linux; U; Android 16; en_US; "
      "Pixel 7 Pro; Build/BP2A.250605.031.A2)")
REGISTER_URL = "https://log.snssdk.com/service/2/device_register/"   # 头条系匿名设备注册
SETTINGS_URL = "https://is.snssdk.com/service/settings/v3/"          # 下发 asr_config.app_key
WS_URL = "wss://frontier-audio-ime-ws.doubao.com/ocean/api/v1/ws"

SAMPLE_RATE = 16000
FRAME_BYTES = 640                # 20ms × 16kHz × 2字节 = 640


# ----------------------------------------------------------------- 1. 拿凭据

def _post(url, params, body, headers):
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), data=body, method="POST")
    for key, value in headers.items():
        req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def get_credentials():
    """返回 (device_id, app_key)。全过程匿名，不需要登录。"""
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
    out = _post(REGISTER_URL, params, body,
                {"Content-Type": "application/json", "User-Agent": UA})
    device_id = str(out.get("device_id_str") or out.get("device_id") or "")
    if not device_id or device_id == "0":
        raise RuntimeError(f"设备注册失败: {out}")

    stub_body = b"body=null"
    params = {
        "device_platform": "android", "os": "android", "ssmix": "a", "channel": "official",
        "aid": str(AID), "app_name": APP_NAME, "version_code": str(VERSION_CODE),
        "version_name": VERSION_NAME, "device_id": device_id, "cdid": cdid,
        "_rticket": str(int(time.time() * 1000)),
    }
    out = _post(SETTINGS_URL, params, stub_body, {
        "User-Agent": UA,
        "x-ss-stub": hashlib.md5(stub_body).hexdigest().upper(),
        "Content-Type": "application/x-www-form-urlencoded",
    })
    app_key = (out.get("data", {}).get("settings", {}).get("asr_config", {}) or {}).get("app_key")
    if not app_key:
        raise RuntimeError(f"settings 没给 app_key: {out}")
    return device_id, app_key


# ----------------------------------------------------------------- 2. protobuf 帧

def _varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | 0x80 if value else byte)
        if not value:
            return bytes(out)


def encode(token="", method="", payload="", audio=b"", request_id="", state=0):
    """上游请求帧的字段号（实测得到）：
       2=token(app_key) 3=service_name 5=method 6=payload 7=audio 8=request_id 9=frame_state
    """
    body = bytearray()
    for num, val in ((2, token), (3, "ASR"), (5, method), (6, payload)):
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


def decode(data):
    """响应帧 → {字段号: [值]}。2=task_id 4=消息类型 5=状态码 6=状态文本 7=结果 JSON"""
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


def text_of(msg, field):
    val = msg.get(field)
    if not val:
        return None
    return val[0].decode("utf-8", "replace") if isinstance(val[0], bytes) else val[0]


# ----------------------------------------------------------------- 3. 跑一次识别

def is_retryable(err):
    """上游会偶发 50700000 service discovery failure（实测约 1/5 概率），重试即可。"""
    msg = str(err)
    return "service discovery" in msg or any(f"[{code}" in msg for code in range(50000000, 60000000))


async def transcribe(path, retries=4):
    """带重试的一次识别：上游抖动 / 连接断开都重试整段。"""
    last_error = None
    for attempt in range(retries):
        try:
            return await _transcribe_once(path)
        except Exception as exc:                     # noqa: BLE001
            last_error = exc
            if not is_retryable(exc) or attempt == retries - 1:
                raise
            print(f"  [重试 {attempt + 1}/{retries - 1}] {exc}", file=sys.stderr)
            await asyncio.sleep(0.6)
    raise last_error


async def _transcribe_once(path):
    with wave.open(path, "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000), \
            "需要 16kHz / 单声道 / 16bit 的 WAV（可用 ffmpeg -ar 16000 -ac 1 转换）"
        pcm = wav.readframes(wav.getnframes())

    device_id, app_key = get_credentials()
    url = f"{WS_URL}?aid={AID}&device_id={device_id}"
    headers = {"User-Agent": UA, "proto-version": "v2", "x-custom-keepalive": "true"}
    request_id = str(uuid.uuid4())
    text, done = "", False

    async with websockets.connect(url, additional_headers=headers,
                                  open_timeout=15, max_size=None) as ws:
        # ① StartTask：建任务
        await ws.send(encode(app_key, "StartTask", request_id=request_id))
        msg = decode(await ws.recv())
        if text_of(msg, 4) != "TaskStarted":
            raise RuntimeError(f"StartTask 失败: {text_of(msg, 4)} {msg.get(5)} {text_of(msg, 6)}")

        # ② StartSession：声明音频格式 + 识别参数
        session_payload = json.dumps({
            "audio_info": {"channel": 1, "format": "raw", "sample_rate": SAMPLE_RATE},
            "enable_punctuation": True,
            "enable_speech_rejection": False,
            "extra": {
                "did": device_id, "input_mode": "tool",
                "strong_ddc": True, "use_twopass_retry": True,
                "enable_asr_twopass": True, "enable_asr_threepass": True,
                "disable_user_words": False, "cell_compress_rate": 8,
            },
        }, ensure_ascii=False)
        await ws.send(encode(app_key, "StartSession", payload=session_payload, request_id=request_id))
        msg = decode(await ws.recv())
        if text_of(msg, 4) != "SessionStarted":
            raise RuntimeError(f"StartSession 失败: {text_of(msg, 4)} {msg.get(5)} {text_of(msg, 6)}")

        # ③ TaskRequest：20ms 一帧；首帧 1、中间 3、末帧 9
        frames = [pcm[i:i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
        for index, frame in enumerate(frames):
            state = 1 if index == 0 else (9 if index == len(frames) - 1 else 3)
            await ws.send(encode("", "TaskRequest", payload="{}", audio=frame,
                                 request_id=request_id, state=state))
            await asyncio.sleep(0.02)          # 按实时速度喂；也可以更快

        # ④ FinishSession：收尾
        await ws.send(encode(app_key, "FinishSession", request_id=request_id))

        while not done:
            msg = decode(await asyncio.wait_for(ws.recv(), timeout=20))
            kind, result = text_of(msg, 4), text_of(msg, 7)
            if result:
                data = json.loads(result)
                for item in data.get("results") or []:
                    if item.get("text"):
                        text = item["text"]        # 累积文本，最后一条即完整结果
            if kind in ("SessionFinished", "TaskFinished"):
                done = True
            elif kind in ("SessionFailed", "TaskFailed"):
                raise RuntimeError(f"识别失败: {msg.get(5)} {text_of(msg, 6)}")
    return text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法: python3 asr_min.py <16k-mono-16bit.wav>")
    print(asyncio.run(transcribe(sys.argv[1])))
