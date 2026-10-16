#!/usr/bin/env node
/**
 * 豆包输入法语音识别 · 最小可用客户端（Node.js）
 *
 * 用法: npm i ws && node asr_min.js <16k-mono-16bit.wav>
 *
 * 为什么不用 Node 自带的全局 WebSocket：上游靠 `proto-version: v2` 这个握手头选择
 * 二进制 protobuf 协议，没有它就回 JSON；而全局 WebSocket 无法设置自定义请求头，
 * 所以这里用 ws 包（浏览器同理——这也是网页端必须走服务端中转的原因）。
 */

const fs = require("node:fs");
const crypto = require("node:crypto");
const WebSocket = require("ws");

const AID = 685343;                 // 豆包输入法的 app id（取自客户端 ImeService.exe）
const APP_NAME = "oime";
const VERSION_CODE = 100102018;
const VERSION_NAME = "1.1.2";
const UA = "com.bytedance.android.doubaoime/100102018 (Linux; U; Android 16; en_US; " +
           "Pixel 7 Pro; Build/BP2A.250605.031.A2)";
const REGISTER_URL = "https://log.snssdk.com/service/2/device_register/";
const SETTINGS_URL = "https://is.snssdk.com/service/settings/v3/";
const WS_URL = "wss://frontier-audio-ime-ws.doubao.com/ocean/api/v1/ws";

const SAMPLE_RATE = 16000;
const FRAME_BYTES = 640;            // 20ms × 16kHz × 2 字节

// ------------------------------------------------------------- 1. 拿凭据

async function postJson(url, params, body, headers) {
  const target = `${url}?${new URLSearchParams(params)}`;
  const resp = await fetch(target, { method: "POST", body, headers });
  if (!resp.ok) throw new Error(`${url} HTTP ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function getCredentials() {
  const cdid = crypto.randomUUID();
  const openudid = crypto.randomUUID().replace(/-/g, "").slice(0, 16);
  const clientudid = crypto.randomUUID();
  const header = {
    device_id: 0, install_id: 0, aid: AID, app_name: APP_NAME,
    version_code: VERSION_CODE, version_name: VERSION_NAME,
    manifest_version_code: VERSION_CODE, update_version_code: VERSION_CODE,
    channel: "official", device_platform: "android", os: "android", os_api: "34",
    os_version: "16", device_type: "Pixel 7 Pro", device_brand: "google",
    device_model: "Pixel 7 Pro", resolution: "1080*2400", dpi: "420",
    language: "zh", timezone: 8, access: "wifi", region: "CN",
    tz_name: "Asia/Shanghai", tz_offset: 28800, openudid, clientudid, cdid,
  };
  const reg = await postJson(REGISTER_URL, {
    device_platform: "android", os: "android", ssmix: "a", _rticket: Date.now(),
    cdid, channel: "official", aid: AID, app_name: APP_NAME,
    version_code: VERSION_CODE, version_name: VERSION_NAME,
    manifest_version_code: VERSION_CODE, update_version_code: VERSION_CODE,
    resolution: "1080*2400", dpi: "420", device_type: "Pixel 7 Pro",
    device_brand: "google", language: "zh", os_api: "34", os_version: "16", ac: "wifi",
  }, JSON.stringify({ magic_tag: "ss_app_log", header, _gen_time: Date.now() }),
     { "Content-Type": "application/json", "User-Agent": UA });

  const deviceId = String(reg.device_id_str || reg.device_id || "");
  if (!deviceId || deviceId === "0") throw new Error(`设备注册失败: ${JSON.stringify(reg)}`);

  const stub = "body=null";
  const settings = await postJson(SETTINGS_URL, {
    device_platform: "android", os: "android", ssmix: "a", channel: "official",
    aid: AID, app_name: APP_NAME, version_code: VERSION_CODE, version_name: VERSION_NAME,
    device_id: deviceId, cdid, _rticket: Date.now(),
  }, stub, {
    "User-Agent": UA,
    "Content-Type": "application/x-www-form-urlencoded",
    "x-ss-stub": crypto.createHash("md5").update(stub).digest("hex").toUpperCase(),
  });
  const appKey = settings?.data?.settings?.asr_config?.app_key;
  if (!appKey) throw new Error(`settings 没给 app_key: ${JSON.stringify(settings)}`);
  return { deviceId, appKey };
}

// ------------------------------------------------------------- 2. protobuf 帧

function varint(value) {
  const out = [];
  while (true) {
    let byte = value & 0x7f;
    value = Math.floor(value / 128);
    out.push(value ? byte | 0x80 : byte);
    if (!value) return Buffer.from(out);
  }
}

function encode({ token = "", method = "", payload = "", audio = null, requestId = "", state = 0 }) {
  const parts = [];
  for (const [num, val] of [[2, token], [3, "ASR"], [5, method], [6, payload]]) {
    if (!val) continue;
    const raw = Buffer.from(val, "utf8");
    parts.push(varint((num << 3) | 2), varint(raw.length), raw);
  }
  if (audio && audio.length) {
    parts.push(varint((7 << 3) | 2), varint(audio.length), audio);
  }
  if (requestId) {
    const raw = Buffer.from(requestId, "utf8");
    parts.push(varint((8 << 3) | 2), varint(raw.length), raw);
  }
  if (state) parts.push(varint(9 << 3), varint(state));
  return Buffer.concat(parts);
}

function decode(buf) {
  const out = {};
  let off = 0;
  const readVarint = (pos) => {
    let val = 0, shift = 0;
    while (true) {
      const byte = buf[pos++];
      val |= (byte & 0x7f) << shift;
      shift += 7;
      if (!(byte & 0x80)) return [val, pos];
    }
  };
  while (off < buf.length) {
    let tag; [tag, off] = readVarint(off);
    const field = tag >> 3, wire = tag & 7;
    if (wire === 0) {
      let val; [val, off] = readVarint(off);
      (out[field] ||= []).push(val);
    } else if (wire === 2) {
      let len; [len, off] = readVarint(off);
      (out[field] ||= []).push(buf.subarray(off, off + len));
      off += len;
    } else if (wire === 1) off += 8;
    else if (wire === 5) off += 4;
    else break;
  }
  return out;
}

const textOf = (msg, field) => {
  const val = msg[field]?.[0];
  return val == null ? null : (Buffer.isBuffer(val) ? val.toString("utf8") : val);
};

// ------------------------------------------------------------- 3. 跑一次识别

function readWav(path) {
  const buf = fs.readFileSync(path);
  if (buf.toString("ascii", 0, 4) !== "RIFF") throw new Error("不是 WAV 文件");
  let off = 12;
  while (off + 8 <= buf.length) {
    const id = buf.toString("ascii", off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    if (id === "fmt ") {
      const channels = buf.readUInt16LE(off + 10);
      const rate = buf.readUInt32LE(off + 12);
      const bits = buf.readUInt16LE(off + 22);
      if (channels !== 1 || rate !== SAMPLE_RATE || bits !== 16) {
        throw new Error(`需要 16kHz/单声道/16bit 的 WAV，实际 ${rate}Hz/${channels}ch/${bits}bit`);
      }
    }
    if (id === "data") {
      return buf.subarray(off + 8, off + 8 + size);
    }
    off += 8 + size + (size % 2);
  }
  throw new Error("WAV 里没有 data 段");
}

async function transcribeOnce(path) {
  const pcm = readWav(path);
  const { deviceId, appKey } = await getCredentials();
  const requestId = crypto.randomUUID();
  const ws = new WebSocket(`${WS_URL}?aid=${AID}&device_id=${deviceId}`, {
    headers: { "User-Agent": UA, "proto-version": "v2", "x-custom-keepalive": "true" },
  });

  let text = "";
  let finished = false;
  let failure = null;
  const queue = [];
  const waiters = [];
  let closed = false;

  ws.on("message", (data) => {
    const msg = decode(Buffer.from(data));
    if (waiters.length) waiters.shift()(msg); else queue.push(msg);
  });
  ws.on("error", (err) => { failure = failure || err; closed = true; while (waiters.length) waiters.shift()(null); });
  ws.on("close", () => { closed = true; while (waiters.length) waiters.shift()(null); });

  const nextMessage = () => new Promise((resolve, reject) => {
    if (queue.length) return resolve(queue.shift());
    if (closed) return reject(new Error("连接已关闭"));
    const timer = setTimeout(() => reject(new Error("等待上游响应超时")), 25000);
    waiters.push((msg) => { clearTimeout(timer); msg ? resolve(msg) : reject(new Error("连接已关闭")); });
  });

  await new Promise((resolve, reject) => {
    ws.once("open", resolve);
    ws.once("error", () => reject(new Error("WebSocket 连接失败")));
  });

  // ① StartTask
  ws.send(encode({ token: appKey, method: "StartTask", requestId }));
  let msg = await nextMessage();
  if (textOf(msg, 4) !== "TaskStarted") {
    throw new Error(`StartTask 失败: ${textOf(msg, 4)} ${msg[5]} ${textOf(msg, 6)}`);
  }

  // ② StartSession
  const sessionPayload = JSON.stringify({
    audio_info: { channel: 1, format: "raw", sample_rate: SAMPLE_RATE },
    enable_punctuation: true,
    enable_speech_rejection: false,
    extra: {
      did: deviceId, input_mode: "tool",
      strong_ddc: true, use_twopass_retry: true,
      enable_asr_twopass: true, enable_asr_threepass: true,
      disable_user_words: false, cell_compress_rate: 8,
    },
  });
  ws.send(encode({ token: appKey, method: "StartSession", payload: sessionPayload, requestId }));
  msg = await nextMessage();
  if (textOf(msg, 4) !== "SessionStarted") {
    throw new Error(`StartSession 失败: ${textOf(msg, 4)} ${msg[5]} ${textOf(msg, 6)}`);
  }

  // ③ 边收结果边发音频
  const collector = (async () => {
    while (!finished) {
      let incoming;
      try {
        incoming = await nextMessage();
      } catch (err) { failure = failure || err; return; }
      if (!incoming) { failure = failure || new Error("连接提前关闭"); return; }
      const kind = textOf(incoming, 4);
      const result = textOf(incoming, 7);
      if (result) {
        const data = JSON.parse(result);
        for (const item of data.results || []) {
          if (item.text) text = item.text;        // 累积文本，最后一条即完整结果
        }
      }
      if (kind === "SessionFinished" || kind === "TaskFinished") finished = true;
      if (kind === "SessionFailed" || kind === "TaskFailed") {
        failure = new Error(`识别失败: ${incoming[5]} ${textOf(incoming, 6)}`);
        finished = true;
      }
    }
  })();

  for (let i = 0; i < pcm.length; i += FRAME_BYTES) {
    const frame = pcm.subarray(i, i + FRAME_BYTES);
    const state = i === 0 ? 1 : (i + FRAME_BYTES >= pcm.length ? 9 : 3);
    ws.send(encode({ method: "TaskRequest", payload: "{}", audio: frame, requestId, state }));
    await new Promise((r) => setTimeout(r, 20));         // 按实时速度喂
  }

  // ④ FinishSession
  ws.send(encode({ token: appKey, method: "FinishSession", requestId }));
  await Promise.race([collector, new Promise((r) => setTimeout(r, 25000))]);
  ws.close();
  if (failure) throw failure;
  return text;
}

const isRetryable = (err) => /service discovery|\[5\d{7}/.test(String(err));

async function transcribe(path, retries = 4) {
  for (let attempt = 0; ; attempt++) {
    try {
      return await transcribeOnce(path);
    } catch (err) {
      if (!isRetryable(err) || attempt >= retries - 1) throw err;
      console.error(`  [重试 ${attempt + 1}/${retries - 1}] ${err.message}`);
      await new Promise((r) => setTimeout(r, 600));
    }
  }
}

const file = process.argv[2];
if (!file) {
  console.error("用法: node asr_min.js <16k-mono-16bit.wav>");
  process.exit(1);
}
transcribe(file).then((text) => console.log(text))
  .catch((err) => { console.error("失败:", err.message); process.exit(2); });
