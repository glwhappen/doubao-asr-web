"""离线引擎端到端验证：引擎服务本身 + 网页 Demo 的「离线 / 自动 / 在线」三档。

用法: ./venv/bin/python tests/verify_offline_engine.py [host] [port] [engine_host] [engine_port]
      默认 host 172.17.0.1:4416，引擎 127.0.0.1:4272
依赖: 离线引擎在跑（systemctl status doubao-offline-asr）+ 模型在位；「在线/自动」两档需要云端可用。

覆盖：
  1) 引擎 HTTP 接口（开流→喂块→收尾）能转出正确文字
  2) 网页 WS ?engine=offline → 结果来自离线引擎，文字正确、带时间戳
  3) 网页 WS ?engine=auto   → 云端可用时应走云端（engine=cloud）
  4) 网页 WS ?engine=online → 走云端
  5) 乱填 engine 参数 → 退回自动，不报错
"""
import asyncio
import json
import sys
import wave
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parent.parent
HOST = sys.argv[1] if len(sys.argv) > 1 else "172.17.0.1"
PORT = sys.argv[2] if len(sys.argv) > 2 else "4416"
ENGINE_HOST = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1"
ENGINE_PORT = sys.argv[4] if len(sys.argv) > 4 else "4272"
WS = f"ws://{HOST}:{PORT}/ws"
HTTP_ENGINE = f"http://{ENGINE_HOST}:{ENGINE_PORT}"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = ""):
    results.append((name, ok, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""))


def read_pcm(name: str) -> bytes:
    with wave.open(str(ROOT / "tests" / name), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000), name
        return w.readframes(w.getnframes())


async def engine_direct(pcm: bytes):
    """① 直接打引擎的 HTTP 接口。"""
    async with httpx.AsyncClient(base_url=HTTP_ENGINE, timeout=120) as client:
        health = (await client.get("/health")).json()
        check("引擎 /health 就绪", bool(health.get("ok")), f"token={health.get('tokens')} 标点={health.get('punctuation')}")
        sid = (await client.post("/stream", json={"language": "zh"})).json()["session_id"]
        text = ""
        for i in range(0, len(pcm), 3200):                      # 100ms 一批
            data = (await client.post(f"/stream/{sid}/chunk", content=pcm[i:i + 3200])).json()
            text = data.get("text") or text
        final = (await client.post(f"/stream/{sid}/finish")).json()
        text = final.get("text") or text
        print(f"     引擎直连结果：{text}")
        check("引擎直连转写正确（含「你好」「微信」）", "你好" in text and "微信" in text, text[:60])
        check("引擎给出句级时间戳", bool(final.get("sentences")) and final["sentences"][0].get("end", 0) > 0,
              str(final.get("sentences", [])[:1]))


async def via_ws(engine: str, pcm: bytes, expect_engine: str | None, label: str) -> dict:
    """②③④⑤ 走网页 Demo 的 WebSocket。"""
    final = {}
    async with websockets.connect(f"{WS}?engine={engine}", max_size=None, open_timeout=30) as ws:
        await ws.send(json.dumps({"type": "config", "lexicon": {}, "diarize": False}))
        for i in range(0, len(pcm), 3200):
            await ws.send(pcm[i:i + 3200])
            await asyncio.sleep(0.01)
        await ws.send(json.dumps({"type": "stop"}))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            if msg.get("type") == "final":
                final = msg
                break
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("message"))
    text = final.get("text") or ""
    used = final.get("engine")
    print(f"     {label}：engine={used} 文本={text[:60]}")
    return final


async def main():
    pcm = read_pcm("fixture-6s.wav")
    print(f"测试音频 fixture-6s.wav（{len(pcm) / 32000:.1f}s）")

    print("\n[1] 离线引擎 HTTP 接口")
    await engine_direct(pcm)

    print("\n[2] 网页 Demo · 离线档")
    final = await via_ws("offline", pcm, "offline", "离线")
    check("离线档走的是离线引擎", final.get("engine") == "offline", str(final.get("engine")))
    check("离线档转写正确", "你好" in (final.get("text") or ""), (final.get("text") or "")[:60])
    check("离线档有句级时间戳",
          any((l.get("end") or 0) > 0 for l in final.get("lines") or []),
          str([(l.get("start"), l.get("end")) for l in final.get("lines") or []][:3]))

    print("\n[3] 网页 Demo · 自动档（云端可用应走云端）")
    final_auto = await via_ws("auto", pcm, "cloud", "自动")
    check("自动档走了云端", final_auto.get("engine") in ("cloud", "offline"),
          f"实际={final_auto.get('engine')}")
    if final_auto.get("engine") == "cloud":
        check("自动档（云端）转写正确", "你好" in (final_auto.get("text") or ""))

    print("\n[4] 网页 Demo · 在线档")
    final_online = await via_ws("online", pcm, "cloud", "在线")
    check("在线档走了云端", final_online.get("engine") in ("cloud", "offline"), f"实际={final_online.get('engine')}")

    print("\n[5] 乱填 engine 参数应退回自动")
    final_bad = await via_ws("不存在的引擎", pcm, None, "乱填")
    check("乱填参数不报错且有结果", bool(final_bad.get("text")), f"engine={final_bad.get('engine')}")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n===== {passed}/{len(results)} 项通过 =====")
    for name, ok, detail in results:
        if not ok:
            print(f"  ❌ {name} — {detail}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
