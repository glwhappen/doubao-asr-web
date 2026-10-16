"""验证声纹库：注册一个人的声纹后，下一次录音能自动认出他。

用法: ./venv/bin/python tests/verify_voiceprint.py [host] [port]
依赖: 服务在跑 + 装了说话人分离模型（tools/fetch_diarization_models.sh）
音频现场合成：中文女声（fixture-6s）+ 静音 + 英文男声（fixture-en-jfk）。

流程：第一遍不注册 → 从 final 里拿每个说话人的声纹向量 → 把第一个注册成「甲」
      → 第二遍带上声纹库 → 断言中文那句被认成「甲」、英文那句仍不认识。
"""
import asyncio
import json
import sys
import wave
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
HOST = sys.argv[1] if len(sys.argv) > 1 else "172.17.0.1"
PORT = sys.argv[2] if len(sys.argv) > 2 else "4416"


def build_pcm() -> bytes:
    pcm = b""
    for name in ("fixture-6s.wav", "fixture-en-jfk.wav"):
        with wave.open(str(ROOT / "tests" / name), "rb") as w:
            pcm += w.readframes(w.getnframes())
        pcm += b"\x00\x00" * 16000
    return pcm


async def run(pcm: bytes, voiceprints=None, label=""):
    final = None
    async with websockets.connect(f"ws://{HOST}:{PORT}/ws?engine=auto", max_size=None, open_timeout=30) as ws:
        config = {"type": "config", "lexicon": {}, "diarize": True}
        if voiceprints:
            config["voiceprints"] = voiceprints
        await ws.send(json.dumps(config))

        async def drain():
            nonlocal final
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("type") in ("final", "error"):
                    final = msg
                    return

        task = asyncio.create_task(drain())
        for i in range(0, len(pcm), 640):
            await ws.send(pcm[i:i + 640])
            await asyncio.sleep(0.02)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(task, timeout=180)

    assert final and final.get("type") == "final", f"[{label}] 没拿到最终结果: {final}"
    print(f"\n[{label}] speakers={final.get('speakers')}")
    for line in final.get("lines") or []:
        print(f"   spk={line.get('spk')} who={line.get('who')!r}  {line.get('text','')[:36]}")
    return final


async def main():
    pcm = build_pcm()
    print(f"测试音频 {len(pcm) / 32000:.1f}s（中文女声 + 静音 + 英文男声）")

    first = await run(pcm, None, "第一遍：不注册声纹")
    vectors = first.get("speaker_vectors") or {}
    assert len(vectors) >= 2, f"应该给出至少 2 个人的声纹向量，实际 {list(vectors)}"
    dims = {len(v) for v in vectors.values()}
    assert len(dims) == 1 and dims.pop() >= 64, f"声纹向量维度不对：{dims}"
    print(f"   → 拿到 {len(vectors)} 条声纹（每条 {len(next(iter(vectors.values())))} 维）")

    # 把中文女声那位注册成「甲」
    target_spk = (first["lines"][0].get("spk"))
    assert target_spk is not None, "第一句没有说话人标签"
    gallery = [{"name": "甲", "vectors": [vectors[str(target_spk)]]}]
    second = await run(pcm, gallery, "第二遍：带上声纹库（甲）")

    lines = second.get("lines") or []
    assert lines, "第二遍没有结果"
    known = [l for l in lines if l.get("who") == "甲"]
    assert known, "没有认出注册过的「甲」"
    assert known[0]["text"].startswith("你好"), f"认错人了：{known[0]['text']}"
    strangers = [l for l in lines if l.get("who") is None]
    assert strangers, "陌生人也应该保持未命名（不能全都硬塞成「甲」）"
    print("\n✅ 通过：注册过的声音被认出来（跨会话），陌生人保持未命名")


if __name__ == "__main__":
    asyncio.run(main())
