#!/usr/bin/env python3
"""把原理页模板 + examples 下的示例代码合成为静态页面。

用途: 生成 static/how.html（原理与复刻指南，把 templates/ 与 examples/ 合成静态页）
用法: ./venv/bin/python tools/build_how_page.py
入参: 无。读 templates/how.html 与 examples/asr_min.{py,js,go}
出参: static/how.html（含 HTML 转义后的示例代码）
依赖: 仅标准库
关键词: 文档, 原理页, 生成, build, how, docs, 示例代码内联, escape
更新: 2026-10-16
"""
from html import escape
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TEMPLATE = BASE / "templates" / "how.html"
OUTPUT = BASE / "static" / "how.html"
SAMPLES = {
    "@@PY@@": BASE / "examples" / "asr_min.py",
    "@@JS@@": BASE / "examples" / "asr_min.js",
    "@@GO@@": BASE / "examples" / "asr_min.go",
}


def main():
    page = TEMPLATE.read_text(encoding="utf-8")
    for marker, path in SAMPLES.items():
        code = path.read_text(encoding="utf-8").rstrip("\n")
        if marker not in page:
            raise SystemExit(f"模板里找不到占位符 {marker}")
        page = page.replace(marker, escape(code))
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"已生成 {OUTPUT}（{len(page)} 字节）")


if __name__ == "__main__":
    main()
