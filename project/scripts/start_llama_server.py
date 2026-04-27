"""
启动本地 llama-server（Windows 友好版）。

用途：
1) 固化 Day1-2 使用的服务参数；
2) 避免每次手敲长命令；
3) 启动前检查可执行文件和模型文件是否存在。

示例：
  python scripts/start_llama_server.py
  python scripts/start_llama_server.py --port 8081 --ctx 4096
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_SERVER_EXE = r"D:\Tools\llama-b8797-bin-win-cuda-12.4-x64\llama-server.exe"
FALLBACK_MODEL = r"D:\0digi-human\project\models\Qwen3-8B-GGUF\Qwen3-8B-Q4_K_M.gguf"


def resolve_default_server_exe() -> str:
    env_path = os.getenv("LLAMA_SERVER_EXE", "").strip()
    if env_path:
        return env_path
    # 常见放置方式：与脚本同级项目目录下的 tools
    candidate = PROJECT_ROOT / "tools" / "llama-server.exe"
    if candidate.exists():
        return str(candidate)
    return FALLBACK_SERVER_EXE


def resolve_default_model() -> str:
    env_path = os.getenv("LLAMA_MODEL_PATH", "").strip()
    if env_path:
        return env_path
    candidate = PROJECT_ROOT / "models" / "Qwen3-8B-GGUF" / "Qwen3-8B-Q4_K_M.gguf"
    if candidate.exists():
        return str(candidate)
    return FALLBACK_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start local llama-server for Qwen3 GGUF")
    parser.add_argument(
        "--server-exe",
        default=resolve_default_server_exe(),
        help="llama-server.exe 路径（优先读取 LLAMA_SERVER_EXE）",
    )
    parser.add_argument(
        "--model",
        default=resolve_default_model(),
        help="GGUF 模型路径（优先读取 LLAMA_MODEL_PATH）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--ctx", type=int, default=4096, help="上下文长度（n_ctx）")
    parser.add_argument(
        "--ngl",
        type=int,
        default=99,
        help="GPU offload 层数；默认 99，尽量使用 GPU",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印命令，不真正启动",
    )
    parser.set_defaults(open_browser=True)
    parser.add_argument(
        "--open-browser",
        dest="open_browser",
        action="store_true",
        help="启动后自动打开网页（默认开启）",
    )
    parser.add_argument(
        "--no-browser",
        dest="open_browser",
        action="store_false",
        help="启动后不自动打开网页",
    )
    return parser.parse_args()


def ensure_exists(path: Path, title: str) -> None:
    if not path.exists():
        print(f"[ERROR] {title} 不存在: {path}", file=sys.stderr)
        raise SystemExit(1)


def main() -> int:
    args = parse_args()
    server_exe = Path(args.server_exe)
    model = Path(args.model)

    ensure_exists(server_exe, "llama-server")
    ensure_exists(model, "GGUF 模型")

    cmd = [
        str(server_exe),
        "-m",
        str(model),
        "-ngl",
        str(args.ngl),
        "-c",
        str(args.ctx),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    web_url = f"http://{args.host}:{args.port}"
    print("即将启动 llama-server：")
    print(" ".join(shlex.quote(part) for part in cmd))
    print(f"接口地址: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"网页地址: {web_url}")
    print("按 Ctrl+C 停止服务。")

    if args.dry_run:
        return 0

    try:
        if args.open_browser:
            # 给服务一点启动时间，再自动打开默认浏览器。
            threading.Timer(1.2, lambda: webbrowser.open(web_url)).start()
        completed = subprocess.run(cmd, check=False)
        return int(completed.returncode)
    except KeyboardInterrupt:
        print("\n已手动停止。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

