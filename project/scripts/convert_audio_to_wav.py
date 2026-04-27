from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

SUPPORTED_EXTS = {".m4a", ".mp3", ".flac", ".ogg", ".aac", ".wma", ".mp4", ".wav"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    default_input = project_dir / "samples" / "20260415_193009.m4a"
    default_output_dir = project_dir / "samples"

    p = argparse.ArgumentParser(description="Convert audio file(s) to wav via ffmpeg")
    p.add_argument(
        "--input",
        default=str(default_input),
        help="输入音频文件或目录（相对路径从当前工作目录解析）",
    )
    p.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help="输出目录（默认 project/samples）",
    )
    p.add_argument("--sample-rate", type=int, default=16000, help="输出采样率，默认 16000")
    p.add_argument("--channels", type=int, default=1, help="输出声道数，默认 1（单声道）")
    p.add_argument(
        "--suffix",
        default="_16k",
        help="输出文件名后缀（不含扩展名），默认 _16k",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在输出文件",
    )
    return p.parse_args()


def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
        return sorted(files)
    raise FileNotFoundError(f"输入路径不存在: {input_path}")


def convert_one(src: Path, dst: Path, sample_rate: int, channels: int, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(src),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    inputs = collect_inputs(input_path)
    if not inputs:
        raise SystemExit("没有找到可转换的音频文件。")

    ok = 0
    for src in inputs:
        out_name = f"{src.stem}{args.suffix}.wav"
        dst = output_dir / out_name
        try:
            convert_one(src, dst, args.sample_rate, args.channels, args.overwrite)
            print(f"[OK] {src} -> {dst}")
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {src} -> {dst} | ffmpeg exit={e.returncode}")

    print(f"done: {ok}/{len(inputs)} converted")
    return 0 if ok == len(inputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

