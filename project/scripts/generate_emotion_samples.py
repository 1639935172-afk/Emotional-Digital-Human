from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "examples"
DEFAULT_TMP_DIR = PROJECT_ROOT / "outputs" / "_tmp_emotion_samples"
TTS_SCRIPT = PROJECT_ROOT / "scripts" / "tts_qwen3_from_pipeline.py"

EMOTION_TEXTS: Dict[str, str] = {
    "happy": "太好了，听到这个消息我也很开心，继续保持这种状态！",
    "sad": "听起来你有点难受，我在这里陪你，我们可以慢慢来。",
    "angry": "我理解你现在很生气，我们先缓一下，我会认真听你说。",
    "neutral": "我在这儿，已经听到你的信息了，我们一步一步处理就好。",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 4 emotion wav samples via tts_qwen3_from_pipeline.py")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    p.add_argument("--tmp-dir", default=str(DEFAULT_TMP_DIR), help="临时 pipeline json 目录")
    p.add_argument("--device", default="cuda:0", help="TTS 设备")
    p.add_argument("--use-voice-id", default="", help="复用已保存 voice_id（推荐）")
    p.add_argument("--ref-audio", default="", help="首次构建音色时的参考音频")
    p.add_argument("--ref-text", default="", help="首次构建音色时的参考文本")
    p.add_argument("--voice-id", default="demo_voice", help="保存/构建时使用的 voice_id")
    p.add_argument("--build-voice-cache", action="store_true", help="是否构建并保存 voice cache")
    p.add_argument("--use-voice-cache", action="store_true", help="是否优先使用 voice cache")
    p.add_argument("--timeout-seconds", type=int, default=300, help="单条样本生成超时秒数")
    return p.parse_args()


def _build_pipeline_json(path: Path, emotion: str, text: str) -> None:
    payload = {
        "task3_tts_control": {
            "text": text,
            "emotion": emotion,
            "tts_params": {},
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_cmd(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"命令执行失败，退出码 {completed.returncode}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tmp_dir = Path(args.tmp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if not TTS_SCRIPT.exists():
        raise SystemExit(f"脚本不存在: {TTS_SCRIPT}")

    if not args.use_voice_id and not args.ref_audio:
        raise SystemExit("未传 --use-voice-id 时，必须提供 --ref-audio。")
    if not args.use_voice_id and not args.ref_text:
        raise SystemExit("未传 --use-voice-id 时，必须提供 --ref-text。")

    manifest = {"generated": []}
    for emotion, text in EMOTION_TEXTS.items():
        pipeline_json = tmp_dir / f"pipeline_{emotion}.json"
        out_wav = output_dir / f"{emotion}.wav"
        _build_pipeline_json(pipeline_json, emotion, text)

        cmd = [
            sys.executable,
            str(TTS_SCRIPT),
            "--pipeline-json",
            str(pipeline_json),
            "--output",
            str(out_wav),
            "--device",
            args.device,
            "--timeout-seconds",
            str(args.timeout_seconds),
        ]

        if args.use_voice_id:
            cmd += ["--use-voice-id", args.use_voice_id]
        else:
            cmd += ["--ref-audio", args.ref_audio, "--ref-text", args.ref_text, "--voice-id", args.voice_id]
            if args.build_voice_cache:
                cmd.append("--build-voice-cache")

        if args.use_voice_cache:
            cmd.append("--use-voice-cache")

        _run_cmd(cmd)
        manifest["generated"].append({"emotion": emotion, "text": text, "wav": str(out_wav)})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 已生成 4 类情感样本: {output_dir}")
    print(f"[OK] 清单文件: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
