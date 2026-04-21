#python "D:\0digi-human\project\scripts\sensevoice_asr_ser.py" --audio "D:\0digi-human\project\samples\20260415_193009.m4a" --device "cuda:0" --disable-update
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from funasr import AutoModel

EMO_PATTERN = re.compile(r"<\|([A-Z_]+)\|>")
TEXT_TAG_PREFIX = re.compile(r"^(?:<\|[^|]+\|>)+")

EMO_TO_4CLASS = {
    "HAPPY": "happy",
    "SAD": "sad",
    "ANGRY": "angry",
    "NEUTRAL": "neutral",
    "FEARFUL": "sad",
    "DISGUSTED": "angry",
    "SURPRISED": "happy",
    "EMO_UNK": "neutral",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SenseVoice ASR + SER to 4-class JSON")
    parser.add_argument("--audio", required=True, help="音频文件路径")
    parser.add_argument("--device", default="cuda:0", help="推理设备，如 cuda:0 或 cpu")
    parser.add_argument("--language", default="auto", help="auto/zh/en/yue/ja/ko/nospeech")
    parser.add_argument("--model", default="iic/SenseVoiceSmall", help="SenseVoice 模型名或本地路径")
    parser.add_argument("--output", default="", help="可选：输出 JSON 文件路径")
    parser.add_argument("--disable-update", action="store_true", help="禁用 funasr 更新检查")
    return parser.parse_args()


def parse_text_and_emotion(raw_text: str) -> Dict[str, str]:
    tags = EMO_PATTERN.findall(raw_text or "")
    # 情感标签通常在 token 中，如 NEUTRAL/HAPPY/SAD/ANGRY
    emo_raw: Optional[str] = next((t for t in tags if t in EMO_TO_4CLASS), None)
    asr_text = TEXT_TAG_PREFIX.sub("", raw_text or "").strip()
    emo_4class = EMO_TO_4CLASS.get(emo_raw or "EMO_UNK", "neutral")
    return {
        "audio_emotion_raw": emo_raw or "EMO_UNK",
        "audio_emotion_4class": emo_4class,
        "asr_text": asr_text,
    }


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"音频不存在: {audio_path}")

    model = AutoModel(
        model=args.model,
        trust_remote_code=False,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=args.device,
        disable_update=args.disable_update,
    )

    res = model.generate(
        input=str(audio_path),
        cache={},
        language=args.language,
        use_itn=True,
        ban_emo_unk=False,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )

    first: Dict[str, Any] = res[0] if res else {}
    raw_text = str(first.get("text", ""))
    parsed = parse_text_and_emotion(raw_text)

    result = {
        "audio_path": str(audio_path),
        "sensevoice_key": str(first.get("key", audio_path.stem)),
        "raw_text_with_tags": raw_text,
        "asr_text": parsed["asr_text"],
        "audio_emotion_raw": parsed["audio_emotion_raw"],
        "audio_emotion": parsed["audio_emotion_4class"],
    }

    out_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(out_text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text, encoding="utf-8")
        print(f"\n已写入: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

