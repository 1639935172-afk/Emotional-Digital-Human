#python "D:\0digi-human\project\scripts\pipeline_e2e_demo.py" --audio "D:\0digi-human\project\samples\20260415_193009.m4a" --device "cuda:0" --disable-update --llm-url "http://127.0.0.1:8080/v1/chat/completions" --llm-model "Qwen3-8B-Q4_K_M.gguf"
#python "D:\0digi-human\project\scripts\pipeline_e2e_demo.py" --audio "D:\0digi-human\project\samples\20260415_193009.m4a" --device "cuda:0" --disable-update --llm-url "http://127.0.0.1:8080/v1/chat/completions" --llm-model "Qwen3-8B-Q4_K_M.gguf" --output "D:\0digi-human\project\outputs\pipeline_e2e_result.json"
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.request import Request, urlopen

from funasr import AutoModel

EMO_PATTERN = re.compile(r"<\|([A-Z_]+)\|>")
TEXT_TAG_PREFIX = re.compile(r"^(?:<\|[^|]+\|>)+")
JSON_BLOCK = re.compile(r"\{.*\}", flags=re.DOTALL)

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

NEGATIVE_TERMINATION_PATTERNS = [
    "别说了",
    "不用再说了",
    "就这样吧",
    "别理我",
    "烦",
    "生气",
    "火大",
    "闭嘴",
]
HAPPY_PATTERNS = ["高兴", "开心", "太棒了", "哈哈", "开心", "不错"]
SAD_PATTERNS = ["难过", "低落", "烦", "压力", "累", "不想说话"]
ANGRY_PATTERNS = ["生气", "火大", "忍不了", "烦死", "气死", "别烦我"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio -> SenseVoice -> Rule Fusion -> LLM JSON")
    parser.add_argument("--audio", required=True, help="音频文件路径")
    parser.add_argument("--device", default="cuda:0", help="SenseVoice 推理设备")
    parser.add_argument("--language", default="auto", help="auto/zh/en/yue/ja/ko/nospeech")
    parser.add_argument("--sv-model", default="iic/SenseVoiceSmall", help="SenseVoice 模型")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8080/v1/chat/completions", help="llama-server OpenAI 接口")
    parser.add_argument("--llm-model", default="Qwen3-8B-Q4_K_M.gguf", help="llama-server model 字段")
    parser.add_argument("--max-tokens", type=int, default=128, help="LLM 最大输出 tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM 温度")
    parser.add_argument("--timeout", type=int, default=60, help="LLM 请求超时秒数")
    parser.add_argument("--output", default="", help="可选：输出 JSON 路径")
    parser.add_argument("--disable-update", action="store_true", help="禁用 funasr 更新检查")
    return parser.parse_args()


def build_tts_params(emotion: str) -> Dict[str, Any]:
    # 任务三：将任务二输出的情感标签映射为可控声学参数
    presets: Dict[str, Dict[str, Any]] = {
        "neutral": {
            "style": "neutral",
            "speed": 1.0,
            "pitch_semitone": 0.0,
            "energy": 1.0,
            "emotion_intensity": 0.45,
            "pause_ms": 180,
        },
        "happy": {
            "style": "happy",
            "speed": 1.08,
            "pitch_semitone": 1.8,
            "energy": 1.1,
            "emotion_intensity": 0.75,
            "pause_ms": 130,
        },
        "sad": {
            "style": "sad",
            "speed": 0.9,
            "pitch_semitone": -1.2,
            "energy": 0.82,
            "emotion_intensity": 0.68,
            "pause_ms": 240,
        },
        "angry": {
            "style": "angry_calm",
            "speed": 1.02,
            "pitch_semitone": 0.8,
            "energy": 1.16,
            "emotion_intensity": 0.8,
            "pause_ms": 110,
        },
    }
    return presets.get(emotion, presets["neutral"])


def parse_sensevoice_result(raw_text: str) -> Tuple[str, str, str]:
    tags = EMO_PATTERN.findall(raw_text or "")
    emo_raw = next((t for t in tags if t in EMO_TO_4CLASS), "EMO_UNK")
    audio_emotion = EMO_TO_4CLASS.get(emo_raw, "neutral")
    asr_text = TEXT_TAG_PREFIX.sub("", raw_text or "").strip()
    return asr_text, emo_raw, audio_emotion


def detect_text_emotion(text: str) -> Tuple[str, float]:
    t = (text or "").strip()
    if not t:
        return "neutral", 0.5
    for kw in ANGRY_PATTERNS:
        if kw in t:
            return "angry", 0.85
    for kw in SAD_PATTERNS:
        if kw in t:
            return "sad", 0.8
    for kw in HAPPY_PATTERNS:
        if kw in t:
            return "happy", 0.8
    return "neutral", 0.6


def has_negative_termination(text: str) -> bool:
    t = (text or "").strip()
    return any(p in t for p in NEGATIVE_TERMINATION_PATTERNS)


def fuse_emotion(audio_emotion: str, text_emotion: str, text_conf: float, text: str) -> Tuple[str, str]:
    # 任务1核心规则：语音 neutral 且文本负向/终止语义时，优先文本情感
    if audio_emotion == "neutral" and text_emotion in {"sad", "angry"} and (has_negative_termination(text) or text_conf >= 0.75):
        return text_emotion, "rule:audio_neutral_text_negative_override"
    if audio_emotion == text_emotion:
        return audio_emotion, "rule:audio_text_agree"
    # 默认偏向文本（更贴近语义意图）
    if text_conf >= 0.75:
        return text_emotion, "rule:text_high_confidence"
    return audio_emotion, "rule:audio_fallback"


def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = JSON_BLOCK.search(text)
        if not match:
            raise ValueError("LLM 输出未包含 JSON")
        return json.loads(match.group(0))


def call_llm_reply(
    llm_url: str,
    llm_model: str,
    final_emotion: str,
    asr_text: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    system_prompt = (
        "/no_think 你是共情对话引擎。"
        "基于用户情感生成一句中文共情回复。"
        "你必须只输出一行严格合法 JSON，不要 markdown，不要解释。"
        'schema: {"emotion":"happy|sad|angry|neutral","reply_text":"string"}。'
        "其中 emotion 必须等于给定 final_emotion。"
    )
    user_prompt = f"final_emotion={final_emotion}; user_text={asr_text}"
    payload = {
        "model": llm_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = Request(
        url=llm_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8", errors="replace"))
    content = body["choices"][0]["message"]["content"]
    parsed = extract_json(content)
    emotion = str(parsed.get("emotion", "")).strip().lower()
    reply_text = str(parsed.get("reply_text", "")).strip()
    if emotion not in {"happy", "sad", "angry", "neutral"}:
        emotion = final_emotion
    if not reply_text:
        reply_text = "我在，愿意听你说。"
    return {"emotion": emotion, "reply_text": reply_text}


def main() -> int:
    args = parse_args()
    audio = Path(args.audio)
    if not audio.exists():
        raise SystemExit(f"音频不存在: {audio}")

    sv_model = AutoModel(
        model=args.sv_model,
        trust_remote_code=False,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=args.device,
        disable_update=args.disable_update,
    )
    sv_res = sv_model.generate(
        input=str(audio),
        cache={},
        language=args.language,
        use_itn=True,
        ban_emo_unk=False,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    first = sv_res[0] if sv_res else {}
    raw_text = str(first.get("text", ""))
    asr_text, audio_emo_raw, audio_emotion = parse_sensevoice_result(raw_text)
    text_emotion, text_conf = detect_text_emotion(asr_text)
    final_emotion, fusion_reason = fuse_emotion(audio_emotion, text_emotion, text_conf, asr_text)

    llm_out = call_llm_reply(
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        final_emotion=final_emotion,
        asr_text=asr_text,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    result = {
        "audio_path": str(audio),
        "task1_emotion_arbitration": {
            "asr_text": asr_text,
            "audio_emotion_raw": audio_emo_raw,
            "audio_emotion": audio_emotion,
            "text_emotion": text_emotion,
            "text_confidence": text_conf,
            "final_emotion": final_emotion,
            "fusion_reason": fusion_reason,
        },
        "task2_empathic_reply": {
            "emotion": llm_out["emotion"],
            "reply_text": llm_out["reply_text"],
        },
        "task3_tts_control": {
            "text": llm_out["reply_text"],
            "emotion": llm_out["emotion"],
            "tts_params": build_tts_params(llm_out["emotion"]),
        },
    }

    out = json.dumps(result, ensure_ascii=False, indent=2)
    print(out)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
        print(f"\n已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

