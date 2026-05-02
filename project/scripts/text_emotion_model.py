from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Chinese-Emotion-Small"
DEFAULT_DEVICE = os.getenv("TEXT_EMOTION_DEVICE", os.getenv("SENSEVOICE_DEVICE", "cuda:0"))
MODEL_PATH = Path(os.getenv("TEXT_EMOTION_MODEL", str(DEFAULT_MODEL_PATH)))
MAX_LENGTH = int(os.getenv("TEXT_EMOTION_MAX_LENGTH", "256"))

RAW_LABELS = {
    0: "Neutral tone",
    1: "Concerned tone",
    2: "Happy tone",
    3: "Angry tone",
    4: "Sad tone",
    5: "Questioning tone",
    6: "Surprised tone",
    7: "Disgusted tone",
}

LABEL_TO_4CLASS = {
    "Neutral tone": "neutral",
    "Concerned tone": "sad",
    "Happy tone": "happy",
    "Angry tone": "angry",
    "Sad tone": "sad",
    "Questioning tone": "neutral",
    "Surprised tone": "neutral",
    "Disgusted tone": "angry",
}

MODEL_NAME = "Johnson8187/Chinese-Emotion-Small"

_MODEL: Any = None
_TOKENIZER: Any = None
_TORCH: Any = None
_LOAD_ERROR: Optional[str] = None


def _load() -> tuple[Any, Any, Any]:
    global _MODEL, _TOKENIZER, _TORCH, _LOAD_ERROR
    if _MODEL is not None and _TOKENIZER is not None and _TORCH is not None:
        return _MODEL, _TOKENIZER, _TORCH
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = torch.device(DEFAULT_DEVICE if torch.cuda.is_available() and "cuda" in DEFAULT_DEVICE else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
        )
        model.to(device)
        model.eval()
        _TORCH = torch
        _TOKENIZER = tokenizer
        _MODEL = model
        _LOAD_ERROR = None
        return model, tokenizer, torch
    except Exception as exc:
        _LOAD_ERROR = str(exc)
        raise


def is_available() -> bool:
    if not MODEL_PATH.exists():
        return False
    try:
        _load()
        return True
    except Exception:
        return False


def get_load_error() -> Optional[str]:
    return _LOAD_ERROR


def predict_text_emotion(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {
            "emotion": "neutral",
            "confidence": 1.0,
            "confidence_type": "empty_text_default",
            "raw_label": "Neutral tone",
            "raw_scores": {"Neutral tone": 1.0},
            "source": MODEL_NAME,
            "model_path": str(MODEL_PATH),
        }

    model, tokenizer, torch = _load()
    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu()

    scores: List[float] = [float(v) for v in probs.tolist()]
    best_idx = int(max(range(len(scores)), key=lambda idx: scores[idx]))
    raw_label = RAW_LABELS.get(best_idx, f"LABEL_{best_idx}")
    raw_scores = {
        RAW_LABELS.get(idx, f"LABEL_{idx}"): round(score, 6)
        for idx, score in enumerate(scores)
    }
    return {
        "emotion": LABEL_TO_4CLASS.get(raw_label, "neutral"),
        "confidence": round(scores[best_idx], 6),
        "confidence_type": "softmax_probability",
        "raw_label": raw_label,
        "raw_scores": raw_scores,
        "source": MODEL_NAME,
        "model_path": str(MODEL_PATH),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chinese text emotion classification")
    parser.add_argument("text", nargs="?", default="", help="text to classify")
    parser.add_argument("--text", dest="text_arg", default="", help="text to classify")
    parser.add_argument("--text-file", default="", help="UTF-8 text file to classify")
    parser.add_argument("--model", default="", help="model path, overrides TEXT_EMOTION_MODEL")
    parser.add_argument("--device", default="", help="device, overrides TEXT_EMOTION_DEVICE, e.g. cuda:0/cpu")
    parser.add_argument("--max-length", type=int, default=0, help="tokenizer max length")
    parser.add_argument("--compact", action="store_true", help="print compact JSON")
    return parser.parse_args()


def main() -> int:
    global MODEL_PATH, DEFAULT_DEVICE, MAX_LENGTH
    args = parse_args()
    if args.model:
        MODEL_PATH = Path(args.model)
    if args.device:
        DEFAULT_DEVICE = args.device
    if args.max_length > 0:
        MAX_LENGTH = args.max_length

    text = args.text_arg or args.text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
    if not text.strip():
        raise SystemExit("missing text: pass positional text, --text, or --text-file")

    result = predict_text_emotion(text)
    print(json.dumps(result, ensure_ascii=False, indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
