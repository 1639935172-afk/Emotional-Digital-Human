from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi import Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from funasr import AutoModel

from pipeline_e2e_demo import (
    build_tts_params,
    detect_text_emotion,
    extract_json,
    fuse_emotion,
    parse_sensevoice_result,
)
import text_emotion_model
from text_emotion_model import get_load_error, predict_text_emotion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "web_demo"
OUTPUT_AUDIO = OUTPUT_ROOT / "audio"
OUTPUT_JSON = OUTPUT_ROOT / "json"
DEFAULT_SESSION_DIR = os.getenv("WEB_DEMO_SESSION_DIR", str(OUTPUT_ROOT / "sessions"))

DEFAULT_SV_MODEL = os.getenv("SENSEVOICE_MODEL", "iic/SenseVoiceSmall")
DEFAULT_DEVICE = os.getenv("SENSEVOICE_DEVICE", "cuda:0")
DEFAULT_LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3-8B-Q4_K_M.gguf")
DEFAULT_LLAMA_SERVER_EXE = os.getenv(
    "LLAMA_SERVER_EXE",
    r"D:\Tools\llama-b8797-bin-win-cuda-12.4-x64\llama-server.exe",
)
DEFAULT_LLAMA_MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "Qwen3-8B-GGUF" / "Qwen3-8B-Q4_K_M.gguf"),
)
DEFAULT_VOICE_ID = os.getenv("WEB_DEMO_DEFAULT_VOICE_ID", "").strip()
DEFAULT_VOICE_PROFILE = Path(
    os.getenv("VOICE_PROFILE_PATH", str(PROJECT_ROOT / "outputs" / "voice_profile.json"))
)
DEFAULT_WEB_REF_AUDIO = os.getenv("WEB_TTS_REF_AUDIO", "").strip()
DEFAULT_WEB_REF_TEXT = os.getenv("WEB_TTS_REF_TEXT", "").strip()
DEFAULT_TEXT_EMOTION_MODEL = os.getenv(
    "TEXT_EMOTION_MODEL",
    str(PROJECT_ROOT / "models" / "Chinese-Emotion-Small"),
)
DEFAULT_TEXT_EMOTION_DEVICE = os.getenv("TEXT_EMOTION_DEVICE", DEFAULT_DEVICE)
DEFAULT_TEXT_EMOTION_WARMUP_TEXT = os.getenv("TEXT_EMOTION_WARMUP_TEXT", "我今天感觉还不错。")

SENSEVOICE_MODEL: Optional[AutoModel] = None
LLAMA_PROCESS: Optional[subprocess.Popen[Any]] = None
MODEL_LOCK = threading.Lock()
SESSION_LOCK = threading.Lock()
SESSION_HISTORY: Dict[str, List[Dict[str, str]]] = {}
MAX_CONTEXT_TURNS = int(os.getenv("WEB_DEMO_MAX_CONTEXT_TURNS", "6"))
VALID_EMOTIONS = {"happy", "sad", "angry", "neutral"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Web demo for multimodal emotional interaction")
    p.add_argument("--host", default="127.0.0.1", help="web host")
    p.add_argument("--port", type=int, default=7860, help="web port")
    p.add_argument("--device", default=DEFAULT_DEVICE, help="SenseVoice device, e.g. cuda:0/cpu")
    p.add_argument("--sv-model", default=DEFAULT_SV_MODEL, help="SenseVoice model id/path")
    p.add_argument("--llm-url", default=DEFAULT_LLM_URL, help="OpenAI-compatible chat completions URL")
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="llm model field")
    p.set_defaults(enable_llm_semantic_emotion=True)
    p.add_argument("--enable-llm-semantic-emotion", dest="enable_llm_semantic_emotion", action="store_true", help="use local LLM for semantic emotion judgement")
    p.add_argument("--disable-llm-semantic-emotion", dest="enable_llm_semantic_emotion", action="store_false", help="disable local LLM semantic emotion judgement")
    p.add_argument("--semantic-max-tokens", type=int, default=int(os.getenv("SEMANTIC_MAX_TOKENS", "160")), help="max tokens for LLM semantic emotion judgement")
    p.add_argument("--llama-server-exe", default=DEFAULT_LLAMA_SERVER_EXE, help="llama-server.exe path")
    p.add_argument("--llama-model-path", default=DEFAULT_LLAMA_MODEL_PATH, help="GGUF model path for auto-started llama-server")
    p.add_argument("--llama-ctx", type=int, default=int(os.getenv("LLAMA_CTX", "4096")), help="llama-server context size")
    p.add_argument("--llama-ngl", type=int, default=int(os.getenv("LLAMA_NGL", "99")), help="llama-server GPU offload layers")
    p.add_argument("--llama-start-timeout", type=int, default=int(os.getenv("LLAMA_START_TIMEOUT", "90")), help="seconds to wait for llama-server")
    p.set_defaults(auto_start_llm=True)
    p.add_argument("--auto-start-llm", dest="auto_start_llm", action="store_true", help="auto start local llama-server")
    p.add_argument("--no-auto-start-llm", dest="auto_start_llm", action="store_false", help="do not auto start llama-server")
    p.add_argument("--disable-update", action="store_true", help="disable funasr update check")
    p.add_argument("--timeout", type=int, default=60, help="llm timeout seconds")
    p.add_argument("--text-emotion-model", default=DEFAULT_TEXT_EMOTION_MODEL, help="local text emotion model path")
    p.add_argument("--text-emotion-device", default=DEFAULT_TEXT_EMOTION_DEVICE, help="text emotion device, e.g. cuda:0/cpu")
    p.add_argument("--text-emotion-warmup-text", default=DEFAULT_TEXT_EMOTION_WARMUP_TEXT, help="warmup text for text emotion model")
    p.add_argument("--no-preload-text-emotion", action="store_true", help="do not preload text emotion model at startup")
    p.add_argument("--session-dir", default=DEFAULT_SESSION_DIR, help="directory for persistent web demo sessions")
    return p.parse_args()


ARGS = parse_args()
text_emotion_model.MODEL_PATH = Path(ARGS.text_emotion_model)
text_emotion_model.DEFAULT_DEVICE = ARGS.text_emotion_device
SESSION_DIR = Path(ARGS.session_dir)


def ensure_dirs() -> None:
    OUTPUT_AUDIO.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.mkdir(parents=True, exist_ok=True)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)


def llama_base_url() -> str:
    parsed = urlparse(ARGS.llm_url)
    if not parsed.scheme or not parsed.netloc:
        return "http://127.0.0.1:8080"
    return f"{parsed.scheme}://{parsed.netloc}"


def is_llama_server_ready(timeout: float = 1.0) -> bool:
    base_url = llama_base_url()
    for path in ("/health", "/v1/models"):
        try:
            with urlopen(f"{base_url}{path}", timeout=timeout) as resp:
                if 200 <= int(resp.status) < 500:
                    return True
        except Exception:
            continue
    return False


def wait_for_llama_server(seconds: int) -> bool:
    deadline = time.time() + max(1, seconds)
    while time.time() < deadline:
        if is_llama_server_ready(timeout=1.5):
            return True
        time.sleep(1.0)
    return False


def auto_start_llama_server() -> None:
    global LLAMA_PROCESS
    if not ARGS.auto_start_llm:
        print("[LLM] auto-start disabled")
        return
    if is_llama_server_ready():
        print(f"[LLM] llama-server already running: {llama_base_url()}")
        return

    server_exe = Path(ARGS.llama_server_exe)
    model_path = Path(ARGS.llama_model_path)
    if not server_exe.exists():
        print(f"[LLM] auto-start skipped: llama-server.exe not found: {server_exe}")
        return
    if not model_path.exists():
        print(f"[LLM] auto-start skipped: GGUF model not found: {model_path}")
        return

    parsed = urlparse(ARGS.llm_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8080
    cmd = [
        str(server_exe),
        "-m",
        str(model_path),
        "-ngl",
        str(ARGS.llama_ngl),
        "-c",
        str(ARGS.llama_ctx),
        "--host",
        host,
        "--port",
        str(port),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    print(f"[LLM] starting llama-server: model={model_path}")
    LLAMA_PROCESS = subprocess.Popen(cmd, creationflags=creationflags)
    atexit.register(stop_auto_started_llama_server)
    if wait_for_llama_server(ARGS.llama_start_timeout):
        print(f"[LLM] llama-server ready: {ARGS.llm_url}")
        return
    print(f"[LLM] llama-server not ready after {ARGS.llama_start_timeout}s; web demo will still start")


def stop_auto_started_llama_server() -> None:
    global LLAMA_PROCESS
    proc = LLAMA_PROCESS
    if proc is None or proc.poll() is not None:
        return
    print("[LLM] stopping auto-started llama-server")
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
    LLAMA_PROCESS = None


def get_sensevoice_model() -> AutoModel:
    global SENSEVOICE_MODEL
    if SENSEVOICE_MODEL is not None:
        return SENSEVOICE_MODEL
    with MODEL_LOCK:
        if SENSEVOICE_MODEL is None:
            SENSEVOICE_MODEL = AutoModel(
                model=ARGS.sv_model,
                trust_remote_code=False,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=ARGS.device,
                disable_update=ARGS.disable_update,
            )
    return SENSEVOICE_MODEL


def run_asr_ser(audio_path: Path) -> Dict[str, str]:
    model = get_sensevoice_model()
    sv_res = model.generate(
        input=str(audio_path),
        cache={},
        language="auto",
        use_itn=True,
        ban_emo_unk=False,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    first = sv_res[0] if sv_res else {}
    raw_text = str(first.get("text", ""))
    asr_text, audio_emo_raw, audio_emotion = parse_sensevoice_result(raw_text)
    return {
        "asr_text": asr_text,
        "sensevoice_text_raw": raw_text,
        "audio_emotion_raw": audio_emo_raw,
        "audio_emotion": audio_emotion,
        "sensevoice_raw_result": json_safe(sv_res),
        "sensevoice_first_result": json_safe(first),
    }


def detect_text_emotion_for_web(text: str) -> Dict[str, Any]:
    try:
        result = predict_text_emotion(text)
        return {
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "confidence_type": result["confidence_type"],
            "raw_label": result["raw_label"],
            "raw_scores": result["raw_scores"],
            "source": result["source"],
            "model_path": result["model_path"],
            "fallback_error": None,
        }
    except Exception as exc:
        rule_emotion, rule_score = detect_text_emotion(text)
        return {
            "emotion": rule_emotion,
            "confidence": None,
            "confidence_type": None,
            "raw_label": None,
            "raw_scores": {},
            "source": "rule_fallback",
            "model_path": None,
            "rule_score": rule_score,
            "fallback_error": str(exc) or get_load_error(),
        }


def preload_text_emotion_model() -> None:
    if ARGS.no_preload_text_emotion:
        print("[TEXT_EMOTION] preload disabled")
        return
    try:
        result = predict_text_emotion(ARGS.text_emotion_warmup_text)
        print(
            "[TEXT_EMOTION] loaded "
            f"model={result.get('model_path')} "
            f"device={text_emotion_model.DEFAULT_DEVICE} "
            f"warmup={result.get('emotion')} "
            f"confidence={result.get('confidence')}"
        )
    except Exception as exc:
        print(
            "[TEXT_EMOTION] preload failed; web demo will use rule_fallback. "
            f"model={text_emotion_model.MODEL_PATH} error={exc}"
        )


def normalize_session_id(session_id: str) -> str:
    session_id = (session_id or "").strip()
    if not session_id:
        return uuid.uuid4().hex
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    cleaned = "".join(ch for ch in session_id if ch in allowed)
    return cleaned[:80] or uuid.uuid4().hex


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def session_json_path(session_id: str) -> Path:
    return SESSION_DIR / f"{normalize_session_id(session_id)}.json"


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def is_relative_to_path(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def delete_session_pipeline_jsons(doc: Dict[str, Any]) -> Dict[str, Any]:
    deleted: List[str] = []
    skipped: List[str] = []
    turns = doc.get("turns", [])
    if not isinstance(turns, list):
        return {"deleted": deleted, "skipped": skipped}
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        full_response = turn.get("full_response", {})
        if not isinstance(full_response, dict):
            continue
        pipeline_json = str(full_response.get("pipeline_json") or "").strip()
        if not pipeline_json:
            continue
        path = Path(pipeline_json)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not is_relative_to_path(path, OUTPUT_JSON):
            skipped.append(str(path))
            continue
        if path.exists() and path.is_file():
            path.unlink()
            deleted.append(str(path))
    return {"deleted": deleted, "skipped": skipped}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


def repair_mojibake_text(text: str) -> str:
    if not text:
        return text
    markers = ("锛", "鍚", "绛", "涓", "浣", "熶", "犵", "殑")
    if not any(marker in text for marker in markers):
        return text
    try:
        repaired = text.encode("gbk", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        return text
    return repaired if repaired else text


def make_session_title(text: str) -> str:
    title = " ".join((text or "").strip().split())
    return (title[:24] if title else "新对话")


def load_session_doc(session_id: str) -> Dict[str, Any]:
    sid = normalize_session_id(session_id)
    now = utc_now_iso()
    doc = read_json_file(session_json_path(sid), {})
    if not isinstance(doc, dict):
        doc = {}
    doc.setdefault("session_id", sid)
    doc.setdefault("title", "新对话")
    doc.setdefault("created_at", now)
    doc.setdefault("updated_at", now)
    doc.setdefault("turns", [])
    if not isinstance(doc.get("turns"), list):
        doc["turns"] = []
    return doc


def save_session_doc(doc: Dict[str, Any]) -> None:
    sid = normalize_session_id(str(doc.get("session_id", "")))
    doc["session_id"] = sid
    doc["updated_at"] = utc_now_iso()
    write_json_file(session_json_path(sid), doc)


def session_index() -> Dict[str, Any]:
    ensure_dirs()
    sessions = []
    for path in SESSION_DIR.glob("*.json"):
        if path.name == "index.json" or path.name.endswith(".tmp"):
            continue
        doc = read_json_file(path, {})
        if not isinstance(doc, dict):
            continue
        turns = doc.get("turns", [])
        sessions.append(
            {
                "session_id": str(doc.get("session_id", path.stem)),
                "title": str(doc.get("title", "新对话")),
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
                "pinned": bool(doc.get("pinned", False)),
                "turn_count": len(turns) if isinstance(turns, list) else 0,
            }
        )
    sessions.sort(
        key=lambda item: (
            1 if item.get("pinned") else 0,
            str(item.get("updated_at") or ""),
        ),
        reverse=True,
    )
    data = {"sessions": sessions}
    write_json_file(SESSION_DIR / "index.json", data)
    return data


def rebuild_context_from_doc(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    turns = doc.get("turns", [])
    if not isinstance(turns, list):
        return history
    for turn in turns[-MAX_CONTEXT_TURNS:]:
        if not isinstance(turn, dict):
            continue
        user = turn.get("user", {}) if isinstance(turn.get("user"), dict) else {}
        assistant = turn.get("assistant", {}) if isinstance(turn.get("assistant"), dict) else {}
        full_response = turn.get("full_response", {}) if isinstance(turn.get("full_response"), dict) else {}
        task1 = full_response.get("task1_emotion_arbitration", {}) if isinstance(full_response.get("task1_emotion_arbitration"), dict) else {}
        task2 = full_response.get("task2_empathic_reply", {}) if isinstance(full_response.get("task2_empathic_reply"), dict) else {}
        history.append(
            {
                "user": str(user.get("text", "")),
                "assistant": str(assistant.get("text", "")),
                "final_emotion": str(task1.get("final_emotion", "neutral")),
                "reply_emotion": str(task2.get("emotion", task1.get("final_emotion", "neutral"))),
            }
        )
    return history


def session_messages_from_doc(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    turns = doc.get("turns", [])
    if not isinstance(turns, list):
        return messages
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        user = turn.get("user", {}) if isinstance(turn.get("user"), dict) else {}
        assistant = turn.get("assistant", {}) if isinstance(turn.get("assistant"), dict) else {}
        full_response = turn.get("full_response", {}) if isinstance(turn.get("full_response"), dict) else {}
        task1 = full_response.get("task1_emotion_arbitration", {}) if isinstance(full_response.get("task1_emotion_arbitration"), dict) else {}
        task2 = full_response.get("task2_empathic_reply", {}) if isinstance(full_response.get("task2_empathic_reply"), dict) else {}
        messages.append(
            {
                "role": "user",
                "text": str(user.get("text", "")),
                "audioName": str(user.get("audio_name", "")),
                "createdAt": turn.get("created_at"),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "text": str(assistant.get("text", "")),
                "emotion": str(task2.get("emotion", task1.get("final_emotion", "neutral"))),
                "audioUrl": full_response.get("tts_audio_url") or "",
                "createdAt": turn.get("created_at"),
                "fullResponse": full_response,
            }
        )
    return messages


def append_persistent_turn(
    session_id: str,
    user_text: str,
    audio_name: str,
    audio_path: Optional[Path],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    sid = normalize_session_id(session_id)
    doc = load_session_doc(sid)
    if not doc.get("turns"):
        doc["title"] = make_session_title(user_text or audio_name)
    llm_out = payload.get("task2_empathic_reply", {}) if isinstance(payload.get("task2_empathic_reply"), dict) else {}
    doc["turns"].append(
        {
            "turn_id": uuid.uuid4().hex,
            "created_at": utc_now_iso(),
            "user": {
                "text": user_text,
                "audio_name": audio_name,
                "audio_path": str(audio_path) if audio_path else None,
            },
            "assistant": {
                "text": str(llm_out.get("reply_text", "")),
                "audio_path": payload.get("tts_audio_url"),
            },
            "full_response": payload,
        }
    )
    save_session_doc(doc)
    session_index()
    return doc


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    with SESSION_LOCK:
        cached = [dict(item) for item in SESSION_HISTORY.get(session_id, [])]
    if cached:
        return cached
    doc = load_session_doc(session_id)
    history = rebuild_context_from_doc(doc)
    if history:
        with SESSION_LOCK:
            SESSION_HISTORY[normalize_session_id(session_id)] = [dict(item) for item in history]
    return history


def append_session_turn(
    session_id: str,
    user_text: str,
    assistant_text: str,
    final_emotion: str,
    reply_emotion: str,
) -> List[Dict[str, str]]:
    turn = {
        "user": user_text,
        "assistant": assistant_text,
        "final_emotion": final_emotion,
        "reply_emotion": reply_emotion,
    }
    with SESSION_LOCK:
        history = SESSION_HISTORY.setdefault(session_id, [])
        history.append(turn)
        if len(history) > MAX_CONTEXT_TURNS:
            del history[: len(history) - MAX_CONTEXT_TURNS]
        return [dict(item) for item in history]


def call_llm_reply_with_history(
    llm_url: str,
    llm_model: str,
    final_emotion: str,
    user_text: str,
    history: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    system_prompt = (
        "/no_think 你是数字人情感交互助手，运行在一个多模态情感交互 Web Demo 中。"
        "你以这个应用助手身份和用户对话，不自称通义千问、Qwen 或语言模型。"
        "按用户当前这句话正常回应，不要生硬套模板。"
        "普通问题就直接回答；表达难受、担心、生气时，先回应情绪，再给一句有用的支持。"
        "回复要结合当前输入，不要重复上一轮建议；20到80个中文字符。"
        "你必须只输出一行严格合法 JSON，不要 markdown，不要解释。"
        'schema: {"emotion":"happy|sad|angry|neutral","reply_text":"string"}。'
        "其中 emotion 必须等于给定 final_emotion。"
    )
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for item in history[-MAX_CONTEXT_TURNS:]:
        prev_user = str(item.get("user", "")).strip()
        prev_assistant = str(item.get("assistant", "")).strip()
        prev_emotion = str(item.get("final_emotion", "")).strip()
        if prev_user:
            messages.append(
                {
                    "role": "user",
                    "content": f"previous_final_emotion={prev_emotion}; user_text={prev_user}",
                }
            )
        if prev_assistant:
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "emotion": item.get("reply_emotion", prev_emotion),
                            "reply_text": prev_assistant,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                {
                    "final_emotion": final_emotion,
                    "user_text": user_text,
                },
                ensure_ascii=False,
            ),
        }
    )
    payload = {
        "model": llm_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    content = ""
    content_source = "content"
    retry_count = 0
    for attempt in range(2):
        req = Request(
            url=llm_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        message = body["choices"][0]["message"]
        content = str(message.get("content") or "")
        content_source = "content"
        if not content.strip() and message.get("reasoning_content"):
            content = str(message.get("reasoning_content") or "")
            content_source = "reasoning_content"
        content = repair_mojibake_text(content)
        if content.strip():
            break
        retry_count = attempt + 1
    parse_error = None
    fallback_used = False
    try:
        parsed = extract_json(content)
    except Exception as exc:
        parse_error = str(exc)
        cleaned = (content or "").strip()
        if cleaned and "{" not in cleaned and "}" not in cleaned:
            parsed = {"emotion": final_emotion, "reply_text": cleaned}
        else:
            fallback_used = True
            parsed = {"emotion": final_emotion, "reply_text": fallback_empathic_reply(final_emotion, user_text)}
    emotion = str(parsed.get("emotion", "")).strip().lower()
    reply_text = str(parsed.get("reply_text", "")).strip()
    if emotion not in VALID_EMOTIONS:
        emotion = final_emotion
    if emotion != final_emotion:
        emotion = final_emotion
    if not reply_text:
        fallback_used = True
        reply_text = fallback_empathic_reply(final_emotion, user_text)
    reply_text = polish_empathic_reply(reply_text, final_emotion, user_text)
    return {
        "emotion": emotion,
        "reply_text": reply_text,
        "llm_raw_content": content,
        "llm_content_source": content_source,
        "llm_parse_error": parse_error,
        "llm_fallback_used": fallback_used,
        "llm_empty_retry_count": retry_count,
    }


def fallback_empathic_reply(emotion: str, user_text: str = "") -> str:
    return "我收到了，请继续说。"


def polish_empathic_reply(reply_text: str, emotion: str, user_text: str = "") -> str:
    return (reply_text or "").strip() or fallback_empathic_reply(emotion, user_text)


def call_llm_semantic_emotion(
    llm_url: str,
    llm_model: str,
    user_text: str,
    history: List[Dict[str, str]],
    timeout: int,
    max_tokens: int,
) -> Dict[str, Any]:
    system_prompt = (
        "/no_think 你是中文语义情绪判断器，只判断用户文本语义，不生成安慰回复。"
        "请识别文本背后的语义情绪、意图、效价和唤醒度。"
        "emotion 只能是 happy、sad、angry、neutral 四选一；担心、焦虑、害怕、压力、失败风险归为 sad；"
        "被冒犯、拒绝、烦躁、指责归为 angry；无明显情绪归为 neutral。"
        "confidence 是你的自评置信度，不是 softmax 概率。"
        "必须只输出一行严格合法 JSON，不要 markdown，不要解释。"
        'schema: {"emotion":"happy|sad|angry|neutral","semantic_confidence":0.0,'
        '"intent":"share|worry|ask_help|complain|reject|end_conversation|other",'
        '"valence":-1.0,"arousal":0.0,"evidence":["string"],"reason":"string"}'
    )
    recent_context = []
    for item in history[-3:]:
        user = str(item.get("user", "")).strip()
        assistant = str(item.get("assistant", "")).strip()
        if user:
            recent_context.append({"user": user, "assistant": assistant})
    user_prompt = json.dumps(
        {
            "current_user_text": user_text,
            "recent_context": recent_context,
        },
        ensure_ascii=False,
    )
    payload = {
        "model": llm_model,
        "temperature": 0.1,
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
    try:
        parsed = extract_json(content)
    except Exception:
        return {
            "emotion": "neutral",
            "semantic_confidence": 0.0,
            "semantic_confidence_type": "llm_self_estimated",
            "intent": "other",
            "valence": 0.0,
            "arousal": 0.0,
            "evidence": [],
            "reason": "LLM semantic output did not contain JSON",
            "source": "local_qwen_semantic_judgement",
            "raw_content": content,
            "error": "missing_json",
        }
    emotion = str(parsed.get("emotion", "neutral")).strip().lower()
    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"
    try:
        semantic_confidence = float(parsed.get("semantic_confidence", 0.0))
    except (TypeError, ValueError):
        semantic_confidence = 0.0
    semantic_confidence = max(0.0, min(1.0, semantic_confidence))
    try:
        valence = float(parsed.get("valence", 0.0))
    except (TypeError, ValueError):
        valence = 0.0
    try:
        arousal = float(parsed.get("arousal", 0.0))
    except (TypeError, ValueError):
        arousal = 0.0
    evidence = parsed.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    return {
        "emotion": emotion,
        "semantic_confidence": round(semantic_confidence, 4),
        "semantic_confidence_type": "llm_self_estimated",
        "intent": str(parsed.get("intent", "other")).strip() or "other",
        "valence": max(-1.0, min(1.0, valence)),
        "arousal": max(0.0, min(1.0, arousal)),
        "evidence": [str(item) for item in evidence[:5]],
        "reason": str(parsed.get("reason", "")).strip(),
        "source": "local_qwen_semantic_judgement",
        "raw_content": content,
    }


def fallback_semantic_emotion(reason: str) -> Dict[str, Any]:
    return {
        "emotion": None,
        "semantic_confidence": None,
        "semantic_confidence_type": None,
        "intent": None,
        "valence": None,
        "arousal": None,
        "evidence": [],
        "reason": None,
        "source": "disabled_or_failed",
        "error": reason,
        "raw_content": None,
    }


def fuse_emotion_with_semantics(
    audio_emotion: str,
    text_emotion: str,
    text_conf: float,
    semantic_result: Dict[str, Any],
    text: str,
) -> tuple[str, str]:
    semantic_emotion = semantic_result.get("emotion")
    semantic_conf = semantic_result.get("semantic_confidence")
    try:
        semantic_conf_f = float(semantic_conf) if semantic_conf is not None else 0.0
    except (TypeError, ValueError):
        semantic_conf_f = 0.0

    if semantic_emotion in {"sad", "angry"} and semantic_conf_f >= 0.65:
        if audio_emotion == semantic_emotion:
            return semantic_emotion, "rule:audio_semantic_agree"
        if audio_emotion == "neutral":
            return semantic_emotion, "rule:audio_neutral_semantic_negative_override"
        if text_emotion == "neutral":
            return semantic_emotion, "rule:text_model_neutral_semantic_negative_override"
        if text_emotion == semantic_emotion:
            return semantic_emotion, "rule:text_semantic_agree"

    if audio_emotion == "angry" and semantic_emotion in {None, "neutral"}:
        return "angry", "rule:audio_high_arousal_overrides_semantic_neutral"

    if semantic_emotion in VALID_EMOTIONS and semantic_conf_f >= 0.75 and text_emotion != semantic_emotion:
        return str(semantic_emotion), "rule:semantic_high_confidence"

    return fuse_emotion(audio_emotion, text_emotion, text_conf, text)


def smooth_tts_params_for_web(params: Dict[str, Any]) -> Dict[str, Any]:
    # 保持与离线直跑一致，不在 Web 端改写参数。
    return dict(params)


def pick_fallback_voice_id(profile_path: Path) -> str:
    if not profile_path.exists():
        return ""
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    voices = data.get("voices", {}) if isinstance(data, dict) else {}
    if not isinstance(voices, dict) or not voices:
        return ""
    # 优先用键名（当前推荐格式），否则回退读取内部 voice_id 字段
    first_key = next(iter(voices.keys()), "")
    if first_key:
        return str(first_key).strip()
    for v in voices.values():
        if isinstance(v, dict):
            vid = str(v.get("voice_id", "")).strip()
            if vid:
                return vid
    return ""


def synthesize_tts(
    pipeline_json_path: Path, use_voice_id: str, use_voice_cache: bool
) -> tuple[Optional[str], Optional[str]]:
    out_wav = OUTPUT_AUDIO / f"tts_{uuid.uuid4().hex}.wav"
    tts_script = PROJECT_ROOT / "scripts" / "tts_qwen3_from_pipeline.py"
    cmd = [
        sys.executable,
        str(tts_script),
        "--pipeline-json",
        str(pipeline_json_path),
        "--output",
        str(out_wav),
        "--device",
        ARGS.device,
        "--disable-postprocess",
    ]
    # 高保真优先：若提供了固定参考音频/文本，则直接走 ref_audio 模式，避免旧 cache 和 m4a 参考带来的伪影。
    ref_audio = DEFAULT_WEB_REF_AUDIO
    ref_text = DEFAULT_WEB_REF_TEXT
    if ref_audio and ref_text:
        cmd.extend(["--ref-audio", ref_audio, "--ref-text", ref_text])
    else:
        voice_id = use_voice_id.strip() or DEFAULT_VOICE_ID or pick_fallback_voice_id(DEFAULT_VOICE_PROFILE)
        if voice_id:
            cmd.extend(["--use-voice-id", voice_id])
        if use_voice_cache:
            cmd.append("--use-voice-cache")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        err = stderr or stdout or f"tts 脚本失败，exit={proc.returncode}"
        return None, err
    return f"/assets/audio/{out_wav.name}", None


app = FastAPI(title="Digital Human Multimodal Demo")
ensure_dirs()
app.mount("/assets/audio", StaticFiles(directory=str(OUTPUT_AUDIO)), name="audio")


@app.get("/", response_class=HTMLResponse)
def chat_index() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>数字人情感对话</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f4f5f7; color: #1f2933; }
    .app { height: 100vh; display: grid; grid-template-columns: 280px minmax(0, 1fr); }
    .sidebar { background: #20242b; color: #f7f8fa; display: flex; flex-direction: column; min-height: 0; }
    .brand { padding: 18px 16px 12px; border-bottom: 1px solid rgba(255,255,255,.08); }
    .brand h1 { font-size: 18px; line-height: 1.25; margin: 0 0 4px; font-weight: 700; }
    .brand .sub { color: #aeb6c2; font-size: 12px; }
    .new-chat { margin: 12px 12px 8px; height: 38px; border: 1px solid rgba(255,255,255,.16); background: #2f3641; color: #fff; border-radius: 8px; cursor: pointer; }
    .sessions { overflow-y: auto; padding: 4px 8px 12px; }
    .session { width: 100%; border: 0; color: #dfe4ea; background: transparent; padding: 8px 8px; border-radius: 8px; cursor: pointer; margin: 2px 0; display: grid; grid-template-columns: minmax(0, 1fr) 28px; gap: 4px; align-items: center; position: relative; }
    .session:hover { background: rgba(255,255,255,.08); }
    .session.active { background: #3d4653; color: #fff; }
    .session-main { min-width: 0; text-align: left; }
    .session-title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 14px; }
    .session-time { color: #9aa4b2; font-size: 11px; margin-top: 3px; }
    .session-more { width: 28px; height: 28px; border: 0; background: transparent; color: #c7ced8; border-radius: 6px; cursor: pointer; font-size: 18px; line-height: 24px; }
    .session-more:hover { background: rgba(255,255,255,.12); color: #fff; }
    .menu { position: fixed; z-index: 20; min-width: 132px; background: #fff; color: #1f2933; border: 1px solid #d0d5dd; border-radius: 8px; box-shadow: 0 10px 24px rgba(16,24,40,.18); padding: 6px; display: none; }
    .menu.open { display: block; }
    .menu button { width: 100%; border: 0; background: transparent; color: #1f2933; text-align: left; padding: 8px 9px; border-radius: 6px; cursor: pointer; }
    .menu button:hover { background: #f2f4f7; }
    .menu .danger { color: #b42318; }
    .main { min-width: 0; height: 100vh; display: grid; grid-template-rows: auto minmax(0, 1fr) auto; background: #fff; }
    .topbar { min-height: 62px; border-bottom: 1px solid #e6e8ec; display: flex; align-items: center; justify-content: space-between; padding: 10px 18px; gap: 12px; }
    .title { font-weight: 700; }
    .meta { color: #667085; font-size: 12px; margin-top: 3px; }
    .top-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
    .control { color: #344054; font-size: 13px; display: inline-flex; gap: 5px; align-items: center; white-space: nowrap; }
    .voice-id { width: 120px; height: 30px; border: 1px solid #d0d5dd; border-radius: 7px; padding: 0 8px; }
    .ghost { height: 32px; border: 1px solid #d0d5dd; background: #fff; color: #344054; border-radius: 7px; cursor: pointer; padding: 0 10px; }
    .chat-shell { min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 320px; }
    .messages { min-height: 0; overflow-y: auto; padding: 22px 28px; background: #fbfcfd; }
    .empty { max-width: 620px; margin: 15vh auto 0; text-align: center; color: #667085; }
    .empty h2 { color: #1f2933; margin: 0 0 10px; font-size: 24px; }
    .msg { display: flex; margin: 14px 0; }
    .msg.user { justify-content: flex-end; }
    .bubble { max-width: min(720px, 76%); padding: 12px 14px; border-radius: 14px; line-height: 1.55; font-size: 15px; box-shadow: 0 1px 2px rgba(16,24,40,.06); white-space: pre-wrap; overflow-wrap: anywhere; }
    .user .bubble { background: #2563eb; color: #fff; border-bottom-right-radius: 5px; }
    .assistant .bubble { background: #fff; color: #1f2933; border: 1px solid #eaecf0; border-bottom-left-radius: 5px; }
    .bubble-meta { margin-top: 8px; font-size: 12px; opacity: .72; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .bubble audio { width: 100%; margin-top: 8px; }
    .inspector { border-left: 1px solid #e6e8ec; background: #fff; min-height: 0; overflow-y: auto; padding: 14px; }
    .panel { border: 1px solid #eaecf0; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }
    .panel summary { cursor: pointer; padding: 10px 12px; font-weight: 700; background: #f8fafc; }
    .panel pre { margin: 0; padding: 10px 12px; background: #fff; color: #344054; overflow: auto; max-height: 280px; font-size: 12px; }
    .composer { border-top: 1px solid #e6e8ec; background: #fff; padding: 12px 18px; }
    .composer-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px; align-items: end; }
    textarea { width: 100%; min-height: 52px; max-height: 140px; resize: vertical; border: 1px solid #d0d5dd; border-radius: 10px; padding: 12px; font-size: 15px; outline: none; }
    textarea:focus { border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,.12); }
    .send { height: 52px; min-width: 86px; border: 0; background: #2563eb; color: #fff; border-radius: 10px; cursor: pointer; font-weight: 700; }
    .send:disabled { background: #98a2b3; cursor: wait; }
    .composer-tools { display: flex; gap: 10px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; color: #475467; font-size: 13px; }
    .file { max-width: 260px; }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { display: none; }
      .chat-shell { grid-template-columns: 1fr; }
      .inspector { display: none; }
      .bubble { max-width: 88%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <h1>数字人情感对话</h1>
        <div class="sub">ASR / SER / LLM / TTS</div>
      </div>
      <button class="new-chat" onclick="newChat()">新建对话</button>
      <div id="sessions" class="sessions"></div>
    </aside>
    <div id="sessionMenu" class="menu"></div>

    <main class="main">
      <header class="topbar">
        <div>
          <div class="title" id="chatTitle">新对话</div>
          <div class="meta">session: <span id="sessionId"></span></div>
        </div>
        <div class="top-actions">
          <label class="control">voice_id <input id="voiceId" class="voice-id" type="text" placeholder="zjh"/></label>
          <label class="control"><input id="useCache" type="checkbox" checked/> voice cache</label>
          <label class="control"><input id="genTts" type="checkbox" checked/> 语音输出</label>
          <button class="ghost" onclick="resetContext()">清空上下文</button>
        </div>
      </header>

      <section class="chat-shell">
        <div id="messages" class="messages"></div>
        <aside class="inspector">
          <details class="panel" open>
            <summary>本轮情感识别</summary>
            <pre id="task1">暂无</pre>
          </details>
          <details class="panel">
            <summary>共情回复 JSON</summary>
            <pre id="task2">暂无</pre>
          </details>
          <details class="panel">
            <summary>TTS 参数</summary>
            <pre id="task3">暂无</pre>
          </details>
          <details class="panel">
            <summary>完整响应</summary>
            <pre id="full">暂无</pre>
          </details>
          <details class="panel">
            <summary>TTS 错误</summary>
            <pre id="ttsErr">无</pre>
          </details>
        </aside>
      </section>

      <footer class="composer">
        <div class="composer-tools">
          <label>语音输入 <input id="audio" class="file" type="file" accept=".wav,.mp3,.m4a,.flac,.ogg,.aac"/></label>
          <span id="inputHint">可只发文字，也可上传音频；文本会覆盖 ASR 结果。</span>
        </div>
        <div class="composer-row">
          <textarea id="textInput" placeholder="输入你想说的话，按 Ctrl+Enter 发送"></textarea>
          <button id="sendBtn" class="send" onclick="sendMessage()">发送</button>
        </div>
      </footer>
    </main>
  </div>

  <script>
    const ACTIVE_KEY = "digi_human_chat_active_session";
    const SESSIONS_KEY = "digi_human_chat_sessions";
    let activeSession = "";
    let busy = false;
    let sessionLoadSeq = 0;

    function uid() {
      if (crypto && crypto.randomUUID) return crypto.randomUUID().replaceAll("-", "");
      return String(Date.now()) + Math.random().toString(16).slice(2);
    }
    function loadSessions() {
      try { return JSON.parse(localStorage.getItem(SESSIONS_KEY) || "[]"); }
      catch { return []; }
    }
    function saveSessions(items) {
      localStorage.setItem(SESSIONS_KEY, JSON.stringify(items));
    }
    function sessionsFromServerIndex(data) {
      if (!data || !data.sessions) return null;
      return data.sessions.map(s => ({
        id: s.session_id,
        title: s.title || "新对话",
        updatedAt: s.updated_at ? Date.parse(s.updated_at) || Date.now() : Date.now(),
        pinned: !!s.pinned
      }));
    }
    function messageKey(sessionId) {
      return "digi_human_chat_messages_" + sessionId;
    }
    function loadMessages(sessionId) {
      try { return JSON.parse(localStorage.getItem(messageKey(sessionId)) || "[]"); }
      catch { return []; }
    }
    function saveMessages(sessionId, messages) {
      localStorage.setItem(messageKey(sessionId), JSON.stringify(messages.slice(-80)));
    }
    async function refreshSessionsFromServer() {
      try {
        const resp = await fetch("/api/sessions");
        const data = await resp.json();
        const sessions = resp.ok ? sessionsFromServerIndex(data) : null;
        if (!sessions) return loadSessions();
        saveSessions(sessions);
        renderSessions();
        return sessions;
      } catch {
        return loadSessions();
      }
    }
    async function ensureSession() {
      let sessions = await refreshSessionsFromServer();
      let sid = localStorage.getItem(ACTIVE_KEY);
      if (!sid || !sessions.some(s => s.id === sid)) {
        sid = uid();
        sessions.unshift({ id: sid, title: "新对话", updatedAt: Date.now() });
        saveSessions(sessions);
        localStorage.setItem(ACTIVE_KEY, sid);
      }
      activeSession = sid;
      renderSessions();
      await loadServerSession(sid);
    }
    function touchSession(title) {
      const sessions = loadSessions();
      const idx = sessions.findIndex(s => s.id === activeSession);
      if (idx >= 0) {
        if (title && sessions[idx].title === "新对话") sessions[idx].title = title.slice(0, 18);
        sessions[idx].updatedAt = Date.now();
        const [item] = sessions.splice(idx, 1);
        sessions.unshift(item);
        saveSessions(sessions);
      }
      renderSessions();
    }
    function renderSessions() {
      const wrap = document.getElementById("sessions");
      const sessions = loadSessions();
      wrap.innerHTML = "";
      sessions.forEach(s => {
        const btn = document.createElement("div");
        btn.className = "session" + (s.id === activeSession ? " active" : "");
        btn.onclick = () => switchSession(s.id);
        const time = s.updatedAt ? new Date(s.updatedAt).toLocaleString() : "";
        btn.innerHTML = `<div class="session-main"><div class="session-title">${s.pinned ? "📌 " : ""}${escapeHtml(s.title || "新对话")}</div><div class="session-time">${time}</div></div><button class="session-more" title="更多" onclick="openSessionMenu(event, '${s.id}')">⋯</button>`;
        wrap.appendChild(btn);
      });
    }
    async function switchSession(sessionId) {
      const loadSeq = ++sessionLoadSeq;
      activeSession = sessionId;
      localStorage.setItem(ACTIVE_KEY, sessionId);
      clearInspector();
      renderSessions();
      await loadServerSession(sessionId, loadSeq);
    }
    async function newChat() {
      const emptySession = loadSessions().find(s => {
        const title = String(s.title || "");
        return loadMessages(s.id).length === 0 && (title === "新对话" || title.indexOf("新") >= 0);
      });
      if (emptySession) {
        await switchSession(emptySession.id);
        return;
      }
      const sid = uid();
      const sessions = loadSessions();
      sessions.unshift({ id: sid, title: "新对话", updatedAt: Date.now(), pinned: false });
      saveSessions(sessions);
      const loadSeq = ++sessionLoadSeq;
      activeSession = sid;
      localStorage.setItem(ACTIVE_KEY, sid);
      saveMessages(sid, []);
      clearInspector();
      renderSessions();
      renderMessages();
      await loadServerSession(sid, loadSeq);
    }
    function closeSessionMenu() {
      const menu = document.getElementById("sessionMenu");
      menu.className = "menu";
      menu.innerHTML = "";
    }
    function openSessionMenu(event, sessionId) {
      event.stopPropagation();
      event.preventDefault();
      const sessions = loadSessions();
      const item = sessions.find(s => s.id === sessionId) || {};
      const menu = document.getElementById("sessionMenu");
      menu.innerHTML = `
        <button onclick="togglePinSession('${sessionId}', ${item.pinned ? "false" : "true"})">${item.pinned ? "取消置顶" : "置顶"}</button>
        <button onclick="renameSession('${sessionId}')">重命名</button>
        <button class="danger" onclick="deleteSession('${sessionId}')">删除</button>
      `;
      const rect = event.currentTarget.getBoundingClientRect();
      menu.style.left = Math.min(rect.right + 4, window.innerWidth - 150) + "px";
      menu.style.top = Math.min(rect.top, window.innerHeight - 140) + "px";
      menu.className = "menu open";
    }
    async function togglePinSession(sessionId, pinned) {
      closeSessionMenu();
      await fetch("/api/sessions/" + encodeURIComponent(sessionId) + "/pin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pinned })
      });
      await refreshSessionsFromServer();
    }
    async function renameSession(sessionId) {
      closeSessionMenu();
      const sessions = loadSessions();
      const item = sessions.find(s => s.id === sessionId) || {};
      const title = prompt("重命名对话", item.title || "新对话");
      if (!title || !title.trim()) return;
      await fetch("/api/sessions/" + encodeURIComponent(sessionId) + "/rename", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title.trim() })
      });
      await refreshSessionsFromServer();
      renderMessages();
    }
    async function deleteSession(sessionId) {
      closeSessionMenu();
      if (!confirm("确定删除这个对话？此操作会删除对应的本地 JSON。")) return;
      const resp = await fetch("/api/sessions/" + encodeURIComponent(sessionId), { method: "DELETE" });
      const data = await resp.json();
      if (!resp.ok) {
        alert(data.detail || "删除失败");
        return;
      }
      localStorage.removeItem(messageKey(sessionId));
      let sessions = sessionsFromServerIndex(data.index) || [];
      saveSessions(sessions);
      renderSessions();
      if (activeSession === sessionId) {
        if (sessions.length) {
          await switchSession(sessions[0].id);
        } else {
          await newChat();
        }
      }
    }
    function escapeHtml(text) {
      return String(text || "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }
    function addMessage(message) {
      const messages = loadMessages(activeSession);
      messages.push(message);
      saveMessages(activeSession, messages);
      renderMessages();
    }
    function updateLastAssistant(patch) {
      const messages = loadMessages(activeSession);
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === "assistant") {
          messages[i] = { ...messages[i], ...patch };
          break;
        }
      }
      saveMessages(activeSession, messages);
      renderMessages();
    }
    function renderMessages() {
      document.getElementById("sessionId").textContent = activeSession;
      const sessions = loadSessions();
      const current = sessions.find(s => s.id === activeSession);
      document.getElementById("chatTitle").textContent = current ? current.title : "新对话";
      const box = document.getElementById("messages");
      const messages = loadMessages(activeSession);
      if (!messages.length) {
        clearInspector();
        box.innerHTML = `<div class="empty"><h2>开始一轮情感对话</h2><div>输入文字或上传语音，系统会结合情绪识别和上下文生成回复。</div></div>`;
        return;
      }
      box.innerHTML = "";
      messages.forEach(m => {
        const row = document.createElement("div");
        row.className = "msg " + m.role;
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.innerHTML = escapeHtml(m.text || "");
        const meta = [];
        if (m.emotion) meta.push("情绪: " + m.emotion);
        if (m.audioName) meta.push("语音: " + m.audioName);
        if (m.pending) meta.push("处理中");
        if (meta.length || m.audioUrl) {
          const foot = document.createElement("div");
          foot.className = "bubble-meta";
          foot.textContent = meta.join(" · ");
          bubble.appendChild(foot);
        }
        if (m.audioUrl) {
          const audio = document.createElement("audio");
          audio.controls = true;
          audio.src = m.audioUrl;
          bubble.appendChild(audio);
        }
        row.appendChild(bubble);
        box.appendChild(row);
      });
      box.scrollTop = box.scrollHeight;
    }
    function setJson(id, obj) {
      document.getElementById(id).textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
    }
    function clearInspector() {
      setJson("task1", "暂无");
      setJson("task2", "暂无");
      setJson("task3", "暂无");
      setJson("full", "暂无");
      setJson("ttsErr", "无");
    }
    async function loadServerSession(sessionId, loadSeq = ++sessionLoadSeq) {
      clearInspector();
      try {
        const resp = await fetch("/api/sessions/" + encodeURIComponent(sessionId));
        const data = await resp.json();
        if (loadSeq !== sessionLoadSeq || sessionId !== activeSession) return;
        if (!resp.ok || !data.ok) throw new Error(data.detail || "load session failed");
        if (data.messages) saveMessages(sessionId, data.messages);
        if (data.session) {
          const sessions = loadSessions();
          const idx = sessions.findIndex(s => s.id === sessionId);
          if (idx >= 0) {
            sessions[idx].title = data.session.title || sessions[idx].title;
            sessions[idx].updatedAt = data.session.updated_at ? Date.parse(data.session.updated_at) || sessions[idx].updatedAt : sessions[idx].updatedAt;
            saveSessions(sessions);
            renderSessions();
          }
        }
        const messages = data.messages || [];
        const lastAssistant = [...messages].reverse().find(m => m.role === "assistant" && m.fullResponse);
        if (lastAssistant && lastAssistant.fullResponse) {
          const full = lastAssistant.fullResponse;
          setJson("task1", full.task1_emotion_arbitration || {});
          setJson("task2", full.task2_empathic_reply || {});
          setJson("task3", full.task3_tts_control || {});
          setJson("full", full);
          setJson("ttsErr", full.tts_error || "无");
        }
      } catch {
        if (loadSeq !== sessionLoadSeq || sessionId !== activeSession) return;
      }
      if (loadSeq !== sessionLoadSeq || sessionId !== activeSession) return;
      renderMessages();
    }
    async function resetContext() {
      const fd = new FormData();
      fd.append("session_id", activeSession);
      const resp = await fetch("/api/reset_session", { method: "POST", body: fd });
      const data = await resp.json();
      saveMessages(activeSession, []);
      setJson("full", data);
      clearInspector();
      setJson("full", data);
      renderMessages();
    }
    async function sendMessage() {
      if (busy) return;
      const input = document.getElementById("textInput");
      const fileInput = document.getElementById("audio");
      const text = input.value.trim();
      const file = fileInput.files[0];
      if (!text && !file) return;
      busy = true;
      document.getElementById("sendBtn").disabled = true;
      const userText = text || "[语音输入]";
      addMessage({ role: "user", text: userText, audioName: file ? file.name : "", createdAt: Date.now() });
      addMessage({ role: "assistant", text: "正在分析情绪并生成回复...", pending: true, createdAt: Date.now() });
      touchSession(text || (file ? file.name : "语音对话"));
      const fd = new FormData();
      if (file) fd.append("audio", file);
      fd.append("text_override", text);
      fd.append("use_voice_id", document.getElementById("voiceId").value || "");
      fd.append("use_voice_cache", String(document.getElementById("useCache").checked));
      fd.append("generate_tts", String(document.getElementById("genTts").checked));
      fd.append("session_id", activeSession);
      try {
        const resp = await fetch("/api/analyze", { method: "POST", body: fd });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "请求失败");
        const reply = data.task2_empathic_reply || {};
        const task1 = data.task1_emotion_arbitration || {};
        updateLastAssistant({
          text: reply.reply_text || "我在，愿意继续听你说。",
          emotion: reply.emotion || task1.final_emotion || "neutral",
          audioUrl: data.tts_audio_url || "",
          fullResponse: data,
          pending: false,
        });
        setJson("task1", data.task1_emotion_arbitration || {});
        setJson("task2", data.task2_empathic_reply || {});
        setJson("task3", data.task3_tts_control || {});
        setJson("full", data);
        setJson("ttsErr", data.tts_error || "无");
        input.value = "";
        fileInput.value = "";
      } catch (err) {
        updateLastAssistant({ text: "请求失败：" + err.message, pending: false });
        setJson("ttsErr", err.message || String(err));
      } finally {
        busy = false;
        document.getElementById("sendBtn").disabled = false;
      }
    }
    document.addEventListener("click", closeSessionMenu);
    window.addEventListener("resize", closeSessionMenu);
    document.getElementById("textInput").addEventListener("keydown", ev => {
      if (ev.ctrlKey && ev.key === "Enter") sendMessage();
    });
    ensureSession();
  </script>
</body>
</html>"""


@app.get("/form", response_class=HTMLResponse)
def index_with_context() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>数字人多模态情感交互 Demo</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; max-width: 980px; }
    h1 { margin-bottom: 8px; }
    .row { margin: 10px 0; }
    textarea { width: 100%; height: 80px; }
    button { padding: 8px 14px; cursor: pointer; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 10px; }
    .muted { color: #666; font-size: 13px; }
    pre { background: #f7f7f7; padding: 10px; border-radius: 6px; overflow: auto; }
  </style>
</head>
<body>
  <h1>数字人多模态情感交互 Web Demo</h1>
  <div class="muted">上传音频后，自动执行 ASR + SER + 文本情感 + 融合仲裁 + LLM 共情回复；可选生成 TTS。</div>
  <div class="muted">已支持上下文互动：同一浏览器会话会保留最近若干轮对话，TTS 只合成最新一轮回复。</div>
  <div class="muted">若设置 WEB_TTS_REF_AUDIO + WEB_TTS_REF_TEXT，Web 将优先使用固定参考音频生成，音质更稳定。</div>

  <div class="row">
    <label>音频文件（建议 16k wav）：</label><br/>
    <input id="audio" type="file" accept=".wav,.mp3,.m4a,.flac,.ogg,.aac"/>
  </div>
  <div class="row">
    <label>文本覆盖（可选，不填则使用 ASR）：</label>
    <textarea id="textOverride" placeholder="可输入文本覆盖 ASR 结果"></textarea>
  </div>
  <div class="row">
    <label>voice_id（建议填写，生成 TTS 时复用）：</label><br/>
    <input id="voiceId" type="text" placeholder="例如 zjh" style="width: 280px;"/>
    <label style="margin-left: 12px;"><input id="useCache" type="checkbox" checked/> use voice cache</label>
    <label style="margin-left: 12px;"><input id="genTts" type="checkbox" checked/> generate tts</label>
  </div>
  <div class="row">
    <button onclick="runAnalyze()">开始分析</button>
    <button onclick="resetContext()" style="margin-left: 8px;">清空上下文</button>
    <span class="muted" style="margin-left: 12px;">session: <span id="sessionId"></span></span>
  </div>

  <div class="grid">
    <div class="card"><b>任务1：多模态情感识别</b><pre id="task1">等待运行...</pre></div>
    <div class="card"><b>任务2：共情回复</b><pre id="task2">等待运行...</pre></div>
  </div>
  <div class="card" style="margin-top:12px;">
    <b>任务3：TTS 参数</b>
    <pre id="task3">等待运行...</pre>
    <audio id="player" controls style="width:100%; margin-top: 8px;"></audio>
  </div>
  <div class="card" style="margin-top:12px;">
    <b>上下文记录</b>
    <pre id="history">暂无</pre>
  </div>
  <div class="card" style="margin-top:12px;">
    <b>完整 JSON</b>
    <pre id="full">等待运行...</pre>
  </div>
  <div class="card" style="margin-top:12px;">
    <b>TTS 错误信息</b>
    <pre id="ttsErr">无</pre>
  </div>

  <script>
    const SESSION_KEY = "digi_human_web_demo_session_id";
    function newSessionId() {
      if (crypto && crypto.randomUUID) return crypto.randomUUID().replaceAll("-", "");
      return String(Date.now()) + Math.random().toString(16).slice(2);
    }
    function getSessionId() {
      let sid = localStorage.getItem(SESSION_KEY);
      if (!sid) {
        sid = newSessionId();
        localStorage.setItem(SESSION_KEY, sid);
      }
      document.getElementById("sessionId").textContent = sid;
      return sid;
    }
    function renderHistory(items) {
      if (!items || !items.length) {
        document.getElementById("history").textContent = "暂无";
        return;
      }
      const lines = [];
      items.forEach((item, idx) => {
        lines.push(`${idx + 1}. 用户(${item.final_emotion || "neutral"}): ${item.user || ""}`);
        lines.push(`   AI(${item.reply_emotion || item.final_emotion || "neutral"}): ${item.assistant || ""}`);
      });
      document.getElementById("history").textContent = lines.join("\\n");
    }
    async function resetContext() {
      const fd = new FormData();
      fd.append("session_id", getSessionId());
      const resp = await fetch("/api/reset_session", { method: "POST", body: fd });
      const data = await resp.json();
      renderHistory([]);
      document.getElementById("full").textContent = JSON.stringify(data, null, 2);
    }
    getSessionId();

    async function runAnalyze() {
      const fileInput = document.getElementById("audio");
      const textOverride = document.getElementById("textOverride").value || "";
      const voiceId = document.getElementById("voiceId").value || "";
      const useCache = document.getElementById("useCache").checked;
      const genTts = document.getElementById("genTts").checked;
      const fd = new FormData();
      if (fileInput.files[0]) fd.append("audio", fileInput.files[0]);
      fd.append("text_override", textOverride);
      fd.append("use_voice_id", voiceId);
      fd.append("use_voice_cache", String(useCache));
      fd.append("generate_tts", String(genTts));
      fd.append("session_id", getSessionId());

      document.getElementById("full").textContent = "处理中...";
      const resp = await fetch("/api/analyze", { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok) {
        document.getElementById("full").textContent = JSON.stringify(data, null, 2);
        document.getElementById("ttsErr").textContent = data.detail || "请求失败";
        return;
      }
      document.getElementById("task1").textContent = JSON.stringify(data.task1_emotion_arbitration, null, 2);
      document.getElementById("task2").textContent = JSON.stringify(data.task2_empathic_reply, null, 2);
      document.getElementById("task3").textContent = JSON.stringify(data.task3_tts_control, null, 2);
      document.getElementById("full").textContent = JSON.stringify(data, null, 2);
      document.getElementById("ttsErr").textContent = data.tts_error || "无";
      renderHistory(data.context_history || []);
      document.getElementById("textOverride").value = "";
      if (data.tts_audio_url) {
        const player = document.getElementById("player");
        player.src = data.tts_audio_url;
        player.load();
      }
    }
  </script>
</body>
</html>"""


@app.get("/legacy", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>数字人多模态情感交互 Demo</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; max-width: 980px; }
    h1 { margin-bottom: 8px; }
    .row { margin: 10px 0; }
    textarea { width: 100%; height: 80px; }
    button { padding: 8px 14px; cursor: pointer; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 10px; }
    .muted { color: #666; font-size: 13px; }
    pre { background: #f7f7f7; padding: 10px; border-radius: 6px; overflow: auto; }
  </style>
</head>
<body>
  <h1>数字人多模态情感交互 Web Demo</h1>
  <div class="muted">上传音频后，自动执行 ASR + SER + 文本情感 + 融合仲裁 + LLM 共情回复；可选生成 TTS。</div>
  <div class="muted">若设置 WEB_TTS_REF_AUDIO + WEB_TTS_REF_TEXT，Web 将优先使用固定参考音频生成（音质更稳定）。</div>

  <div class="row">
    <label>音频文件（建议 16k wav）：</label><br/>
    <input id="audio" type="file" accept=".wav,.mp3,.m4a,.flac,.ogg,.aac"/>
  </div>
  <div class="row">
    <label>文本覆盖（可选，不填则使用ASR）：</label>
    <textarea id="textOverride" placeholder="可输入文本覆盖 ASR 结果"></textarea>
  </div>
  <div class="row">
    <label>voice_id（建议填写，生成TTS时复用）：</label><br/>
    <input id="voiceId" type="text" placeholder="例如 zjh" style="width: 280px;"/>
    <label style="margin-left: 12px;"><input id="useCache" type="checkbox" checked/> use voice cache</label>
    <label style="margin-left: 12px;"><input id="genTts" type="checkbox" checked/> generate tts</label>
  </div>
  <div class="row"><button onclick="runAnalyze()">开始分析</button></div>

  <div class="grid">
    <div class="card"><b>任务1：多模态情感识别</b><pre id="task1">等待运行...</pre></div>
    <div class="card"><b>任务2：共情回复</b><pre id="task2">等待运行...</pre></div>
  </div>
  <div class="card" style="margin-top:12px;">
    <b>任务3：TTS参数</b>
    <pre id="task3">等待运行...</pre>
    <audio id="player" controls style="width:100%; margin-top: 8px;"></audio>
  </div>
  <div class="card" style="margin-top:12px;">
    <b>完整JSON</b>
    <pre id="full">等待运行...</pre>
  </div>
  <div class="card" style="margin-top:12px;">
    <b>TTS错误信息</b>
    <pre id="ttsErr">无</pre>
  </div>

  <script>
    async function runAnalyze() {
      const fileInput = document.getElementById("audio");
      const textOverride = document.getElementById("textOverride").value || "";
      const voiceId = document.getElementById("voiceId").value || "";
      const useCache = document.getElementById("useCache").checked;
      const genTts = document.getElementById("genTts").checked;
      const fd = new FormData();
      if (fileInput.files[0]) fd.append("audio", fileInput.files[0]);
      fd.append("text_override", textOverride);
      fd.append("use_voice_id", voiceId);
      fd.append("use_voice_cache", String(useCache));
      fd.append("generate_tts", String(genTts));

      document.getElementById("full").textContent = "处理中...";
      const resp = await fetch("/api/analyze", { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok) {
        document.getElementById("full").textContent = JSON.stringify(data, null, 2);
        document.getElementById("ttsErr").textContent = data.detail || "请求失败";
        return;
      }
      document.getElementById("task1").textContent = JSON.stringify(data.task1_emotion_arbitration, null, 2);
      document.getElementById("task2").textContent = JSON.stringify(data.task2_empathic_reply, null, 2);
      document.getElementById("task3").textContent = JSON.stringify(data.task3_tts_control, null, 2);
      document.getElementById("full").textContent = JSON.stringify(data, null, 2);
      document.getElementById("ttsErr").textContent = data.tts_error || "无";
      if (data.tts_audio_url) {
        const player = document.getElementById("player");
        player.src = data.tts_audio_url;
        player.load();
      }
    }
  </script>
</body>
</html>"""


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "sensevoice_model": ARGS.sv_model,
        "device": ARGS.device,
        "llm_url": ARGS.llm_url,
        "llm_model": ARGS.llm_model,
        "llm_semantic_emotion": ARGS.enable_llm_semantic_emotion,
        "semantic_max_tokens": ARGS.semantic_max_tokens,
        "text_emotion_model": str(text_emotion_model.MODEL_PATH),
        "text_emotion_device": text_emotion_model.DEFAULT_DEVICE,
        "text_emotion_preload": not ARGS.no_preload_text_emotion,
        "text_emotion_load_error": get_load_error(),
        "session_dir": str(SESSION_DIR),
    }


@app.post("/api/reset_session")
async def reset_session(session_id: str = Form(default="")) -> Dict[str, Any]:
    sid = normalize_session_id(session_id)
    with SESSION_LOCK:
        SESSION_HISTORY.pop(sid, None)
    doc = load_session_doc(sid)
    doc["turns"] = []
    doc["title"] = "新对话"
    save_session_doc(doc)
    session_index()
    return {"ok": True, "session_id": sid, "context_history": []}


@app.get("/api/sessions")
def list_sessions() -> Dict[str, Any]:
    return session_index()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    sid = normalize_session_id(session_id)
    doc = load_session_doc(sid)
    history = rebuild_context_from_doc(doc)
    with SESSION_LOCK:
        SESSION_HISTORY[sid] = [dict(item) for item in history]
    return {
        "ok": True,
        "session": doc,
        "messages": session_messages_from_doc(doc),
        "context_history": history,
    }


@app.post("/api/sessions/{session_id}/rename")
def rename_session(session_id: str, body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    sid = normalize_session_id(session_id)
    title = " ".join(str(body.get("title", "")).strip().split())[:40]
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    doc = load_session_doc(sid)
    doc["title"] = title
    save_session_doc(doc)
    return {"ok": True, "session": doc, "index": session_index()}


@app.post("/api/sessions/{session_id}/pin")
def pin_session(session_id: str, body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    sid = normalize_session_id(session_id)
    doc = load_session_doc(sid)
    doc["pinned"] = bool(body.get("pinned", True))
    save_session_doc(doc)
    return {"ok": True, "session": doc, "index": session_index()}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    sid = normalize_session_id(session_id)
    with SESSION_LOCK:
        SESSION_HISTORY.pop(sid, None)
    path = session_json_path(sid)
    doc = read_json_file(path, {})
    artifacts = delete_session_pipeline_jsons(doc) if isinstance(doc, dict) else {"deleted": [], "skipped": []}
    deleted_json = False
    if path.exists():
        path.unlink()
        deleted_json = True
    index = session_index()
    return {
        "ok": True,
        "session_id": sid,
        "deleted_json": deleted_json,
        "deleted_json_path": str(path),
        "deleted_pipeline_jsons": artifacts["deleted"],
        "skipped_pipeline_jsons": artifacts["skipped"],
        "index": index,
    }


@app.post("/api/analyze")
async def analyze(
    audio: Optional[UploadFile] = File(default=None),
    text_override: str = Form(default=""),
    generate_tts: str = Form(default="true"),
    use_voice_id: str = Form(default=""),
    use_voice_cache: str = Form(default="true"),
    session_id: str = Form(default=""),
) -> Dict[str, Any]:
    ensure_dirs()
    request_id = uuid.uuid4().hex
    sid = normalize_session_id(session_id)
    asr_text = ""
    audio_emo_raw = "EMO_UNK"
    audio_emotion = "neutral"
    local_audio: Optional[Path] = None
    audio_name = ""
    audio_result: Dict[str, Any] = {
        "source": "sensevoice",
        "status": "skipped",
        "reason": "no_audio_uploaded",
        "input_audio_name": "",
        "input_audio_path": None,
        "asr_text": "",
        "sensevoice_text_raw": "",
        "audio_emotion_raw": audio_emo_raw,
        "audio_emotion": audio_emotion,
        "sensevoice_first_result": {},
        "sensevoice_raw_result": [],
    }

    if audio is not None:
        suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
        audio_name = audio.filename or ""
        local_audio = OUTPUT_AUDIO / f"input_{request_id}{suffix}"
        local_audio.write_bytes(await audio.read())
        try:
            result = run_asr_ser(local_audio)
            asr_text = result["asr_text"]
            audio_emo_raw = result["audio_emotion_raw"]
            audio_emotion = result["audio_emotion"]
            audio_result = {
                "source": "sensevoice",
                "status": "ok",
                "input_audio_name": audio_name,
                "input_audio_path": str(local_audio),
                "asr_text": asr_text,
                "sensevoice_text_raw": result.get("sensevoice_text_raw", ""),
                "audio_emotion_raw": audio_emo_raw,
                "audio_emotion": audio_emotion,
                "sensevoice_first_result": result.get("sensevoice_first_result", {}),
                "sensevoice_raw_result": result.get("sensevoice_raw_result", []),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"ASR/SER失败: {exc}") from exc

    final_text = text_override.strip() if text_override.strip() else asr_text
    if not final_text:
        raise HTTPException(status_code=400, detail="缺少输入文本：请上传音频或填写文本覆盖。")

    text_result = detect_text_emotion_for_web(final_text)
    text_emotion = str(text_result["emotion"])
    text_conf = (
        float(text_result["confidence"])
        if text_result.get("confidence") is not None
        else float(text_result.get("rule_score") or 0.0)
    )
    context_before = get_session_history(sid)
    if ARGS.enable_llm_semantic_emotion:
        try:
            semantic_result = call_llm_semantic_emotion(
                llm_url=ARGS.llm_url,
                llm_model=ARGS.llm_model,
                user_text=final_text,
                history=context_before,
                timeout=ARGS.timeout,
                max_tokens=ARGS.semantic_max_tokens,
            )
        except Exception as exc:
            semantic_result = fallback_semantic_emotion(str(exc))
    else:
        semantic_result = fallback_semantic_emotion("disabled")
    final_emotion, fusion_reason = fuse_emotion_with_semantics(
        audio_emotion,
        text_emotion,
        text_conf,
        semantic_result,
        final_text,
    )

    try:
        llm_out = call_llm_reply_with_history(
            llm_url=ARGS.llm_url,
            llm_model=ARGS.llm_model,
            final_emotion=final_emotion,
            user_text=final_text,
            history=context_before,
            temperature=0.4,
            max_tokens=256,
            timeout=ARGS.timeout,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM请求失败: {exc}") from exc

    payload = {
        "task0_audio_asr_ser": audio_result,
        "task1_emotion_arbitration": {
            "asr_text": final_text,
            "audio_emotion_raw": audio_emo_raw,
            "audio_emotion": audio_emotion,
            "audio_judgement": audio_result,
            "text_emotion": text_emotion,
            "text_confidence": text_result.get("confidence"),
            "text_confidence_type": text_result.get("confidence_type"),
            "text_emotion_raw": text_result.get("raw_label"),
            "text_emotion_source": text_result.get("source"),
            "text_emotion_model_path": text_result.get("model_path"),
            "text_emotion_scores": text_result.get("raw_scores"),
            "text_rule_score": text_result.get("rule_score"),
            "text_emotion_fallback_error": text_result.get("fallback_error"),
            "semantic_emotion": semantic_result.get("emotion"),
            "semantic_confidence": semantic_result.get("semantic_confidence"),
            "semantic_confidence_type": semantic_result.get("semantic_confidence_type"),
            "semantic_intent": semantic_result.get("intent"),
            "semantic_valence": semantic_result.get("valence"),
            "semantic_arousal": semantic_result.get("arousal"),
            "semantic_evidence": semantic_result.get("evidence"),
            "semantic_reason": semantic_result.get("reason"),
            "semantic_source": semantic_result.get("source"),
            "semantic_error": semantic_result.get("error"),
            "final_emotion": final_emotion,
            "fusion_reason": fusion_reason,
        },
        "task2_empathic_reply": llm_out,
        "task3_tts_control": {
            "text": llm_out["reply_text"],
            "emotion": llm_out["emotion"],
            "tts_params": smooth_tts_params_for_web(build_tts_params(llm_out["emotion"])),
        },
    }
    context_after = append_session_turn(
        sid,
        user_text=final_text,
        assistant_text=llm_out["reply_text"],
        final_emotion=final_emotion,
        reply_emotion=llm_out["emotion"],
    )

    json_path = OUTPUT_JSON / f"pipeline_{request_id}.json"
    tts_audio_url = None
    want_tts = generate_tts.strip().lower() in {"1", "true", "yes", "y", "on"}
    want_cache = use_voice_cache.strip().lower() in {"1", "true", "yes", "y", "on"}
    tts_error = None
    if want_tts:
        tts_audio_url, tts_error = synthesize_tts(
            json_path, use_voice_id=use_voice_id, use_voice_cache=want_cache
        )

    payload["request_id"] = request_id
    payload["session_id"] = sid
    payload["context_turns"] = len(context_after)
    payload["context_history"] = context_after
    payload["pipeline_json"] = str(json_path)
    payload["tts_audio_url"] = tts_audio_url
    payload["tts_error"] = tts_error
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    append_persistent_turn(
        sid,
        user_text=final_text,
        audio_name=audio_name,
        audio_path=local_audio,
        payload=payload,
    )
    return payload


if __name__ == "__main__":
    import uvicorn

    auto_start_llama_server()
    preload_text_emotion_model()
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
