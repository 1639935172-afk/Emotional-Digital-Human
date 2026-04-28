from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from funasr import AutoModel

from pipeline_e2e_demo import (
    build_tts_params,
    call_llm_reply,
    detect_text_emotion,
    fuse_emotion,
    parse_sensevoice_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "web_demo"
OUTPUT_AUDIO = OUTPUT_ROOT / "audio"
OUTPUT_JSON = OUTPUT_ROOT / "json"

DEFAULT_SV_MODEL = os.getenv("SENSEVOICE_MODEL", "iic/SenseVoiceSmall")
DEFAULT_DEVICE = os.getenv("SENSEVOICE_DEVICE", "cuda:0")
DEFAULT_LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3-8B-Q4_K_M.gguf")
DEFAULT_VOICE_ID = os.getenv("WEB_DEMO_DEFAULT_VOICE_ID", "").strip()
DEFAULT_VOICE_PROFILE = Path(
    os.getenv("VOICE_PROFILE_PATH", str(PROJECT_ROOT / "outputs" / "voice_profile.json"))
)
DEFAULT_WEB_REF_AUDIO = os.getenv("WEB_TTS_REF_AUDIO", "").strip()
DEFAULT_WEB_REF_TEXT = os.getenv("WEB_TTS_REF_TEXT", "").strip()

SENSEVOICE_MODEL: Optional[AutoModel] = None
MODEL_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Web demo for multimodal emotional interaction")
    p.add_argument("--host", default="127.0.0.1", help="web host")
    p.add_argument("--port", type=int, default=7860, help="web port")
    p.add_argument("--device", default=DEFAULT_DEVICE, help="SenseVoice device, e.g. cuda:0/cpu")
    p.add_argument("--sv-model", default=DEFAULT_SV_MODEL, help="SenseVoice model id/path")
    p.add_argument("--llm-url", default=DEFAULT_LLM_URL, help="OpenAI-compatible chat completions URL")
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="llm model field")
    p.add_argument("--disable-update", action="store_true", help="disable funasr update check")
    p.add_argument("--timeout", type=int, default=60, help="llm timeout seconds")
    return p.parse_args()


ARGS = parse_args()


def ensure_dirs() -> None:
    OUTPUT_AUDIO.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.mkdir(parents=True, exist_ok=True)


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
        "audio_emotion_raw": audio_emo_raw,
        "audio_emotion": audio_emotion,
    }


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
    }


@app.post("/api/analyze")
async def analyze(
    audio: Optional[UploadFile] = File(default=None),
    text_override: str = Form(default=""),
    generate_tts: str = Form(default="true"),
    use_voice_id: str = Form(default=""),
    use_voice_cache: str = Form(default="true"),
) -> Dict[str, Any]:
    ensure_dirs()
    request_id = uuid.uuid4().hex
    asr_text = ""
    audio_emo_raw = "EMO_UNK"
    audio_emotion = "neutral"

    if audio is not None:
        suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
        local_audio = OUTPUT_AUDIO / f"input_{request_id}{suffix}"
        local_audio.write_bytes(await audio.read())
        try:
            result = run_asr_ser(local_audio)
            asr_text = result["asr_text"]
            audio_emo_raw = result["audio_emotion_raw"]
            audio_emotion = result["audio_emotion"]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"ASR/SER失败: {exc}") from exc

    final_text = text_override.strip() if text_override.strip() else asr_text
    if not final_text:
        raise HTTPException(status_code=400, detail="缺少输入文本：请上传音频或填写文本覆盖。")

    text_emotion, text_conf = detect_text_emotion(final_text)
    final_emotion, fusion_reason = fuse_emotion(audio_emotion, text_emotion, text_conf, final_text)

    try:
        llm_out = call_llm_reply(
            llm_url=ARGS.llm_url,
            llm_model=ARGS.llm_model,
            final_emotion=final_emotion,
            asr_text=final_text,
            temperature=0.4,
            max_tokens=128,
            timeout=ARGS.timeout,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM请求失败: {exc}") from exc

    payload = {
        "task1_emotion_arbitration": {
            "asr_text": final_text,
            "audio_emotion_raw": audio_emo_raw,
            "audio_emotion": audio_emotion,
            "text_emotion": text_emotion,
            "text_confidence": text_conf,
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

    json_path = OUTPUT_JSON / f"pipeline_{request_id}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    tts_audio_url = None
    want_tts = generate_tts.strip().lower() in {"1", "true", "yes", "y", "on"}
    want_cache = use_voice_cache.strip().lower() in {"1", "true", "yes", "y", "on"}
    tts_error = None
    if want_tts:
        tts_audio_url, tts_error = synthesize_tts(
            json_path, use_voice_id=use_voice_id, use_voice_cache=want_cache
        )

    payload["request_id"] = request_id
    payload["pipeline_json"] = str(json_path)
    payload["tts_audio_url"] = tts_audio_url
    payload["tts_error"] = tts_error
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
