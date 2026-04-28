#python .\project\scripts\tts_qwen3_from_pipeline.py --pipeline-json .\project\outputs\pipeline_e2e_result.json --use-voice-id zjh --use-voice-cache --output .\project\outputs\tts_from_pipeline.wav --device cuda:0
#python .\project\scripts\tts_qwen3_from_pipeline.py --pipeline-json .\project\outputs\pipeline_e2e_result.json --use-voice-id zjh --build-voice-cache --use-voice-cache --output .\project\outputs\tts_from_pipeline.wav --device cuda:0
#python .\project\scripts\tts_qwen3_from_pipeline.py --pipeline-json .\project\outputs\pipeline_e2e_result.json --use-voice-id zjh --output .\project\outputs\tts_from_pipeline.wav --device cuda:0
#python "D:\0digi-human\project\scripts\tts_qwen3_from_pipeline.py" --pipeline-json "D:\0digi-human\project\outputs\pipeline_e2e_result.json" --ref-audio "D:\0digi-human\project\samples\20260415_193009_16k.wav" --ref-text "一二三四五。" --language "Chinese" --output "D:\0digi-human\project\outputs\tts_from_pipeline.wav" --device "cuda:0"
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_or_default(env_name: str, fallback: Path | str) -> str:
    import os

    v = os.getenv(env_name, "").strip()
    if v:
        return v
    return str(fallback)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate TTS wav from pipeline task3_tts_control")
    p.add_argument(
        "--pipeline-json",
        required=True,
        help="pipeline_e2e_demo 输出 JSON 路径（建议相对项目根目录）",
    )
    p.add_argument(
        "--tts-model",
        default=_env_or_default(
            "QWEN3_TTS_MODEL_PATH",
            PROJECT_ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base",
        ),
        help="Qwen3-TTS 模型路径（优先读取 QWEN3_TTS_MODEL_PATH）",
    )
    p.add_argument("--ref-audio", default="", help="参考音频路径（建议 16k wav）")
    p.add_argument("--ref-text", default="", help="参考音频对应文本")
    p.add_argument("--language", default="Chinese", help="TTS 语言（Chinese/English/...）")
    p.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "tts_from_pipeline.wav"),
        help="输出 wav 路径",
    )
    p.add_argument("--device", default="cuda:0", help="推理设备，如 cuda:0 或 cpu")
    p.add_argument(
        "--voice-id",
        default="default_voice",
        help="音色配置 ID，用于区分不同说话人配置",
    )
    p.add_argument(
        "--use-voice-id",
        default="",
        help="从 voice_profile.json 读取该 voice_id 的配置（ref_audio/ref_text 等）",
    )
    p.add_argument(
        "--voice-profile",
        default=_env_or_default(
            "VOICE_PROFILE_PATH",
            PROJECT_ROOT / "outputs" / "voice_profile.json",
        ),
        help="音色配置保存路径",
    )
    p.add_argument(
        "--voice-cache-dir",
        default=_env_or_default(
            "VOICE_CACHE_DIR",
            PROJECT_ROOT / "outputs" / "voice_cache",
        ),
        help="音色缓存目录（优先读取 VOICE_CACHE_DIR）",
    )
    p.add_argument(
        "--build-voice-cache",
        action="store_true",
        help="根据 ref_audio/ref_text 构建并保存 voice clone prompt 缓存",
    )
    p.add_argument(
        "--use-voice-cache",
        action="store_true",
        help="优先使用缓存的 voice clone prompt 生成",
    )
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="单次 TTS 生成超时秒数，<=0 表示不限制，默认 300",
    )
    p.add_argument(
        "--disable-postprocess",
        action="store_true",
        help="禁用 apply_tts_params 后处理，直接输出模型原始波形",
    )
    p.add_argument(
        "--enable-postprocess",
        dest="disable_postprocess",
        action="store_false",
        help="启用 apply_tts_params 后处理（默认关闭）",
    )
    p.set_defaults(disable_postprocess=True)
    return p.parse_args()


def load_task3(path: Path) -> Tuple[str, str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    task3 = data.get("task3_tts_control", {})
    text = str(task3.get("text", "")).strip()
    emotion = str(task3.get("emotion", "neutral")).strip().lower()
    params = task3.get("tts_params", {}) if isinstance(task3.get("tts_params", {}), dict) else {}
    if not text:
        text = "我在，愿意听你说。"
    return text, emotion, params


def load_voice_profile(path: Path, voice_id: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"voice profile 不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    voices = data.get("voices", {}) if isinstance(data, dict) else {}
    if not isinstance(voices, dict):
        raise KeyError(f"voice profile 结构非法，缺少 voices: {path}")

    # 优先按 voices 的键名匹配（当前推荐存储方式）
    if voice_id in voices:
        profile = voices.get(voice_id, {})
    else:
        # 兼容旧数据：键名可能是 default_voice，但内部 voice_id 才是 zjh
        profile = None
        for _, v in voices.items():
            if isinstance(v, dict) and str(v.get("voice_id", "")).strip() == voice_id:
                profile = v
                break
        if profile is None:
            available_keys = ", ".join(sorted(voices.keys()))
            raise KeyError(f"voice_id 不存在: {voice_id}；可用键: [{available_keys}]")

    if not isinstance(profile, dict):
        raise ValueError(f"voice_id 配置非法: {voice_id}")
    return profile


def _build_voice_clone_prompt(model: Qwen3TTSModel, ref_audio: Path, ref_text: str, language: str) -> Any:
    # 不同版本 qwen-tts 可能参数签名略有差异，做兼容调用。
    try:
        return model.create_voice_clone_prompt(
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            language=language,
        )
    except TypeError:
        try:
            return model.create_voice_clone_prompt(str(ref_audio), ref_text, language)
        except TypeError:
            return model.create_voice_clone_prompt(str(ref_audio), ref_text)


def _generate_with_prompt(model: Qwen3TTSModel, text: str, language: str, voice_prompt: Any) -> Tuple[Any, int]:
    # 不同版本对参数名支持可能不同，逐级尝试。
    try:
        return model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_prompt,
        )
    except TypeError:
        try:
            return model.generate_voice_clone(
                text=text,
                language=language,
                prompt=voice_prompt,
            )
        except TypeError:
            raise


def _load_voice_prompt_cache(cache_file: Path) -> Any:
    # PyTorch 2.6+ 默认 weights_only=True，读取自定义对象会失败。
    # 这里先走默认安全路径；若失败，再对“本地自建缓存”回退为 weights_only=False。
    try:
        return torch.load(str(cache_file), map_location="cpu")
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            print("[WARN] voice cache 触发 PyTorch 安全反序列化限制，回退 weights_only=False（仅限本地自建缓存）。")
            return torch.load(str(cache_file), map_location="cpu", weights_only=False)
        raise


def _ensure_mono_float32(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 2:
        wav = wav[:, 0]
    return wav.astype(np.float32, copy=False)


def apply_tts_params(wav: np.ndarray, sr: int, params: Dict[str, Any]) -> np.ndarray:
    # 延迟导入，避免没有 librosa 时影响主流程报错定位
    import librosa

    wav = _ensure_mono_float32(wav)
    speed = float(params.get("speed", 1.0))
    pitch_semitone = float(params.get("pitch_semitone", 0.0))
    energy = float(params.get("energy", 1.0))
    emo_intensity = float(params.get("emotion_intensity", 0.5))
    pause_ms = int(params.get("pause_ms", 0))

    # 语速控制：>1 更快，<1 更慢
    if abs(speed - 1.0) > 1e-3:
        # librosa time_stretch 参数是倍率（>1 更快）
        wav = librosa.effects.time_stretch(wav, rate=max(0.5, min(1.8, speed)))

    # 音高控制：单位半音
    if abs(pitch_semitone) > 1e-3:
        wav = librosa.effects.pitch_shift(
            wav,
            sr=sr,
            n_steps=max(-6.0, min(6.0, pitch_semitone)),
        )

    # 能量/情感强度控制：线性增益并做限幅
    gain = max(0.3, min(2.0, energy * (0.9 + 0.2 * emo_intensity)))
    wav = wav * gain
    peak = float(np.max(np.abs(wav)) + 1e-8)
    if peak > 0.98:
        wav = wav * (0.98 / peak)

    # 结尾停顿（毫秒）
    if pause_ms > 0:
        tail = np.zeros(int(sr * pause_ms / 1000.0), dtype=np.float32)
        wav = np.concatenate([wav, tail], axis=0)

    return wav.astype(np.float32, copy=False)


def main() -> int:
    t0 = time.perf_counter()
    args = parse_args()
    print(f"[TIME] start")
    pipeline_json = Path(args.pipeline_json)
    output = Path(args.output)
    voice_profile_path = Path(args.voice_profile)
    voice_cache_dir = Path(args.voice_cache_dir)
    selected_voice_id = args.use_voice_id.strip() or args.voice_id.strip()

    profile: Dict[str, Any] = {}
    if args.use_voice_id.strip():
        profile = load_voice_profile(voice_profile_path, args.use_voice_id.strip())
        print(f"[INFO] loaded profile voice_id={args.use_voice_id.strip()}")

    # 显式参数优先；未显式提供时从 profile 回填
    ref_audio_arg = args.ref_audio.strip() or str(profile.get("ref_audio", "")).strip()
    ref_text = args.ref_text.strip() or str(profile.get("ref_text", "")).strip()
    language = args.language.strip() or str(profile.get("language", "Chinese")).strip()
    tts_model_arg = args.tts_model.strip() or str(profile.get("tts_model", "")).strip()

    if not ref_audio_arg:
        raise SystemExit("缺少参考音频：请传 --ref-audio 或使用 --use-voice-id 从 profile 读取。")
    if not ref_text:
        raise SystemExit("缺少参考文本：请传 --ref-text 或使用 --use-voice-id 从 profile 读取。")
    if not tts_model_arg:
        raise SystemExit("缺少 TTS 模型路径：请传 --tts-model 或使用 --use-voice-id 从 profile 读取。")

    ref_audio = Path(ref_audio_arg)
    tts_model_path = Path(tts_model_arg)
    cache_file = voice_cache_dir / f"{selected_voice_id}.pt"

    if not pipeline_json.exists():
        raise SystemExit(f"pipeline json 不存在: {pipeline_json}")
    if not ref_audio.exists():
        raise SystemExit(f"参考音频不存在: {ref_audio}")
    if not tts_model_path.exists():
        raise SystemExit(f"TTS 模型路径不存在: {tts_model_path}")

    t1 = time.perf_counter()
    text, emotion, tts_params = load_task3(pipeline_json)
    print(f"[INFO] emotion={emotion}, text={text}")
    print(f"[INFO] tts_params={json.dumps(tts_params, ensure_ascii=False)}")
    print(f"[TIME] load_task3: {time.perf_counter() - t1:.2f}s")

    t2 = time.perf_counter()
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = Qwen3TTSModel.from_pretrained(
        str(tts_model_path),
        device_map=args.device,
        dtype=dtype,
    )
    print(f"[TIME] load_model: {time.perf_counter() - t2:.2f}s")

    voice_prompt = None
    t_prompt = time.perf_counter()
    if args.use_voice_cache and cache_file.exists():
        voice_prompt = _load_voice_prompt_cache(cache_file)
        print(f"[INFO] loaded voice cache: {cache_file}")
    elif args.use_voice_cache and not cache_file.exists():
        print(f"[INFO] voice cache not found, fallback build/use ref: {cache_file}")

    if args.build_voice_cache or (args.use_voice_cache and voice_prompt is None):
        voice_prompt = _build_voice_clone_prompt(model, ref_audio, ref_text, language)
        voice_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(voice_prompt, str(cache_file))
        print(f"[INFO] saved voice cache: {cache_file}")
    print(f"[TIME] voice_prompt_prepare: {time.perf_counter() - t_prompt:.2f}s")

    def _do_generate() -> Tuple[Any, int]:
        if voice_prompt is not None:
            try:
                return _generate_with_prompt(model, text, language, voice_prompt)
            except TypeError:
                # 某些版本不支持 prompt 直传，自动回退到 ref_audio/ref_text 方式
                pass
        return model.generate_voice_clone(text=text, language=language, ref_audio=str(ref_audio), ref_text=ref_text)

    t3 = time.perf_counter()
    if args.timeout_seconds > 0:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_generate)
            try:
                wavs, sr = fut.result(timeout=args.timeout_seconds)
            except FuturesTimeoutError as e:
                raise SystemExit(
                    f"TTS 生成超时：>{args.timeout_seconds}s。"
                    " 可尝试缩短文本/参考文本，或增大 --timeout-seconds。"
                ) from e
    else:
        wavs, sr = _do_generate()
    print(f"[TIME] generate_voice_clone: {time.perf_counter() - t3:.2f}s")

    t4 = time.perf_counter()
    wav = _ensure_mono_float32(np.asarray(wavs[0]))
    if not args.disable_postprocess:
        wav = apply_tts_params(wav, sr, tts_params)
    print(f"[TIME] postprocess_params: {time.perf_counter() - t4:.2f}s")

    t5 = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), wav, sr)
    print(f"[TIME] write_wav: {time.perf_counter() - t5:.2f}s")

    t6 = time.perf_counter()
    profile = {
        "voice_id": selected_voice_id,
        "tts_model": str(tts_model_path),
        "language": language,
        "device": args.device,
        "ref_audio": str(ref_audio),
        "ref_text": ref_text,
        "pipeline_json": str(pipeline_json),
        "emotion": emotion,
        "tts_params": tts_params,
        "last_output_wav": str(output),
        "sample_rate": int(sr),
        "voice_cache_file": str(cache_file),
    }
    voice_profile_path.parent.mkdir(parents=True, exist_ok=True)
    if voice_profile_path.exists():
        try:
            existing = json.loads(voice_profile_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {"voices": {}}
        except Exception:
            existing = {"voices": {}}
    else:
        existing = {"voices": {}}
    if "voices" not in existing or not isinstance(existing["voices"], dict):
        existing["voices"] = {}
    existing["voices"][selected_voice_id] = profile
    voice_profile_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[TIME] save_voice_profile: {time.perf_counter() - t6:.2f}s")
    print(f"[OK] saved voice profile: {voice_profile_path} (voice_id={selected_voice_id})")

    print(f"[OK] saved wav: {output}")
    print(f"[TIME] total: {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

