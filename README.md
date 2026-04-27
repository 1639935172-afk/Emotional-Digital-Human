# 数字人多模态情感交互 Demo

本项目目标是构建一个可演示的端到端闭环：

`音频输入 -> ASR -> 情感识别/仲裁 -> LLM 共情回复(JSON) -> 情感TTS -> 音频输出`

当前已完成任务 1/2/3 的可运行闭环版本（含音频输出）。

---

## 1. 当前实现状态

- 已跑通 `Qwen3-8B-GGUF + llama.cpp` 本地推理与 API 服务
- 已完成 LLM JSON 稳定性回归：20/20 通过，JSON 可解析率 100%
- 已跑通 `SenseVoice`（GPU）：
  - 音频转文本（ASR）
  - 语音情感标签提取（SER token）
- 已实现任务 1：规则仲裁输出 `final_emotion`
- 已实现任务 2：基于仲裁情感生成共情回复，结构化输出：
  - `emotion`
  - `reply_text`
- 已实现任务 3：`task3_tts_control -> Qwen3-TTS -> wav` 音频输出闭环
- 已支持音色配置保存与复用：
  - `voice_profile.json`（按 `voice_id` 管理）
  - `voice cache`（`create_voice_clone_prompt` 缓存并复用）

---

## 2. 目录说明

- `project/scripts/start_llama_server.py`  
  启动 `llama-server`（支持自动打开网页，支持环境变量软编码）
- `project/scripts/regression_20.py`  
  对本地 LLM 接口做 20 条 JSON 输出回归
- `project/scripts/sensevoice_asr_ser.py`  
  运行 SenseVoice，输出 `asr_text + audio_emotion`
- `project/scripts/pipeline_e2e_demo.py`  
  端到端串联任务 1 + 任务 2 + 任务 3 参数层
- `project/scripts/convert_audio_to_wav.py`  
  将 `m4a/mp3/flac/...` 批量转换为 `16k` 单声道 `wav`
- `project/scripts/tts_qwen3_from_pipeline.py`  
  读取 `pipeline_e2e_result.json` 的 `task3_tts_control`，调用 Qwen3-TTS 生成 `wav`，并支持 `voice_id` 与缓存
- `project/models/`  
  本地模型目录（Qwen3-8B-GGUF、Qwen3-TTS、Tokenizer 等）
- `project/outputs/`  
  结果输出目录（回归报告、pipeline 输出、音频文件）

---

## 2.1 第三方开源项目说明（SenseVoice）

本项目中的 ASR 与语音情感识别能力基于第三方开源项目 **SenseVoice**，该项目并非本仓库原创。

- Project: SenseVoice
- Source: [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- Main capability used here: ASR + SER (emotion tokens)
- Upstream docs: `SenseVoice-main/README_zh.md`

本仓库对 SenseVoice 的使用方式主要是工程集成（调用推理接口、解析输出标签、映射到 4 类情感），不包含对 SenseVoice 模型本体权重的再发布。  
请在实际分发与商用前，遵循上游仓库与模型页面中的许可证与使用条款。

---

## 3. 环境建议

推荐使用两个环境隔离：

- `digi-human`：LLM 侧（llama.cpp 客户端/服务调用）
- `sensevoice`：ASR/SER/TTS 侧（funasr、qwen-tts 等）

> 注意：RTX 5060（sm_120）建议使用支持新架构的 PyTorch nightly cu128 组合。

---

## 4. 快速开始

### 4.1 启动本地 LLM 服务（llama-server）

```powershell
python "D:\0digi-human\project\scripts\start_llama_server.py"
```

默认会自动打开浏览器。若不想自动打开：

```powershell
python "D:\0digi-human\project\scripts\start_llama_server.py" --no-browser
```

### 4.2 运行 LLM JSON 回归（20 条）

```powershell
python "D:\0digi-human\project\scripts\regression_20.py"
```

输出报告：

`D:\0digi-human\project\outputs\regression_20_report.json`

### 4.3 运行 SenseVoice ASR + SER

```powershell
python "D:\0digi-human\project\scripts\sensevoice_asr_ser.py" --audio "D:\0digi-human\project\samples\20260415_193009.m4a" --device "cuda:0" --disable-update
```

### 4.4 运行端到端任务 1/2/3（参数层）

```powershell
python "D:\0digi-human\project\scripts\pipeline_e2e_demo.py" --audio "D:\0digi-human\project\samples\20260415_193009.m4a" --device "cuda:0" --disable-update --llm-url "http://127.0.0.1:8080/v1/chat/completions" --llm-model "Qwen3-8B-Q4_K_M.gguf" --output "D:\0digi-human\project\outputs\pipeline_e2e_result.json"
```

### 4.5 参考音频转 16k wav（建议）

```powershell
python ".\project\scripts\convert_audio_to_wav.py" --input ".\project\samples\20260415_193009.m4a" --output-dir ".\project\samples" --overwrite
```

### 4.6 任务三音频合成（Qwen3-TTS）

首次（构建音色缓存并生成）：

```powershell
python ".\project\scripts\tts_qwen3_from_pipeline.py" --pipeline-json ".\project\outputs\pipeline_e2e_result.json" --ref-audio ".\project\samples\20260415_193009_16k.wav" --ref-text "一二三四五。" --voice-id zjh --build-voice-cache --use-voice-cache --output ".\project\outputs\tts_from_pipeline.wav" --device cuda:0
```

后续（直接复用 voice_id + cache）：

```powershell
python ".\project\scripts\tts_qwen3_from_pipeline.py" --pipeline-json ".\project\outputs\pipeline_e2e_result.json" --use-voice-id zjh --use-voice-cache --output ".\project\outputs\tts_from_pipeline.wav" --device cuda:0
```
---

## 5. 关键输出格式

`pipeline_e2e_demo.py` 输出示意：

```json
{
  "task1_emotion_arbitration": {
    "asr_text": "...",
    "audio_emotion": "neutral",
    "text_emotion": "sad",
    "final_emotion": "sad",
    "fusion_reason": "rule:audio_neutral_text_negative_override"
  },
  "task2_empathic_reply": {
    "emotion": "sad",
    "reply_text": "听起来你有点难受，我在这里陪你。"
  },
  "task3_tts_control": {
    "text": "听起来你有点难受，我在这里陪你。",
    "emotion": "sad",
    "tts_params": {
      "speed": 0.9,
      "pitch_semitone": -1.2,
      "energy": 0.82,
      "emotion_intensity": 0.68,
      "pause_ms": 240
    }
  }
}
```

---

## 6. 已知问题与建议

- `m4a` 作为 TTS 参考音频时，可能触发 `sox` 相关警告；建议先转为 `16k wav` 再做克隆
- `flash-attn` 未安装时仅影响速度，不影响功能可用性
- `voice cache` 在 PyTorch 2.6+ 下可能触发反序列化安全限制；脚本已做兼容回退处理
- 若部署机器路径变化，优先使用环境变量：
  - `LLAMA_SERVER_EXE`
  - `LLAMA_MODEL_PATH`
  - `QWEN3_TTS_MODEL_PATH`
  - `VOICE_PROFILE_PATH`
  - `VOICE_CACHE_DIR`

---

## 7. 下一步（交付收尾）

- 生成 4 类情感样本音频（Happy/Sad/Angry/Neutral）并整理到 `project/outputs/examples/`
- 增加轻量评测表（ASR/SER/融合仲裁/JSON 成功率 + 3 个完整用例）
- 录制 1 个冲突用例演示视频（语音 neutral + 文本负向）
- 完善 `.gitignore` 与仓库发布说明（避免上传权重与缓存）

