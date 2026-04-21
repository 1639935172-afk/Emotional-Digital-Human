# 数字人多模态情感交互 Demo（14 天交付版）

本项目目标是构建一个可演示的端到端闭环：

`音频输入 -> ASR -> 情感识别/仲裁 -> LLM 共情回复(JSON) -> 情感TTS -> 音频输出`

当前已完成任务 1/2 的可运行版本，并已输出任务 3 的结构化 TTS 控制参数。

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
- 已实现任务 3 参数层：输出可驱动 TTS 的 `tts_params`（语速、音高、能量、强度、停顿）

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
- 若部署机器路径变化，优先使用环境变量：
  - `LLAMA_SERVER_EXE`
  - `LLAMA_MODEL_PATH`

---

## 7. 下一步（任务 3 完整闭环）

- 将 `task3_tts_control` 直接接入 Qwen3-TTS 推理脚本，输出最终 `wav`
- 增加批量音频评测脚本（端到端成功率、延迟、情感一致性）
- 增加简易 Web UI（录音上传、情感标签显示、回复文本、音频播放）

