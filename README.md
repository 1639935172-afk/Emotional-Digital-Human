# 数字人多模态情感交互 Demo

端到端流程：

`音频/文本输入 -> ASR -> SER + 文本情感 + Qwen 语义情绪判断 -> 情感仲裁融合 -> Qwen 共情回复(JSON) -> 情感 TTS -> 音频输出`

主情感标签统一为：

`happy / sad / angry / neutral`

当前目标是先保证 Web Demo 稳定可演示，再继续优化情绪指标、TTS 听感和测试覆盖。

## 当前状态

- 本地 `Qwen3-8B GGUF + llama.cpp` 已接入 OpenAI 兼容接口。
- Web Demo 可自动检查并启动本地 `llama-server`。
- 默认 GGUF：
  - `project/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`
- SenseVoice 用于音频 ASR/SER，解析 `<|HAPPY|>`、`<|SAD|>`、`<|ANGRY|>`、`<|NEUTRAL|>` 等情绪 token。
- 文本情感模型：
  - `project/scripts/text_emotion_model.py`
  - `project/models/Chinese-Emotion-Small`
  - `text_confidence` 是真实 softmax 概率。
- Web Demo 支持：
  - 文本输入和音频输入
  - 多轮上下文
  - 左侧历史会话
  - 会话置顶、重命名、删除
  - 会话 JSON 持久化
  - 完整响应 JSON 面板
  - Qwen 语义情绪判断
  - Qwen 共情回复
  - TTS 音频输出

## 关键脚本

- `project/scripts/web_demo_app.py`
  - Web Demo 主入口。
  - 自动启动 `llama-server`。
  - 执行 ASR/SER、文本情感、Qwen 语义判断、融合仲裁、Qwen 回复、TTS。
  - 保存会话 JSON 和每轮 pipeline JSON。

- `project/scripts/text_emotion_model.py`
  - 中文文本情感模型封装。
  - 默认加载 `project/models/Chinese-Emotion-Small`。

- `project/scripts/start_llama_server.py`
  - 单独启动 llama-server。
  - Web Demo 当前也能自动启动同一 GGUF 模型。

- `project/scripts/tts_qwen3_from_pipeline.py`
  - 从 pipeline JSON 生成 Qwen3-TTS 音频。

- `project/scripts/pipeline_e2e_demo.py`
  - 命令行端到端 pipeline。

- `project/scripts/sensevoice_asr_ser.py`
  - SenseVoice ASR/SER 命令行脚本。

以下是临时 probe 脚本，不属于当前主链路必需文件：

- `project/scripts/hf_chinese_emotion_probe.py`
- `project/scripts/sensevoice_text_emotion_probe.py`

## 快速启动

推荐在 `sensevoice` 环境运行：

```powershell
conda activate sensevoice; python "D:\0digi-human\project\scripts\web_demo_app.py"
```

启动后访问：

```text
http://127.0.0.1:7860
```

指定会话目录：

```powershell
conda activate sensevoice; python "D:\0digi-human\project\scripts\web_demo_app.py" --session-dir "D:\0digi-human\project\outputs\web_demo\sessions"
```

不自动启动 LLM：

```powershell
conda activate sensevoice; python "D:\0digi-human\project\scripts\web_demo_app.py" --no-auto-start-llm
```

关闭 Qwen 语义情绪判断：

```powershell
conda activate sensevoice; python "D:\0digi-human\project\scripts\web_demo_app.py" --disable-llm-semantic-emotion
```

如果 SenseVoice GPU 遇到 CUDA kernel 兼容问题，可以先让 ASR/SER 用 CPU，文本情感仍用 CUDA：

```powershell
conda activate sensevoice; python "D:\0digi-human\project\scripts\web_demo_app.py" --device cpu --text-emotion-device cuda:0
```

固定 TTS 参考音频：

```powershell
$env:WEB_TTS_REF_AUDIO="D:\0digi-human\project\samples\20260415_193009_16k.wav"; $env:WEB_TTS_REF_TEXT="一二三四五。"; conda activate sensevoice; python "D:\0digi-human\project\scripts\web_demo_app.py"
```

## 当前 torch / torchaudio 说明

RTX 5060 Laptop GPU 需要支持 `sm_120` 的 torch wheel。当前可用组合以本机实测为准：

- `torch 2.10.0.dev20251118+cu128`
- `torchaudio 2.9.1+cu128`
- CUDA wheel：`cu128`

验证命令：

```powershell
conda activate sensevoice; python -c "import torch, torchaudio; print(torch.__version__); print(torchaudio.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_capability(0))"
```

不要随意替换 torch wheel，否则 5060 的 `sm_120` CUDA 算子可能不可用。

## 文本情感模型

基础测试：

```powershell
conda run -n sensevoice python project\scripts\text_emotion_model.py "我今天真的很开心"
```

紧凑 JSON 输出：

```powershell
conda run -n sensevoice python project\scripts\text_emotion_model.py "我有点难过，感觉撑不住了" --compact
```

指定模型和设备：

```powershell
conda run -n sensevoice python project\scripts\text_emotion_model.py "我现在很担心任务完成不了" --model "D:\0digi-human\project\models\Chinese-Emotion-Small" --device cuda:0
```

输出字段：

- `emotion`：映射后的四分类情绪。
- `confidence`：softmax 概率。
- `confidence_type`：通常为 `softmax_probability`。
- `raw_label`：模型原始标签。
- `raw_scores`：原始标签分数。

## Qwen 调用说明

Web Demo 中 Qwen 有两次独立调用：

1. `call_llm_semantic_emotion()`
   - 用于语义情绪判断。
   - 默认 `temperature=0.1`。
   - 输出 `semantic_emotion`、`semantic_confidence`、`intent`、`valence`、`arousal`、`evidence`、`reason`。

2. `call_llm_reply_with_history()`
   - 用于生成共情回复。
   - 默认 `temperature=0.4`，`max_tokens=256`。
   - 输出 JSON：`{"emotion":"happy|sad|angry|neutral","reply_text":"string"}`。
   - `emotion` 会被强制对齐到仲裁后的 `final_emotion`。

兼容说明：

- llama-server 有时会返回 `message.content=""`，但真实 JSON 在 `message.reasoning_content`。
- Web Demo 已兼容读取 `reasoning_content`。
- 完整响应中会记录：
  - `llm_raw_content`
  - `llm_content_source`
  - `llm_parse_error`
  - `llm_fallback_used`
  - `llm_empty_retry_count`

## Web Demo 会话持久化

默认会话目录：

```text
project/outputs/web_demo/sessions
```

结构：

```text
project/outputs/web_demo/sessions/
  index.json
  <session_id>.json
```

每个 `<session_id>.json` 保存：

- `session_id`
- `title`
- `pinned`
- `created_at`
- `updated_at`
- `turns`
- 每轮 user 文本、音频路径
- 每轮 assistant 回复、音频路径
- 每轮完整 `full_response`

每轮 pipeline JSON 保存到：

```text
project/outputs/web_demo/json/pipeline_<request_id>.json
```

删除历史会话时会同步删除：

- `project/outputs/web_demo/sessions/<session_id>.json`
- 该 session 引用的 `project/outputs/web_demo/json/pipeline_*.json`
- 并重建 `project/outputs/web_demo/sessions/index.json`

删除接口返回：

- `deleted_json`
- `deleted_json_path`
- `deleted_pipeline_jsons`
- `skipped_pipeline_jsons`
- `index`

## 完整响应 JSON

当前 `/api/analyze` 返回的主要结构：

- `task0_audio_asr_ser`
  - SenseVoice 原始返回和解析后的语音判断。
- `task1_emotion_arbitration`
  - ASR 文本、语音情绪、文本情绪、Qwen 语义情绪、最终情绪和融合原因。
- `task2_empathic_reply`
  - Qwen 回复 JSON、原始输出、解析诊断。
- `task3_tts_control`
  - TTS 文本、情绪和控制参数。
- `request_id`
- `session_id`
- `context_history`
- `pipeline_json`
- `tts_audio_url`
- `tts_error`

## 情感融合仲裁

当前 Web Demo 的情绪来源：

1. SenseVoice 语音情绪：`audio_emotion`
2. 中文文本情感模型：`text_emotion + text_confidence`
3. 本地 Qwen 语义情绪判断：`semantic_emotion + semantic_confidence`

典型规则：

- 语音为 `neutral`，但 Qwen 语义为 `sad/angry` 且置信较高时，采用 Qwen 语义情绪。
- 文本模型为 `neutral`，但 Qwen 语义识别到担心、焦虑、压力、愤怒等时，采用 Qwen 语义情绪。
- 语音为 `angry` 且语义/文本偏中性时，保留语音高唤醒信号。
- 多源一致时采用一致情绪。
- 其他情况回退基础融合规则。

示例：

```json
{
  "final_emotion": "sad",
  "fusion_reason": "rule:audio_neutral_semantic_negative_override"
}
```

## TTS

Web Demo 调用：

```text
project/scripts/tts_qwen3_from_pipeline.py
```

当前 Web Demo 默认使用 `--disable-postprocess`，优先保证清晰度。

单独从 pipeline JSON 生成 TTS：

```powershell
python ".\project\scripts\tts_qwen3_from_pipeline.py" --pipeline-json ".\project\outputs\pipeline_e2e_result.json" --use-voice-id zjh --use-voice-cache --output ".\project\outputs\tts_from_pipeline.wav" --device cuda:0
```

## 常用环境变量

- `LLAMA_SERVER_EXE`
- `LLAMA_MODEL_PATH`
- `LLM_URL`
- `LLM_MODEL`
- `TEXT_EMOTION_MODEL`
- `TEXT_EMOTION_DEVICE`
- `WEB_DEMO_SESSION_DIR`
- `WEB_TTS_REF_AUDIO`
- `WEB_TTS_REF_TEXT`
- `VOICE_PROFILE_PATH`
- `VOICE_CACHE_DIR`

## Git 提交注意

提交前必须白名单检查，避免模型、缓存、音频、outputs、sessions JSON 被误传。

不要使用：

```powershell
git add .
```

先检查：

```powershell
git status --short
git diff --cached --name-only
```

按需只添加明确文件，例如：

```powershell
git add README.md project/scripts/web_demo_app.py project/scripts/text_emotion_model.py project/scripts/tts_qwen3_from_pipeline.py project/scripts/pipeline_e2e_demo.py project/scripts/sensevoice_asr_ser.py project/scripts/start_llama_server.py
```

当前两个 probe 脚本不是主链路必需文件，提交前默认不要加入：

```text
project/scripts/hf_chinese_emotion_probe.py
project/scripts/sensevoice_text_emotion_probe.py
```
