# 数字人多模态情感交互 Demo

本项目用于构建一个可演示的端到端数字人多模态情感交互闭环：

`音频/文本输入 -> ASR -> SER + 文本情感 + Qwen 语义情绪判断 -> 情感仲裁融合 -> LLM 共情回复(JSON) -> 情感 TTS -> 音频输出`

主情感标签统一为：`happy / sad / angry / neutral`。当前目标是先完成稳定可演示版本，再继续优化指标、语音质量和评测覆盖。

---

## 当前状态

- 本地 `Qwen3-8B GGUF + llama.cpp` 已接入 OpenAI 兼容接口。
- `web_demo_app.py` 默认可自动启动本地 llama-server。
- 默认 GGUF 模型路径：
  - `project/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`
- SenseVoice 已用于：
  - ASR 音频转文本
  - SER 语音情感 token 解析
- 文本情感已接入本地开源模型：
  - `project/models/Chinese-Emotion-Small`
  - 模块：`project/scripts/text_emotion_model.py`
  - `text_confidence` 来自模型 softmax 概率
- Web Demo 已支持：
  - 多轮上下文对话
  - 每个对话窗口持久化为 JSON
  - 左侧历史会话列表
  - 会话置顶、重命名、删除
  - 重启 Web Demo 后恢复历史会话和完整响应
  - Qwen 语义情绪判断 + SenseVoice 语音情绪 + 文本模型情绪的融合仲裁
  - 可选 TTS 生成和网页播放
- TTS 当前默认高保真优先：
  - 默认禁用后处理
  - 支持固定参考音频
  - 支持 `voice_id` 和 voice cache

---

## 主要脚本

- `project/scripts/web_demo_app.py`
  - Web Demo 主入口
  - 自动启动 llama-server
  - 加载文本情感模型
  - 执行 ASR/SER、文本情感、Qwen 语义判断、融合仲裁、共情回复、TTS
  - 管理会话 JSON 持久化

- `project/scripts/text_emotion_model.py`
  - 中文文本情感分类模块
  - 默认模型路径：`project/models/Chinese-Emotion-Small`
  - 支持命令行测试

- `project/scripts/start_llama_server.py`
  - 单独启动 llama-server
  - Web Demo 现在也能自动启动同一 GGUF 模型

- `project/scripts/pipeline_e2e_demo.py`
  - 离线端到端 pipeline

- `project/scripts/sensevoice_asr_ser.py`
  - SenseVoice ASR + SER 单独测试

- `project/scripts/tts_qwen3_from_pipeline.py`
  - 从 pipeline JSON 生成 Qwen3-TTS 音频

- `project/scripts/convert_audio_to_wav.py`
  - 批量转换 `project/samples` 下音频为 16k wav

---

## 快速启动

推荐在 `sensevoice` 环境运行 Web Demo：

```powershell
conda activate sensevoice; python "D:\0digi-human\project\scripts\web_demo_app.py"
```

默认会：

- 自动检查/启动 llama-server
- 使用默认 Qwen3-8B GGUF
- 预加载本地文本情感模型
- 启动 Web Demo：`http://127.0.0.1:7860`

显式指定会话目录：

```powershell
conda activate sensevoice; python "D:\0digi-human\project\scripts\web_demo_app.py" --session-dir "D:\0digi-human\project\outputs\web_demo\sessions"
```

不自动启动 LLM：

```powershell
conda activate sensevoice; python "D:\0digi-human\project\scripts\web_demo_app.py" --no-auto-start-llm
```

不预加载文本情感模型：

```powershell
conda activate sensevoice; python "D:\0digi-human\project\scripts\web_demo_app.py" --no-preload-text-emotion
```

关闭 Qwen 语义情绪判断：

```powershell
conda activate sensevoice; python "D:\0digi-human\project\scripts\web_demo_app.py" --disable-llm-semantic-emotion
```

---

## 文本情感模型测试

直接测试：

```powershell
conda run -n sensevoice python project\scripts\text_emotion_model.py "我今天真的很开心"
```

紧凑 JSON 输出：

```powershell
conda run -n sensevoice python project\scripts\text_emotion_model.py "我有点难过，感觉撑不住了" --compact
```

指定模型路径：

```powershell
conda run -n sensevoice python project\scripts\text_emotion_model.py "我现在很担心任务完成不了" --model "D:\0digi-human\project\models\Chinese-Emotion-Small" --device cuda:0
```

输出字段说明：

- `emotion`：映射后的四类情感
- `confidence`：模型 softmax 概率
- `confidence_type`：`softmax_probability`
- `raw_label`：模型原始标签
- `raw_scores`：原始类别概率分布

---

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

- 会话标题
- 是否置顶
- 创建/更新时间
- 每轮用户输入
- 每轮 AI 回复
- 完整响应 JSON
- TTS 音频 URL

重启 Web Demo 时，前端会通过 `/api/sessions` 和 `/api/sessions/{session_id}` 恢复历史会话，并将历史 turns 重建为后端上下文。

---

## 情感仲裁逻辑

当前 Web Demo 的情绪来源包括：

1. SenseVoice 语音情绪：`audio_emotion`
2. 中文文本情感模型：`text_emotion + text_confidence`
3. 本地 Qwen 语义情绪判断：`semantic_emotion`

融合时优先处理：

- 语音 `neutral`，但 Qwen 语义为 `sad/angry` 且置信较高
- 文本模型 `neutral`，但 Qwen 语义识别到担心、焦虑、压力、愤怒等
- 语音 `angry` 等高唤醒情绪与文本中性冲突
- 多源一致时直接采用一致情绪

最终输出示例：

```json
{
  "final_emotion": "sad",
  "fusion_reason": "rule:audio_neutral_semantic_negative_override"
}
```

---

## 共情回复逻辑

回复由本地 Qwen 生成，但 `emotion` 会被强制对齐到仲裁后的 `final_emotion`。

当前回复原则：

- 普通问题直接回答
- 用户表达难受、担心、生气时，先回应情绪，再给一句有用支持
- 用户明确问怎么办时，再给具体建议
- 用户感谢、确认或打招呼时，简短回应
- 不编造用户身份
- 不把所有话题都套成任务/作业/步骤建议
- LLM 未返回 JSON 时会自动容错，不让 Web Demo 整轮失败

---

## TTS 说明

当前 TTS 默认高保真优先，`tts_qwen3_from_pipeline.py` 默认禁用后处理。

固定参考音频推荐：

```powershell
$env:WEB_TTS_REF_AUDIO="D:\0digi-human\project\samples\20260415_193009_16k.wav"; $env:WEB_TTS_REF_TEXT="一二三四五。"; conda activate sensevoice; python "D:\0digi-human\project\scripts\web_demo_app.py"
```

离线生成：

```powershell
python ".\project\scripts\tts_qwen3_from_pipeline.py" --pipeline-json ".\project\outputs\pipeline_e2e_result.json" --use-voice-id zjh --use-voice-cache --output ".\project\outputs\tts_from_pipeline.wav" --device cuda:0
```

---

## 重要环境变量

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

---

## Git 提交注意

模型、缓存、输出音频、会话 JSON 不应提交。

提交前检查：

```powershell
git status --short
git diff --cached --name-only
```

建议白名单提交：

```powershell
git add README.md check.txt project/scripts/web_demo_app.py project/scripts/text_emotion_model.py
```

确认没有模型、大音频、缓存后再提交：

```powershell
git commit -m "Update web demo context and emotion arbitration"
git push origin master
```
