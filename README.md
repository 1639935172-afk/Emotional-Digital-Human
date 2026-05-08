# 数字人多模态情感交互 Demo

本项目实现一个可演示的端到端数字人多模态情感交互流程：

`音频/文本输入 -> ASR -> SER -> 文本情感识别 -> Qwen 语义情绪判断 -> 情感融合仲裁 -> Qwen 共情回复(JSON) -> 情感 TTS -> 音频输出`

主情感标签统一为：`happy / sad / angry / neutral`。

## 已完成功能

- Web Demo：文本输入、语音输入、情绪识别结果展示、共情回复展示、TTS 音频播放、完整 JSON 响应查看。
- ASR/SER：基于 SenseVoice 识别语音文本，并解析语音情感 token。
- 文本情感：本地中文情感分类模型 `Chinese-Emotion-Small`，输出 softmax 置信度。
- Qwen 语义情绪判断：本地 Qwen3-8B GGUF 通过 llama.cpp OpenAI 兼容接口调用。
- 情感融合：融合语音情感、文本情感、Qwen 语义情绪，输出最终四分类情绪。
- 共情回复：Qwen 输出严格 JSON：`{"emotion":"...","reply_text":"..."}`。
- TTS：根据回复文本和情绪参数生成语音，可指定参考音频克隆音色。
- 会话持久化：支持历史会话、重命名、置顶、删除；删除会话时同步清理对应 JSON。

## 目录说明

```text
project/scripts/web_demo_app.py              Web Demo 主入口
project/scripts/start_llama_server.py        单独启动 llama-server
project/scripts/text_emotion_model.py        中文文本情感模型封装
project/scripts/sensevoice_asr_ser.py        SenseVoice ASR/SER 单独测试
project/scripts/pipeline_e2e_demo.py         命令行端到端 pipeline
project/scripts/tts_qwen3_from_pipeline.py   从 pipeline JSON 生成 TTS 音频
project/scripts/convert_audio_to_wav.py      音频转 16k wav 工具
```

## 模型文件

模型文件不上传 GitHub，需要本地放置：

```text
project/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
project/models/Chinese-Emotion-Small/
project/models/Qwen3-TTS-12Hz-0.6B-Base/
```

如果模型路径不同，通过环境变量覆盖。

## 推荐环境

本项目主要在 Windows + conda 环境下验证：

```powershell
conda activate sensevoice
```

RTX 5060 Laptop GPU 需要支持 `sm_120` 的 torch wheel。当前本机实测可用组合：

```text
torch 2.10.0.dev20251118+cu128
torchaudio 2.9.1+cu128
```

验证命令：

```powershell
python -c "import torch, torchaudio; print(torch.__version__); print(torchaudio.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## 环境变量

常用环境变量如下：

```text
LLAMA_SERVER_EXE      llama-server.exe 路径
LLAMA_MODEL_PATH      Qwen3 GGUF 模型路径
LLM_URL               OpenAI 兼容 chat completions 地址
LLM_HOST              默认 127.0.0.1
LLM_PORT              默认 8080
LLM_MODEL             默认 Qwen3-8B-Q4_K_M.gguf
SENSEVOICE_MODEL      默认 iic/SenseVoiceSmall
SENSEVOICE_DEVICE     默认 cuda:0
TEXT_EMOTION_MODEL    文本情感模型路径
TEXT_EMOTION_DEVICE   文本情感模型设备
WEB_DEMO_HOST         默认 127.0.0.1
WEB_DEMO_PORT         默认 7860
WEB_DEMO_SESSION_DIR  会话保存目录
WEB_TTS_REF_AUDIO     TTS 参考音频
WEB_TTS_REF_TEXT      TTS 参考音频对应文本
VOICE_PROFILE_PATH    voice profile JSON 路径
VOICE_CACHE_DIR       voice cache 目录
TTS_DEVICE            TTS 推理设备
```

示例，设置 llama-server 和 GGUF 路径：

```powershell
$env:LLAMA_SERVER_EXE="你的llama-server.exe路径"; $env:LLAMA_MODEL_PATH="你的Qwen3 GGUF路径"
```

示例，设置 TTS 参考音频：

```powershell
$env:WEB_TTS_REF_AUDIO="project\samples\2tts_16k.wav"; $env:WEB_TTS_REF_TEXT="清晨推开窗就能感受到微风拂面，世间万物都在安静慢慢生长，保持平和的心态，认真过好当下的每一天，用心感受生活里每一份细碎的美好与温柔。"
```

## 启动 Web Demo

在仓库根目录运行：

```powershell
conda activate sensevoice; python "project\scripts\web_demo_app.py"
```

浏览器访问：

```text
http://127.0.0.1:7860
```

如果不希望 Web Demo 自动启动 llama-server：

```powershell
conda activate sensevoice; python "project\scripts\web_demo_app.py" --no-auto-start-llm
```

如果需要指定会话目录：

```powershell
conda activate sensevoice; python "project\scripts\web_demo_app.py" --session-dir "project\outputs\web_demo\sessions"
```

## 单独启动 llama-server

```powershell
conda activate sensevoice; python "project\scripts\start_llama_server.py"
```

也可以直接传路径：

```powershell
conda activate sensevoice; python "project\scripts\start_llama_server.py" --server-exe "你的llama-server.exe路径" --model "你的Qwen3 GGUF路径"
```

## 文本情感测试

```powershell
conda run -n sensevoice python project\scripts\text_emotion_model.py "我今天真的很开心"
```

紧凑 JSON 输出：

```powershell
conda run -n sensevoice python project\scripts\text_emotion_model.py "我有点难过，感觉撑不住了" --compact
```

指定模型和设备：

```powershell
conda run -n sensevoice python project\scripts\text_emotion_model.py "我现在很担心任务完成不了" --model "project\models\Chinese-Emotion-Small" --device cuda:0
```

## SenseVoice 测试

```powershell
conda activate sensevoice; python "project\scripts\sensevoice_asr_ser.py" --audio "project\samples\2tts_16k.wav" --device cuda:0 --disable-update
```

## TTS 说明

Web Demo 默认调用：

```text
project/scripts/tts_qwen3_from_pipeline.py
```

仓库内提供一个小体积验证音频：`project/samples/2tts_16k.wav`。TTS 参考音频建议使用 10-30 秒、单人、无背景噪声、文本准确的 16k wav。

固定参考音频启动示例：

```powershell
$env:WEB_TTS_REF_AUDIO="project\samples\2tts_16k.wav"; $env:WEB_TTS_REF_TEXT="清晨推开窗就能感受到微风拂面，世间万物都在安静慢慢生长，保持平和的心态，认真过好当下的每一天，用心感受生活里每一份细碎的美好与温柔。"; conda activate sensevoice; python "project\scripts\web_demo_app.py"
```

## 输出文件

运行产生的文件默认保存在：

```text
project/outputs/web_demo/audio/
project/outputs/web_demo/json/
project/outputs/web_demo/sessions/
```
