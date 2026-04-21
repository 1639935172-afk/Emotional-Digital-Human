from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ALLOWED_EMOTIONS = {"happy", "sad", "angry", "neutral"}

SYSTEM_PROMPT = (
    "/no_think 你是“数字人多模态情感交互”引擎中的对话模块。"
    "你必须只输出一行严格合法 JSON，不要 markdown，不要解释，不要多余字符。"
    '输出 schema: {"emotion":"happy|sad|angry|neutral","reply_text":"string"}。'
    "规则：emotion 只能四选一；reply_text 必须是自然中文且简短共情；"
    "angry 场景要安抚，不得拱火；当出现终止语义时优先 neutral 风格。"
)

TEST_CASES = [
    "我今天特别开心，项目终于跑通了！",
    "唉，感觉有点低落，不太想说话。",
    "我现在很生气，真的忍不了了。",
    "今天就那样吧，平平淡淡。",
    "行，我知道了，不用再说了。",
    "别说了，我想安静一下。",
    "谢谢你，我心情好多了。",
    "最近压力很大，晚上总睡不好。",
    "太棒了！我拿到 offer 了！",
    "烦死了，怎么总出 bug。",
    "没事了，就这样吧。",
    "你能不能快点，我很急。",
    "我有点难过，但还能撑住。",
    "哈哈，今天运气真不错。",
    "无所谓，先这样。",
    "我现在情绪很差，谁都别理我。",
    "你说得对，我冷静一点。",
    "这结果我不满意，有点火大。",
    "嗯嗯，收到。",
    "其实我挺高兴的，谢谢你一直帮我。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 20-case JSON regression for llama-server")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions", help="chat completions endpoint")
    parser.add_argument("--model", default="Qwen3-8B-Q4_K_M.gguf", help="request model field")
    parser.add_argument("--temperature", type=float, default=0.2, help="sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=128, help="max output tokens")
    parser.add_argument("--timeout", type=int, default=60, help="request timeout seconds")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "regression_20_report.json"),
        help="output report path",
    )
    return parser.parse_args()


def post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("未找到 JSON 对象")
        return json.loads(match.group(0))


def validate_output(obj: Dict[str, Any]) -> Tuple[bool, str]:
    emotion = str(obj.get("emotion", "")).strip().lower()
    reply_text = str(obj.get("reply_text", "")).strip()
    if emotion not in ALLOWED_EMOTIONS:
        return False, f"emotion 非法: {emotion!r}"
    if not reply_text:
        return False, "reply_text 为空"
    return True, "ok"


def main() -> int:
    args = parse_args()
    start = time.time()

    results: List[Dict[str, Any]] = []
    success = 0

    for idx, user_text in enumerate(TEST_CASES, start=1):
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        }
        item: Dict[str, Any] = {"id": idx, "input": user_text}
        try:
            resp = post_json(args.url, payload, args.timeout)
            raw = resp["choices"][0]["message"]["content"]
            item["raw_output"] = raw
            parsed = extract_json(raw)
            item["parsed"] = parsed
            ok, reason = validate_output(parsed)
            item["ok"] = ok
            item["reason"] = reason
            if ok:
                success += 1
        except HTTPError as e:
            item["ok"] = False
            item["reason"] = f"HTTPError {e.code}"
        except URLError as e:
            item["ok"] = False
            item["reason"] = f"URLError {e.reason}"
        except Exception as e:  # noqa: BLE001
            item["ok"] = False
            item["reason"] = f"{type(e).__name__}: {e}"
        results.append(item)
        print(f"[{idx:02d}/20] {'PASS' if item['ok'] else 'FAIL'} - {item['reason']}")

    elapsed = round(time.time() - start, 3)
    total = len(TEST_CASES)
    rate = round((success / total) * 100.0, 2)

    report = {
        "summary": {
            "total": total,
            "success": success,
            "failed": total - success,
            "json_parse_rate_percent": rate,
            "elapsed_seconds": elapsed,
            "endpoint": args.url,
            "model": args.model,
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== REGRESSION SUMMARY ===")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report saved: {output_path}")

    return 0 if success == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

