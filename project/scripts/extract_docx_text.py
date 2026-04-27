from __future__ import annotations

import argparse
import html
import re
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Word .docx to plain text (document.xml)")
    p.add_argument("--docx", required=True, help="Path to .docx file")
    p.add_argument("--out", required=True, help="Output txt path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    docx = Path(args.docx)
    out = Path(args.out)
    if not docx.exists():
        raise SystemExit(f"not found: {docx}")

    with zipfile.ZipFile(docx) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")

    # keep paragraph breaks
    xml = re.sub(r"</w:p>", "\n", xml)
    texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml, flags=re.DOTALL)
    text = "\n".join(html.unescape(t) for t in texts)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote: {out} (chars={len(text)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

