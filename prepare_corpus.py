import json
import re
from pathlib import Path

import jieba

ROOT = Path("corpora/authors")
COLLECTIONS = [
    ("鲁迅", "呐喊"),
    ("鲁迅", "彷徨"),
    ("鲁迅", "故事新编"),
    ("沈从文", "沈从文"),  # folder is 沈从文小说集, label is 沈从文
]

FOLDER_FOR_LABEL = {"沈从文": "沈从文小说集"}

NON_WORD = re.compile(r"[^\w]+|_", re.UNICODE)

documents = []
problems = []

for author, collection in COLLECTIONS:
    folder = FOLDER_FOR_LABEL.get(collection, collection)
    directory = ROOT / author / folder
    if not directory.is_dir():
        problems.append(f"missing folder: {directory}")
        continue
    for path in sorted(directory.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            problems.append(f"unreadable: {path} ({exc})")
            continue
        char_count = len(text)
        cleaned = NON_WORD.sub(" ", text)
        tokens = [t for t in jieba.lcut(cleaned) if t and all(c.isalnum() for c in t)]
        if not tokens:
            problems.append(f"no tokens: {path}")
            continue
        documents.append({
            "collection": collection,
            "author": author,
            "path": str(Path("corpora") / path.relative_to("corpora")),
            "char_count": char_count,
            "token_count": len(tokens),
            "tokens": tokens,
        })

out = Path("corpus_tokens.json")
out.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")

for d in documents:
    print(f"{d['collection']:>4} | {d['author']} | {d['path']} | chars={d['char_count']} tokens={d['token_count']}")
print(f"\n{len(documents)} documents written to {out}")
print("problems:", problems if problems else "none")
