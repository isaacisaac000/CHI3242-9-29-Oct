import csv
import json
from pathlib import Path

import numpy as np
from qhchina.analytics.stylometry import Stylometry
from sklearn.decomposition import PCA

AUTHOR_ORDER = ["鲁迅", "沈从文"]
COLLECTION_ORDER = ["呐喊", "彷徨", "故事新编", "沈从文"]
ZSCORE_LABEL = "mean_qhchina_zscore_corpus_standardized"

docs = json.loads(Path("corpus_tokens.json").read_text(encoding="utf-8"))
assert len(docs) == 43

# Group documents by author (dict input to qhchina), preserving collection order
corpus = {}
for author in AUTHOR_ORDER:
    author_docs = sorted(
        (d for d in docs if d["author"] == author),
        key=lambda d: COLLECTION_ORDER.index(d["collection"]),
    )
    corpus[author] = [d["tokens"] for d in author_docs]

model = Stylometry(n_features=100, distance="cosine")  # default transform='zscore'
model.fit_transform(corpus)

vectors = np.asarray(model.document_vectors)
ids = list(model.document_ids)
labels = list(model.document_labels)
features = list(model.features)

assert vectors.shape == (len(docs), 100), vectors.shape
assert len(ids) == len(docs) and len(labels) == len(docs)

# Flatten grouped order back to per-document metadata
flat_docs = []
for author in AUTHOR_ORDER:
    flat_docs.extend(sorted(
        (d for d in docs if d["author"] == author),
        key=lambda d: COLLECTION_ORDER.index(d["collection"]),
    ))

pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)
assert coords.shape[0] == len(docs), f"PCA points {coords.shape[0]} != stories {len(docs)}"
print(f"PCA points: {coords.shape[0]} == story files: {len(docs)}; "
      f"explained variance: {pca.explained_variance_ratio_.round(4)}")

# 1. Story-coordinate CSV
with open("story_coordinates.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["document_id", "author", "collection", "source_file",
                "token_count", "PC1", "PC2"])
    for i, d in enumerate(flat_docs):
        w.writerow([ids[i], d["author"], d["collection"], d["path"],
                    d["token_count"], f"{coords[i, 0]:.6f}", f"{coords[i, 1]:.6f}"])

# 2. Selected feature names
with open("selected_features.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["rank", "feature"])
    for rank, feat in enumerate(features, 1):
        w.writerow([rank, feat])

# 3. Long-format top-features CSV
feat_index = {feat: j for j, feat in enumerate(features)}
with open("collection_top_features.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["collection", "author", "rank", "feature", "count",
                "total_collection_tokens", "freq_per_10k", ZSCORE_LABEL])
    for collection in COLLECTION_ORDER:
        cdocs = [i for i, d in enumerate(flat_docs) if d["collection"] == collection]
        total_tokens = sum(flat_docs[i]["token_count"] for i in cdocs)
        rows = []
        for feat in features:
            j = feat_index[feat]
            count = sum(flat_docs[i]["tokens"].count(feat) for i in cdocs)
            mean_z = float(np.mean(vectors[cdocs, j]))
            rows.append((feat, count, mean_z))
        rows.sort(key=lambda r: (-r[1], r[0]))
        for rank, (feat, count, mean_z) in enumerate(rows, 1):
            w.writerow([collection, flat_docs[cdocs[0]]["author"], rank, feat,
                        count, total_tokens, f"{count / total_tokens * 10000:.4f}",
                        f"{mean_z:.6f}"])

print(f"features: {len(features)}; wrote story_coordinates.csv, "
      f"selected_features.csv, collection_top_features.csv")
print("top 10 features:", features[:10])
