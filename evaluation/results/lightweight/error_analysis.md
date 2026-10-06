# FindX Error Analysis

This file is generated from actual benchmark output. It deliberately includes regressions and misses rather than showing only successful queries.

## Important failing/difficult examples

### text-011 — `wireless earbuds with clear calls`

- Language: `en`; type: `semantic`
- First relevant rank — BM25: `1`; dense ANN: `1`; dense exact: `1`; hybrid: `1`; reranked: `1`
- Hybrid→reranked nDCG@10 delta: `-0.0442`
- Detected issue(s): reranker decreases per-query nDCG@10.

### text-006 — `comfortable shoes for long distance running`

- Language: `en`; type: `semantic`
- First relevant rank — BM25: `1`; dense ANN: `1`; dense exact: `1`; hybrid: `1`; reranked: `1`
- Hybrid→reranked nDCG@10 delta: `-0.0052`
- Detected issue(s): reranker decreases per-query nDCG@10.
- Manual annotation: This is intentionally paraphrastic. Dense retrieval should have an opportunity to help because the wording need not exactly match a title; if BM25 wins, inspect descriptive terms already present in the catalog.

### telugu-001 — `నల్ల రన్నింగ్ షూస్`

- Language: `te`; type: `semantic`
- First relevant rank — BM25: `None`; dense ANN: `1`; dense exact: `1`; hybrid: `1`; reranked: `1`
- Hybrid→reranked nDCG@10 delta: `+0.0000`
- Detected issue(s): dense ANN places a relevant result earlier than BM25; multilingual slice (te) misses labelled relevant items.
- Manual annotation: FindX has no Telugu rewrite dictionary. This query is retained specifically to measure how much the multilingual encoder and corpus can do without language-specific normalization.

### telugu-002 — `వైర్‌లెస్ ఇయర్‌బడ్స్`

- Language: `te`; type: `semantic`
- First relevant rank — BM25: `None`; dense ANN: `None`; dense exact: `None`; hybrid: `None`; reranked: `None`
- Hybrid→reranked nDCG@10 delta: `+0.0000`
- Detected issue(s): multilingual slice (te) misses labelled relevant items.
- Manual annotation: Telugu is an evaluation slice, not a guaranteed feature. A miss should be reported rather than hidden by aggregate English performance.

### telugu-003 — `నీలం కుర్తీ`

- Language: `te`; type: `semantic`
- First relevant rank — BM25: `None`; dense ANN: `None`; dense exact: `None`; hybrid: `None`; reranked: `None`
- Hybrid→reranked nDCG@10 delta: `+0.0000`
- Detected issue(s): multilingual slice (te) misses labelled relevant items.

### telugu-004 — `ల్యాప్‌టాప్ బ్యాక్‌ప్యాక్`

- Language: `te`; type: `semantic`
- First relevant rank — BM25: `None`; dense ANN: `None`; dense exact: `None`; hybrid: `None`; reranked: `None`
- Hybrid→reranked nDCG@10 delta: `+0.0000`
- Detected issue(s): multilingual slice (te) misses labelled relevant items.

### telugu-005 — `ఫిట్‌నెస్ స్మార్ట్‌వాచ్`

- Language: `te`; type: `semantic`
- First relevant rank — BM25: `None`; dense ANN: `None`; dense exact: `None`; hybrid: `None`; reranked: `None`
- Hybrid→reranked nDCG@10 delta: `+0.0000`
- Detected issue(s): multilingual slice (te) misses labelled relevant items.

### telugu-006 — `స్టీల్ వాటర్ బాటిల్`

- Language: `te`; type: `semantic`
- First relevant rank — BM25: `None`; dense ANN: `None`; dense exact: `None`; hybrid: `None`; reranked: `None`
- Hybrid→reranked nDCG@10 delta: `+0.0000`
- Detected issue(s): multilingual slice (te) misses labelled relevant items.

### text-020 — `fitness watch with heart rate`

- Language: `en`; type: `semantic`
- First relevant rank — BM25: `2`; dense ANN: `1`; dense exact: `1`; hybrid: `1`; reranked: `1`
- Hybrid→reranked nDCG@10 delta: `+0.0000`
- Detected issue(s): dense ANN places a relevant result earlier than BM25.

### text-002 — `3k ke andar black running shoe`

- Language: `hinglish`; type: `hinglish`
- First relevant rank — BM25: `3`; dense ANN: `2`; dense exact: `2`; hybrid: `2`; reranked: `2`
- Hybrid→reranked nDCG@10 delta: `+0.0629`
- Detected issue(s): dense ANN places a relevant result earlier than BM25.

### text-059 — `nik running shoe under 3000`

- Language: `en`; type: `typo`
- First relevant rank — BM25: `2`; dense ANN: `1`; dense exact: `1`; hybrid: `1`; reranked: `1`
- Hybrid→reranked nDCG@10 delta: `+0.0762`
- Detected issue(s): dense ANN places a relevant result earlier than BM25.

### text-024 — `study lamp black`

- Language: `en`; type: `attribute`
- First relevant rank — BM25: `2`; dense ANN: `1`; dense exact: `1`; hybrid: `2`; reranked: `1`
- Hybrid→reranked nDCG@10 delta: `+0.3691`
- Detected issue(s): dense ANN places a relevant result earlier than BM25.

### text-054 — `नीले वायरलेस ईयरबड्स`

- Language: `hi`; type: `hindi`
- First relevant rank — BM25: `1`; dense ANN: `6`; dense exact: `6`; hybrid: `4`; reranked: `1`
- Hybrid→reranked nDCG@10 delta: `+0.5693`
- Detected issue(s): BM25 places a relevant result earlier than dense ANN.

## Image retrieval

- `image-006` from `ELE-007` — Recall@5 `0.0000`; ranked `WAT-006, WAT-008, WAT-002, WAT-009, APP-012`.
- `image-010` from `WAT-001` — Recall@5 `0.0000`; ranked `WAT-010, WAT-005, WAT-003, WAT-004, WAT-007`.
- `image-011` from `HOM-001` — Recall@5 `0.0000`; ranked `HOM-007, APP-007, APP-011, SHO-010, SHO-009`.
- `image-012` from `HOM-005` — Recall@5 `0.0000`; ranked `HOM-003, HOM-009, BAG-006, SHO-013, HOM-010`.
- `image-007` from `BAG-001` — Recall@5 `0.3333`; ranked `BAG-005, BAG-007, BAG-008, BAG-003, BAG-010`.
