# Keyword vs. Semantic Retrieval — A Small, Fully Reproducible Benchmark

Does an embedding-based search engine actually beat keyword search on the kind of
small, curated knowledge base most teams really run? This repository answers that
question empirically: **100 short technical notes**, **36 natural-language queries**
in three categories (exact-term / paraphrase / ambiguous), three retrievers, and a
one-command evaluation harness. Every number in the results below (and in the
[paper](paper/research_paper.md) and its [LaTeX/PDF version](paper/research_paper.pdf))
is a measured output of `src/evaluate.py` — nothing is fabricated.

## Results summary

Overall (36 queries, 100-document corpus; binary relevance = grade ≥ 1):

| Retriever | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 0.528 | 0.250 | 0.200 | 0.111 | 0.287 | 0.389 | 0.505 | 0.556 | 0.591 | 0.516 |
| TF-IDF | 0.528 | 0.278 | 0.217 | 0.114 | 0.287 | 0.421 | 0.542 | 0.569 | 0.604 | 0.530 |
| Dense (bge-small) | **0.722** | **0.398** | **0.294** | **0.169** | **0.421** | **0.625** | **0.750** | **0.843** | **0.812** | **0.760** |

Per category (12 queries each) — the interaction is the whole story:

| Category | BM25 P@1 | TF-IDF P@1 | Dense P@1 | BM25 MRR@10 | Dense MRR@10 |
|---|---|---|---|---|---|
| lexical (exact terms) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| semantic (paraphrase) | 0.250 | 0.250 | **0.667** | 0.267 | **0.783** |
| ambiguous (multi-hop) | 0.333 | 0.333 | **0.500** | 0.505 | **0.654** |

Query latency (mean / p50 / p95 ms, single CPU, warm cache): BM25 0.20 / 0.21 / 0.25,
TF-IDF 0.35 / 0.35 / 0.42, Dense 8.49 / 6.30 / 32.90.

Head-to-head rank of the first relevant document (dense vs. BM25): dense wins 16,
ties 17, loses 3 — and the 3 losses are all queries whose wording echoes the
answer document verbatim (see [results/error_cases.md](results/error_cases.md)).

## Quickstart

```bash
pip install -r requirements.txt
python src/evaluate.py          # runs the full benchmark, ~30 s incl. model download
```

Outputs (all written to `results/`):

- `metrics.json` — overall + per-category metrics, latency, experiment metadata
- `per_query_ranks.json` — per-query rank of every relevant document, per retriever
- `summary_tables.md` — generated markdown tables (source of truth for docs)

The dense retriever uses `BAAI/bge-small-en-v1.5` (384-dim) through
[fastembed](https://github.com/qdrant/fastembed) — deterministic local ONNX
inference, no network and no paid API at query time (first run downloads the
~35 MB model once). If fastembed is unavailable, the harness automatically falls
back to a deterministic LSA variant (TF-IDF → TruncatedSVD, seed 42) so the
benchmark always runs; the backend actually used is recorded in
`results/metrics.json` under `experiment.dense_backend`.

## Repository map

```
├── README.md                  <- you are here
├── requirements.txt
├── Makefile                   <- make eval / make paper
├── LICENSE                    <- MIT
├── data/
│   ├── corpus.jsonl           <- 100 notes (id, title, text, tags)
│   └── queries.jsonl          <- 36 queries (id, query, category, graded qrels)
├── src/
│   ├── retrievers.py          <- BM25 / TF-IDF / dense (+ deterministic fallback)
│   └── evaluate.py            <- P@k, R@k, MRR@10, nDCG@10; per-category; latency
├── results/
│   ├── metrics.json           <- real run output
│   ├── per_query_ranks.json   <- real run output
│   ├── summary_tables.md      <- real run output
│   └── error_cases.md         <- qualitative "lexical wins / semantic wins" analysis
├── paper/
│   ├── research_paper.md      <- full research paper (markdown)
│   ├── research_paper.tex     <- LaTeX source (compiles with tectonic)
│   └── cover.html             <- designed cover page source (rendered via Playwright)
└── substack/
    └── essay.md               <- narrative write-up for a general technical audience
```

## Dataset design (why these 100 notes)

The corpus is a fictional developer-cloud helpdesk ("Nimbus DevCloud"), 10 topic
clusters × 10 notes, deliberately mixing:

- **precise-terminology notes** — error codes, header names, CLI commands
  (`ERR_GATEWAY_TIMEOUT`, `X-RateLimit-Remaining`, `nimbus db migrate`), and
- **concept notes** — ideas whose canonical vocabulary a naive question never
  uses (metric *cardinality*, *read-your-writes*, *expand-contract migrations*).

Queries come in three 12-query categories: **lexical** (contains the answer's
exact identifier), **semantic** (paraphrases the answer with no distinctive term
overlap), **ambiguous** (multi-hop or several plausible answers). Every query
carries graded qrels (2 = directly answers, 1 = useful supporting document):
74 pairs total (42 grade-2, 32 grade-1). Judgments are defensible from document
text alone.

Full methodology: [paper/research_paper.md](paper/research_paper.md).

## Metric definitions

Identical across code, README, and both paper versions:

- Binary relevance: a document counts as relevant iff grade ≥ 1.
- `Precision@k = |relevant ∩ top-k| / k`, `Recall@k = |relevant ∩ top-k| / |all relevant|`
- `MRR@10 = 1/rank` of the first relevant document within top-10 (else 0)
- `nDCG@10` with graded gains `2^grade − 1`, logarithmic discount
- Macro-averaged over queries; per category = macro over that category's queries
- Ranking ties broken by corpus order (stable sort) → bit-reproducible runs

## Building the paper PDF

The compiled 9-page PDF ships with the project archive (this Git host keeps text
sources only; the CI workflow produces the PDF as a build artifact on every push
touching `paper/`). To rebuild the body text locally:

```bash
cd paper && tectonic research_paper.tex   # -> research_paper.pdf (8-page body)
```

The shipped PDF additionally has a designed dark cover page: `cover.html` is
rendered with Playwright (`html2poster.js`) and merged with the body via pypdf.

## Reproducibility notes

- Runs are deterministic: stable-sort tie-breaking, fixed seeds, deterministic
  ONNX inference. Re-running `src/evaluate.py` reproduces `metrics.json` exactly
  (latency numbers vary with hardware).
- Software versions used for the committed results: Python 3.12, rank-bm25 0.2.2,
  scikit-learn 1.5.2, numpy 2.1.3, fastembed 0.9.1 (ONNX runtime 1.32).

## License

MIT — see [LICENSE](LICENSE).
