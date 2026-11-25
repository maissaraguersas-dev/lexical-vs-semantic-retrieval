"""Evaluate keyword (BM25, TF-IDF) vs. dense embedding retrieval.

Reads data/corpus.jsonl and data/queries.jsonl, runs every query through
every retriever, and writes:

  results/metrics.json        overall + per-category metrics and latency
  results/per_query_ranks.json  per-query ranks of every relevant document
  results/summary_tables.md   markdown tables used by README / paper

Metric definitions (identical across paper, README, and this code):
  * Binary relevance: a document counts as relevant iff grade >= 1.
  * Precision@k  = |relevant in top-k| / k
  * Recall@k     = |relevant in top-k| / |all relevant for the query|
  * MRR@10       = 1 / rank of the first relevant doc within top-10 (else 0)
  * nDCG@10      = DCG@10 / IDCG@10 with graded gains 2^grade - 1
  * Averages are macro averages over queries (per category: macro over that
    category's queries).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrievers import build_retrievers  # noqa: E402

K_VALUES = [1, 3, 5, 10]
CATEGORIES = ["lexical", "semantic", "ambiguous"]


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def precision_at_k(ranked: List[str], relevant: set, k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k


def recall_at_k(ranked: List[str], relevant: set, k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: List[str], relevant: set, k: int) -> float:
    for i, doc in enumerate(ranked[:k]):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: List[str], grades: Dict[str, int], k: int) -> float:
    dcg = sum((2 ** grades[d] - 1) / math.log2(i + 2)
              for i, d in enumerate(ranked[:k]) if d in grades)
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retriever(name: str, retriever, docs: List[dict], queries: List[dict]):
    """Run all queries; returns (metrics dict, per-query records, latencies)."""
    per_metric: Dict[str, List[float]] = {}
    per_cat_metric: Dict[str, Dict[str, List[float]]] = {
        c: {} for c in CATEGORIES}
    per_query: List[dict] = []
    latencies: List[float] = []

    retriever.search("warmup query for latency measurement", k=10)  # warm-up

    for q in queries:
        grades = {d: int(g) for d, g in q["relevant"].items()}
        relevant = {d for d, g in grades.items() if g >= 1}

        t0 = time.perf_counter()
        ranked = retriever.search(q["query"], k=len(docs))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        q_vals = {
            f"precision@{k}": precision_at_k(ranked, relevant, k)
            for k in K_VALUES
        }
        q_vals.update({
            f"recall@{k}": recall_at_k(ranked, relevant, k) for k in K_VALUES
        })
        q_vals["mrr@10"] = mrr_at_k(ranked, relevant, 10)
        q_vals["ndcg@10"] = ndcg_at_k(ranked, grades, 10)

        for key, val in q_vals.items():
            per_metric.setdefault(key, []).append(val)
            bucket = per_cat_metric[q["category"]]
            bucket.setdefault(key, []).append(val)

        doc_ranks = {
            d: (ranked.index(d) + 1 if d in ranked else None) for d in grades
        }
        per_query.append({
            "id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "first_relevant_rank": min(
                (r for r in doc_ranks.values() if r is not None), default=None),
            "relevant_doc_ranks": doc_ranks,
        })

    metrics = {key: statistics.mean(vals) for key, vals in per_metric.items()}
    by_category = {
        cat: {key: statistics.mean(vals) for key, vals in bucket.items()}
        for cat, bucket in per_cat_metric.items()
    }
    latency = {
        "mean_ms": statistics.mean(latencies),
        "p50_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
    }
    return {"metrics": metrics, "by_category": by_category,
            "per_query": per_query, "latency": latency}


def fmt_table(rows: List[List[str]], header: List[str]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=os.path.join(here, "data", "corpus.jsonl"))
    ap.add_argument("--queries", default=os.path.join(here, "data", "queries.jsonl"))
    ap.add_argument("--outdir", default=os.path.join(here, "results"))
    ap.add_argument("--skip-dense", action="store_true",
                    help="skip dense retriever (offline smoke test)")
    args = ap.parse_args()

    docs = load_jsonl(args.corpus)
    queries = load_jsonl(args.queries)
    doc_ids = [d["id"] for d in docs]
    documents = [d["title"] + ". " + d["text"] for d in docs]

    retrievers = build_retrievers(doc_ids, documents, skip_dense=args.skip_dense)
    dense_backend = getattr(retrievers.get("dense"), "backend", "skipped")

    results, per_query_all, latency_all = {}, {}, {}
    for name, retriever in retrievers.items():
        out = evaluate_retriever(name, retriever, docs, queries)
        results[name] = {"overall": out["metrics"],
                         "by_category": out["by_category"]}
        latency_all[name] = out["latency"]
        for rec in out["per_query"]:
            per_query_all.setdefault(rec["id"], {})["meta"] = {
                "category": rec["category"], "query": rec["query"]}
            per_query_all[rec["id"]][name] = {
                "first_relevant_rank": rec["first_relevant_rank"],
                "relevant_doc_ranks": rec["relevant_doc_ranks"],
            }
        print(f"[{name}] P@1={out['metrics']['precision@1']:.3f} "
              f"P@5={out['metrics']['precision@5']:.3f} "
              f"R@5={out['metrics']['recall@5']:.3f} "
              f"MRR@10={out['metrics']['mrr@10']:.3f} "
              f"nDCG@10={out['metrics']['ndcg@10']:.3f} "
              f"latency_mean={out['latency']['mean_ms']:.2f}ms")

    # Per-query winner summary: compare rank of the first relevant doc
    # (lower is better; missing counts as worst) between bm25 and dense.
    wins = {"bm25": 0, "dense": 0, "tie": 0}
    if "dense" in per_query_all.get("Q01", {}):
        def rank_key(entry):
            r = entry["first_relevant_rank"]
            return r if r is not None else len(docs) + 1
        for qid in sorted(per_query_all):
            b = per_query_all[qid]["bm25"]["first_relevant_rank"]
            d = per_query_all[qid]["dense"]["first_relevant_rank"]
            if b is None and d is None:
                wins["tie"] += 1
            elif rank_key(per_query_all[qid]["bm25"]) < rank_key(per_query_all[qid]["dense"]):
                wins["bm25"] += 1
            elif rank_key(per_query_all[qid]["bm25"]) > rank_key(per_query_all[qid]["dense"]):
                wins["dense"] += 1
            else:
                wins["tie"] += 1

    experiment = {
        "corpus_size": len(docs),
        "num_queries": len(queries),
        "num_queries_by_category": {
            c: sum(1 for q in queries if q["category"] == c) for c in CATEGORIES},
        "document_text": "title + text",
        "k_values": K_VALUES,
        "dense_backend": dense_backend,
        "rank_tie_breaking": "stable sort by corpus order",
        "seed": 42,
    }
    print("first-relevant-rank wins (bm25 vs dense):", wins)

    os.makedirs(args.outdir, exist_ok=True)
    metrics_out = {
        "experiment": experiment,
        "overall": {n: results[n]["overall"] for n in results},
        "by_category": {n: results[n]["by_category"] for n in results},
        "latency_ms": latency_all,
        "first_relevant_rank_wins": wins,
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.outdir, "per_query_ranks.json"), "w", encoding="utf-8") as f:
        json.dump(per_query_all, f, indent=2, ensure_ascii=False)

    # Markdown summary tables (source of truth for README + paper).
    metric_keys = ([f"precision@{k}" for k in K_VALUES]
                   + [f"recall@{k}" for k in K_VALUES] + ["mrr@10", "ndcg@10"])
    rows = []
    for name in results:
        m = results[name]["overall"]
        rows.append([name] + [f"{m[key]:.3f}" for key in metric_keys])
    overall_md = fmt_table(rows, ["retriever"] + metric_keys)

    cat_rows = []
    for cat in CATEGORIES:
        for name in results:
            m = results[name]["by_category"][cat]
            cat_rows.append([cat, name] + [
                f"{m[f'precision@{k}']:.3f}" for k in K_VALUES]
                + [f"{m['mrr@10']:.3f}", f"{m['ndcg@10']:.3f}"])
    cat_md = fmt_table(
        cat_rows, ["category", "retriever", "P@1", "P@3", "P@5", "P@10",
                   "MRR@10", "nDCG@10"])

    lat_rows = [[name,
                 f"{latency_all[name]['mean_ms']:.2f}",
                 f"{latency_all[name]['p50_ms']:.2f}",
                 f"{latency_all[name]['p95_ms']:.2f}"]
                for name in results]
    lat_md = fmt_table(lat_rows, ["retriever", "mean ms", "p50 ms", "p95 ms"])

    summary = (
        "# Summary tables (generated by src/evaluate.py)\n\n"
        "Regenerate with: `python src/evaluate.py`\n\n"
        f"## Overall metrics (n={len(queries)} queries, corpus={len(docs)} docs)\n\n"
        f"{overall_md}\n\n"
        "## Per-category metrics\n\n"
        f"{cat_md}\n\n"
        "## Query latency (end-to-end, including embedding for dense)\n\n"
        f"{lat_md}\n"
    )
    with open(os.path.join(args.outdir, "summary_tables.md"), "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + summary)


if __name__ == "__main__":
    main()
