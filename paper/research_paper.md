# Keyword Matching versus Dense Semantics: An Empirical Comparison of Lexical and Embedding-Based Retrieval on a Small Curated Knowledge Base

**Author:** Maissara Guersas
**Date:** November 2026
**Code and data:** [github.com/maissaraguersas-dev/lexical-vs-semantic-retrieval](https://github.com/maissaraguersas-dev/lexical-vs-semantic-retrieval)

---

## Abstract

Keyword (lexical) retrieval and dense embedding retrieval fail in complementary ways, but practitioners rarely see both failure modes quantified on the same small corpus. We present a controlled comparison of three retrievers - Okapi BM25, TF-IDF with cosine similarity, and dense retrieval with the `BAAI/bge-small-en-v1.5` sentence-embedding model - on a purpose-built knowledge base of 100 short technical notes from a fictional developer-cloud helpdesk, evaluated with 36 natural-language queries in three categories: exact-term (lexical), paraphrase (semantic), and ambiguous/multi-hop. All experiments are fully reproducible from a public repository; every number in this paper is a measured output of the evaluation harness. Results show a sharp category interaction: on exact-term queries all three retrievers achieve a perfect Precision@1 of 1.000, while on paraphrase queries lexical retrieval collapses (BM25 P@1 = 0.250, MRR@10 = 0.267) and dense retrieval dominates (P@1 = 0.667, MRR@10 = 0.783). Overall, dense retrieval wins 16 of 36 queries against BM25 on first-relevant-document rank, loses 3, and ties 17, finishing with nDCG@10 of 0.760 versus 0.516 (BM25) and 0.530 (TF-IDF) - at the cost of a query latency of 8.49 ms versus 0.20 ms on a single CPU core. Qualitative error analysis shows the three BM25 wins are all cases where verbatim phrasing echoes or exact identifiers anchor the answer, while dense failures concentrate on queries whose wording drifts topic-general. We conclude with practical guidance: diagnose the query distribution first, treat dense retrieval as the better default for heterogeneous traffic, and consider reciprocal-rank fusion to hedge both failure modes.

**Keywords:** information retrieval, BM25, TF-IDF, dense retrieval, sentence embeddings, semantic search, evaluation, precision@k, recall@k, nDCG, error analysis

---

## 1. Introduction

Every search system embodies a bet about how questions relate to answers. Lexical retrieval systems - from classical TF-IDF vector-space models to modern probabilistic rankers such as BM25 - bet that the words a person types will overlap with the words in the document that answers them. Dense retrieval systems bet differently: they embed queries and documents into a shared vector space where semantic proximity, not surface overlap, determines ranking. Neither bet is universally right, and the literature that compares them tends to test on web-scale corpora with millions of passages, where the trade-offs are smoothed by volume.

Yet a very large fraction of real search problems look nothing like the web. Internal knowledge bases, product FAQs, support wikis, and documentation sites commonly hold tens to thousands of short documents, written by one team, in one voice, with consistent terminology. On corpora of this size, the deployment decision between a lexical retriever and an embedding-based retriever is genuinely open: both are fast, both are trivially deployable, and both have failure modes that a small corpus will not average away. The question "which one should we ship?" deserves an answer measured on data shaped like the actual deployment, not extrapolated from MS MARCO.

This paper provides such a measurement. We built a 100-document knowledge base in a single technical domain (a fictional developer-cloud platform helpdesk), deliberately mixing documents that state facts with precise identifiers (error codes, header names, CLI commands) and documents that explain concepts in prose (idempotency, cardinality, blue-green deployments). We wrote 36 information needs as natural-language queries and grouped them into three categories designed to stress each bet: **lexical** queries containing the exact identifiers their answers use, **semantic** queries that paraphrase their answers with no distinctive term overlap, and **ambiguous** queries that legitimately match several documents or require combining two notes. We then ran three retrievers - BM25, TF-IDF, and dense embeddings from `BAAI/bge-small-en-v1.5` - over the corpus and evaluated Precision@k, Recall@k, MRR@10, and nDCG@10 for k in {1, 3, 5, 10}, overall and per category, plus end-to-end query latency.

Three findings summarize the contribution. First, the category interaction is dramatic: lexical retrieval is *perfect* on identifier queries (P@1 = 1.000 across all three retrievers) and *broken* on paraphrase queries (BM25 P@1 = 0.250; dense 0.667). Second, on this corpus dense retrieval is the dominant overall choice (nDCG@10 = 0.760 vs. 0.516/0.530), but it is not invincible: it loses 3 of 36 head-to-head first-relevant-rank comparisons to BM25, all three involving verbatim phrasing echoes between question and answer. Third, cost asymmetry is real but small at this scale: dense queries take ~8.5 ms end-to-end versus ~0.2 ms for BM25 - irrelevant for human-facing search on small corpora, relevant at fleet scale. All code, data, and per-query outputs are released for reproduction.

The rest of the paper proceeds as follows. Section 2 situates the work in the retrieval literature. Section 3 specifies the dataset and experimental setup. Section 4 reports measured results. Section 5 analyzes errors qualitatively. Section 6 discusses limitations and future work, and Section 7 concludes.

## 2. Background and Related Work

**Lexical retrieval.** The vector-space model represents queries and documents as weighted term vectors and ranks by cosine similarity; TF-IDF weighting - term frequency damped by document frequency - is its classical instantiation (Salton, Wong & Yang, 1975). BM25 reframes retrieval probabilistically as estimating relevance from query-term occurrences, with two tunable saturation parameters: term-frequency saturation k1 and document-length normalization b (Robertson & Zaragoza, 2009). BM25 remains the strongest known baseline without learned parameters and is the default first ranker in most production search stacks (Manning, Raghavan & Schütze, 2008). Both approaches share the same fundamental limitation: they can only match surface forms. If a user asks about a "server refusing requests because it fires them too fast" and the answer document says "rate limit," no weighting scheme can bridge the gap - the vocabulary mismatch problem has been documented since the earliest retrieval experiments.

**Dense retrieval.** Latent semantic analysis was the first attempt to overcome vocabulary mismatch by folding term-document matrices into low-rank latent spaces (Deerwester et al., 1990). Modern dense retrieval descends instead from neural language representations: word embeddings (Mikolov et al., 2013; Pennington, Socher & Manning, 2014) made semantics geometric, Sentence-BERT made sentence-level embedding practical at scale (Reimers & Gurevych, 2019), and dense passage retrieval demonstrated that dual-encoder similarity could rival BM25 on open-domain QA (Karpukhin et al., 2020). The BGE family of embedding models used in this paper was trained with contrastive learning on large paired corpora and is a widely used open-weight choice for semantic search (Xiao et al., 2023).

**Benchmarks and the scale gap.** Public benchmarks shaped the field's intuition about lexical-vs-dense trade-offs. MS MARCO (Bajaj et al., 2016) and Natural Questions (Kwiatkowski et al., 2019) drove the dense-retrieval revolution at web scale; BEIR (Thakur et al., 2021) then showed, across 18 heterogeneous tasks, that zero-shot dense retrievers often *lose* to BM25 out of domain - a calibration the field badly needed. What the public benchmarks under-represent is the small, homogeneous, curated corpus: 10^2-10^4 documents, one domain, one authorial voice, queries written by the same community that reads the answers. BEIR's smallest corpus has 19K documents; internal knowledge bases are two to three orders of magnitude smaller, where lexical statistics like document frequency are noisy and embedding-space neighborhoods are dense. Our study is deliberately scoped to that regime: not to claim generalizable leaderboard numbers, but to measure the two failure modes on a corpus shaped like the deployments practitioners actually make.

**Evaluation methodology.** We adopt the standard Cranfield-style evaluation: a fixed corpus, a fixed query set with graded relevance judgments (qrels), and rank-sensitive metrics. Precision@k and Recall@k characterize set overlap at fixed depth; MRR@10 captures how high the first relevant document appears; nDCG@10 uses graded gains to reward ordering quality (Manning et al., 2008). All four are macro-averaged over queries, per category and overall.

## 3. Dataset and Experimental Setup

### 3.1 Corpus

We constructed a corpus of 100 short notes for the fictional **Nimbus DevCloud** developer platform, written as a helpdesk knowledge base. Each note is 2-3 sentences (mean 29.6 tokens, range 20-43, after lowercasing and tokenizing on `[a-z0-9_]+`). Notes are organized in ten clusters of ten: authentication and API keys; rate limits and quotas; errors and troubleshooting; webhooks and events; SDKs and clients; databases and storage; deployments and builds; observability; networking; billing and plans.

The design is intentional in two ways. First, **terminology duality**: roughly half the notes state facts through precise identifiers (`ERR_GATEWAY_TIMEOUT`, `X-RateLimit-Remaining`, `Idempotency-Key`, `nimbus db migrate`), while the other half explain concepts whose canonical terms appear nowhere in a naive user's question (metric *cardinality*, *read-your-writes* consistency, *expand-contract* migrations). Second, **cluster adjacency**: within each cluster, documents share topic vocabulary (ten auth notes all mention keys or tokens), which punishes retrievers that match topic words instead of the specific note - a realistic difficulty for small homogeneous corpora. Each note carries a stable ID (`D001`-`D100`), a title, body text, and tags. For indexing we use title + body text concatenated.

### 3.2 Query set and relevance judgments

We wrote 36 queries (mean 11.6 tokens, range 5-20) as authentic-seeming information needs, 12 per category:

- **Lexical (12):** the query contains the distinctive surface form its answer uses - an error code ("How do I debug a 504 ERR_GATEWAY_TIMEOUT?"), a header name ("How do I interpret X-RateLimit-Remaining?"), or a CLI command ("How do I run a schema migration with nimbus db migrate?").
- **Semantic (12):** the query paraphrases the answer with no distinctive term overlap - e.g., "My script fires requests too fast and the server keeps refusing them" (answer: the rate-limit notes), "Are webhook notifications guaranteed to arrive exactly once?" (answer: the at-least-once delivery note), "Our metrics bill exploded after we started labeling every series with the customer id" (answer: the metric-cardinality note).
- **Ambiguous (12):** boundary or multi-hop needs that legitimately match several documents or require combining two notes - e.g., "How should I make my retry logic safe?" (retries + idempotency + 429 backoff), "The deploy went green but users are seeing errors. What do I check?" (health checks + rollback + triage), "If a bug triples our usage overnight, what happens to the bill and to the service?" (overage policy + spend alerts + metering).

Every query carries graded relevance judgments (qrels): grade 2 = highly relevant (a document that directly answers the need), grade 1 = partially relevant (useful supporting or adjacent document). The set contains 74 query-document pairs (42 grade-2, 32 grade-1); 6 queries have one relevant document, 22 have two, and 8 have three (mean 2.06). Category pair counts: lexical 22, semantic 21, ambiguous 31. Judgments were authored by the corpus creator with the deliberate bias that a judgment must be defensible from the document text alone - a grade-2 document answers the question; a grade-1 document is the one a support engineer would also hand you.

### 3.3 Retrievers

All retrievers share the interface `search(query, k) -> ranked document IDs`, over the same 100 documents, with ties broken deterministically by corpus order (stable sort), so every run is bit-reproducible.

1. **BM25** (`bm25`): Okapi BM25 from the `rank-bm25` 0.2.2 library, default parameters (k1 = 1.5, b = 0.75), tokenized with the shared lowercase `[a-z0-9_]+` pattern that keeps identifiers (`err_gateway_timeout`) and numbers (`429`) intact.
2. **TF-IDF** (`tfidf`): scikit-learn 1.5.2 `TfidfVectorizer` (lowercased, default word tokenization, L2-normalized vectors); cosine similarity is the dot product of normalized vectors.
3. **Dense** (`dense`): `BAAI/bge-small-en-v1.5` (384 dimensions) via `fastembed` 0.9.1 - local ONNX inference, no network access and no paid API at query time. Documents are embedded as passages; queries use the model's dedicated query-encoding path (the BGE query prefix). Similarity is cosine over L2-normalized vectors. If the embedding library or model weights are unavailable, the harness automatically falls back to a deterministic scikit-learn LSA variant (TF-IDF reduced by TruncatedSVD, random seed 42) so the benchmark always runs; the backend actually used is recorded in `results/metrics.json`.

### 3.4 Metrics and protocol

For each query we rank the full corpus and score depth k in {1, 3, 5, 10}. Binary metrics treat a document as relevant iff grade >= 1: **Precision@k** = |relevant in top-k| / k; **Recall@k** = |relevant in top-k| / |all relevant|; **MRR@10** = 1 / rank of the first relevant document within the top 10 (0 if none). **nDCG@10** uses graded gains 2^grade - 1 with logarithmic discount, normalized by the ideal ordering. Metrics are macro-averaged over all 36 queries (overall) and within each category. Latency is measured end-to-end per query (including embedding for dense) on a single CPU, after a warm-up call, reporting mean, median (p50), and p95 over the 36 queries. The full protocol is one command: `python src/evaluate.py`.

## 4. Results

### 4.1 Overall metrics

Table 1 reports overall metrics. Dense retrieval dominates every metric at every depth: +0.194 Precision@1, +0.245 Recall@5, +0.222 MRR@10, and +0.244 nDCG@10 over BM25. TF-IDF is a slightly better lexical baseline than BM25 on this corpus (MRR@10 0.604 vs. 0.591; nDCG@10 0.530 vs. 0.516) - a reminder that BM25's hyperparameters are not automatically optimal on very short documents - but the difference is second-order next to the lexical-vs-dense gap.

**Table 1. Overall results (36 queries, 100-document corpus).** Bold marks the best value per column.

| Retriever | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 0.528 | 0.250 | 0.200 | 0.111 | 0.287 | 0.389 | 0.505 | 0.556 | 0.591 | 0.516 |
| TF-IDF | 0.528 | 0.278 | 0.217 | 0.114 | 0.287 | 0.421 | 0.542 | 0.569 | 0.604 | 0.530 |
| Dense (bge-small) | **0.722** | **0.398** | **0.294** | **0.169** | **0.421** | **0.625** | **0.750** | **0.843** | **0.812** | **0.760** |

Head-to-head on first-relevant-document rank, dense wins 16 queries, BM25 wins 3, and 17 tie. Dense placed the first relevant document in the top 3 for 32 of 36 queries; BM25 managed 21.

### 4.2 Per-category breakdown

Table 2 decomposes the results by query category, where the design of Section 3.2 pays off.

**Table 2. Per-category results (12 queries per category).**

| Category | Retriever | P@1 | P@3 | P@5 | P@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| lexical | BM25 | 1.000 | 0.472 | 0.300 | 0.158 | 1.000 | **0.927** |
| lexical | TF-IDF | 1.000 | 0.472 | 0.317 | 0.158 | 1.000 | **0.933** |
| lexical | Dense | 1.000 | 0.500 | 0.333 | 0.183 | 1.000 | 0.917 |
| semantic | BM25 | 0.250 | 0.111 | 0.083 | 0.050 | 0.267 | 0.234 |
| semantic | TF-IDF | 0.250 | 0.111 | 0.100 | 0.058 | 0.297 | 0.255 |
| semantic | Dense | **0.667** | **0.306** | **0.233** | **0.125** | **0.783** | **0.733** |
| ambiguous | BM25 | 0.333 | 0.167 | 0.217 | 0.125 | 0.505 | 0.388 |
| ambiguous | TF-IDF | 0.333 | 0.250 | 0.233 | 0.125 | 0.516 | 0.402 |
| ambiguous | Dense | **0.500** | **0.389** | **0.317** | **0.200** | **0.654** | **0.631** |

The lexical category is the surprise worth dwelling on: **all three retrievers achieve P@1 = 1.000 and MRR@10 = 1.000.** With a distinctive identifier in the query, even the embedding model finds the right note first; modern contrastively trained encoders evidently do not lose exact-match ability on a 100-document corpus. The residual differences show up in ordering depth: TF-IDF edges out dense on graded ordering (nDCG@10 0.933 vs. 0.917), and BM25/TF-IDF rank the second relevant document more tightly on several queries (e.g., Q11's auto-disable/retry pair). When the answer is anchored by an exact form, term matching remains the cleanest instrument.

The semantic category is the mirror image, and the gap is not incremental - it is structural. BM25 and TF-IDF place the first relevant document first for only 3 of 12 paraphrase queries (P@1 = 0.250) and rarely recover deeper (P@10 = 0.050-0.058, R@10 = 0.292-0.333); dense answers 8 of 12 at rank 1 (P@1 = 0.667) and lifts Recall@10 to 0.750. No parameter tuning of a term-overlap ranker can fix this; the terms simply do not overlap.

The ambiguous category sits between the extremes and shows the most mixed behavior: dense wins on average (nDCG@10 0.631 vs. 0.388/0.402) but cedes exactly the three queries BM25 wins overall (Section 5.2). Ambiguous, multi-hop needs are where retrieval choice changes individual outcomes, not just averages.

### 4.3 Latency

**Table 3. End-to-end query latency (36 queries, single CPU, warm cache; dense includes query embedding).**

| Retriever | Mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|
| BM25 | 0.20 | 0.21 | 0.25 |
| TF-IDF | 0.35 | 0.35 | 0.42 |
| Dense (bge-small) | 8.49 | 6.30 | 32.90 |

Dense retrieval is ~30x slower in the median case (~43x by mean), entirely due to the fixed cost of one forward pass through the embedding model (~6 ms for a single query; scoring 100 precomputed vectors is negligible). At this corpus scale every option is instantaneous by human standards; the latency asymmetry only becomes an architecture decision at high query rates or in tight in-process budgets. The asymmetry that does matter at any scale is the *indexing* asymmetry: adding one note to the corpus costs a microsecond for BM25 and one embedding inference (~10 ms) for dense.

## 5. Discussion and Error Analysis

### 5.1 The two failure modes, by name

The per-category tables operationalize two classical failure modes. **Vocabulary mismatch** (user words ≠ document words) is what breaks lexical retrieval on semantic queries: BM25 ranks the metric-cardinality note 32nd for "our metrics bill exploded after we started labeling every series with the customer id," because the note says *cardinality* and *storage cost*, never *bill* or *exploded*. **Vocabulary over-specificity** - the mirror failure - is what costs dense retrieval its three losses: when the query echoes the document's exact phrasing ("credentials are valid but not allowed"), the embedding model reads the query as a *topic* ("authorization problem") and spreads rank over the whole auth cluster, while BM25 locks onto the verbatim match.

### 5.2 Where dense loses: the three BM25 wins

All three dense losses involve ambiguous queries:

1. **Q30 - "The API says I'm not allowed, but my token is definitely valid. Why?"** The 401-vs-403 note is BM25's rank 1 (the query echoes its "valid but not allowed" wording); dense places it 8th and demotes the scopes note from BM25's 19th to 48th, preferring a diffuse set of auth-cluster notes.
2. **Q26 - "The deploy went green but users are seeing errors. What do I check?"** BM25's exact matches on deploy/error tokens put the rollback note at rank 1; dense's first relevant document appears at rank 3. Notably, dense finds the health-checks note (BM25: rank 49) at rank 5 - it compensates differently rather than strictly worse across all relevant documents.
3. **Q27 - "Which credentials should a background worker running on a server use?"** The service-tokens note contains "server" and "worker" verbatim; BM25 rank 2 vs. dense rank 3.

The pattern is consistent: dense retrieval loses when question and answer share distinctive wording, because the embedding space organizes by *topic*, and within a ten-document topic cluster every document is close to every other. On a small homogeneous corpus, topic-level similarity is necessary but not sufficient.

### 5.3 Where dense wins: paraphrase as translation

The semantic category's dense wins are large and repeatable: rank 32 → 1 (metric cardinality), 20 → 1 (alert fatigue/paging), 21 → 1 (leaked tokens), 13 → 2 (point-in-time recovery), 17 → 2 (rate limiting), 28 → 5 (backup durability). In each case the question is phrased in operator language ("bad migration," "exact second," "stale numbers") while the documentation uses canonical systems vocabulary ("PITR," "WAL replay," "replica lag"). The embedding model acts as a learned translation layer between those registers - precisely the capability lexical retrieval lacks by construction.

### 5.4 The tie mass and what hybridization would buy

Seventeen of 36 head-to-heads tie, and dense's median advantage within the semantic category (twelve rank positions on first relevant document) coexists with BM25's precise anchoring on identifier queries. This is the textbook case for **reciprocal rank fusion** (Cormack, Clarke & Buettcher, 2009): concatenating the two rankings with a small fusion constant would capture dense's paraphrase coverage and BM25's exact-form anchoring without any model training. We leave hybrid fusion as future work (Section 6) but note that the per-category tables let a practitioner predict from their own query mix whether the added complexity pays.

### 5.5 Reading P@1 alone would mislead

On the lexical category, all three systems tie at P@1 = MRR@10 = 1.000, yet nDCG@10 separates them by up to 0.016 - and that ordering difference is exactly what a user scanning the top 3-5 results experiences. Single-position metrics hide ordering quality; graded metrics expose it. Small-corpus evaluations should always report both.

## 6. Limitations and Future Work

**Scale.** One hundred documents and 36 queries cannot support significance testing with useful power; per-category deltas of 0.1-0.5 nDCG on 12 queries are directional, not inferential. The corpus is also small enough that embedding-space neighborhoods are unusually dense, which may flatter the dense retriever relative to deployments at 10^4-10^6 documents. Conversely, lexical document-frequency statistics are noisy at this scale, which may *disfavor* BM25. Both biases are inherent to the regime we study, and we report them openly rather than average them away.

**Authoring bias.** The corpus author, query author, and judgment author are the same person. The qrel discipline (grade 2 must be answerable from the text alone) mitigates but cannot eliminate this; an independent annotator study is the obvious strengthening step.

**One embedding model, one language, one domain.** We test a single 384-dimensional open-weight encoder in English on a single domain. Larger or domain-tuned encoders could shift the dense curve upward; weaker ones (or the LSA fallback) shift it toward lexical performance. The harness accepts any `fastembed`-supported model with a one-line change, so replication sweeps are cheap.

**No tuning, no reranking, no hybrid.** We ran BM25 at library defaults and TF-IDF at scikit-learn defaults; neither was tuned on the query set (tuning on 36 queries would overfit trivially). A cross-encoder reranker, query expansion, and reciprocal-rank fusion are all unimplemented by design - the study isolates the two base retrieval bets. Future work: (1) add RRF fusion and measure whether it dominates both parents per category; (2) sweep embedding models (e.g., 33M-335M parameter encoders) to plot quality vs. latency frontiers at this scale; (3) replicate on a second corpus with different authorship to break the single-annotator dependency; (4) scale the corpus 10x by paraphrase augmentation to locate the crossover where lexical statistics stabilize.

## 7. Conclusion

On a small curated technical corpus shaped like the knowledge bases most teams actually run, the keyword-vs-embeddings decision is a decision about *query distribution*, not about a leaderboard. When users search with the exact identifiers the documentation uses, lexical retrieval is perfect at rank 1 and free - BM25's P@1 = 1.000 on identifier queries is a result no embedding needs to beat. When users describe problems in their own words, lexical retrieval is structurally broken (P@1 = 0.250) and a small open-weight embedding model translates between the two registers, more than tripling Precision@1 (0.667) and nearly tripling MRR@10 (0.783). Overall, dense retrieval is the better default (nDCG@10 0.760 vs. 0.516), winning 16 of 36 head-to-heads and losing only 3 - all to verbatim phrasing echoes - while costing ~8.5 ms rather than ~0.2 ms per query. The practical recipe that falls out of the measurement: inventory whether your users type identifiers or paraphrases; default to dense retrieval for mixed traffic; keep a lexical index anyway, because it anchors exact forms cheaply and is the natural partner for fusion. All data, code, per-query outputs, and both paper formats are released to make every number in this paper reproducible with one command.

## References

1. Bajaj, P., Campos, D., Craswell, N., Deng, S., Gao, J., Liu, X., Majumder, R., McNamara, A., Mitra, B., Nguyen, T., Rosenberg, R., Song, X., Sheikh, J., Wang, S., Yang, Y., Zhou, C., & Zou, K. (2016). *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.* arXiv:1611.09268.
2. Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *Proceedings of SIGIR 2009*, 758-759.
3. Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by latent semantic analysis. *Journal of the American Society for Information Science*, 41(6), 391-407.
4. Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, D., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Kelcey, N., Chang, L., Lee, M., Toutanova, K., Lewis, M., & Nallapati, R. (2019). Natural Questions: A Benchmark for Question Answering Research. *Transactions of the Association for Computational Linguistics*, 7, 453-466.
5. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *Proceedings of EMNLP 2020*, 6769-6781.
6. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.
7. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *Advances in Neural Information Processing Systems 26*, 3111-3119.
8. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. *Proceedings of EMNLP 2014*, 1532-1543.
9. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP-IJCNLP 2019*, 3982-3992.
10. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.
11. Salton, G., Wong, A., & Yang, C. S. (1975). A Vector Space Model for Automatic Indexing of Documents. *Communications of the ACM*, 18(11), 613-620.
12. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *Proceedings of the NeurIPS 2021 Datasets and Benchmarks Track.*
13. Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged Resources To Advance General Chinese Embedding. *Proceedings of SIGIR 2024* / arXiv:2309.07597. (Introduces the BGE embedding family; `bge-small-en-v1.5` model card: BAAI, 2023.)
