# Why Your Search Engine Can't Find "My Server Keeps Refusing Me"

### A small experiment in keyword matching versus embeddings — with real numbers, and one surprising place where the old-school approach wins.

*Companion repo: [lexical-vs-semantic-retrieval](https://github.com/maissaraguersas-dev/lexical-vs-semantic-retrieval) — every number below is reproducible with one command.*

---

Run this query against a technical knowledge base:

> *"Our metrics bill exploded after we started labeling every series with the customer id. Why?"*

The correct answer is a short note about **metric cardinality** — the idea that putting unbounded values like `user_id` on metric labels makes your monitoring bill explode. A support engineer would find it instantly. A keyword search engine ranks it **32nd out of 100 documents**. An embedding-based search engine ranks it **1st**.

Now flip the experiment. Run this one:

> *"The API says I'm not allowed, but my token is definitely valid. Why?"*

Here the keyword engine is **perfect** — it puts the exact right note (the one explaining 401 vs 403 errors) at rank 1, because the query echoes the document's own phrasing: *valid but not allowed*. The embedding engine — the fancy one — puts it **8th** and scatters the rest of its top-10 across the general auth cluster.

Two queries. Two opposite failures. That's the whole story of keyword-versus-semantic search in miniature, and I built a small reproducible experiment to measure exactly where each approach breaks. This post walks through what I found.

## The two religions of search

**Keyword search** (BM25, TF-IDF) is the 50-year-old workhorse running underneath almost every search box you've ever used. Its logic is blunt: break the query into words, find documents containing those words, rank by a weighted overlap that rewards rare words and penalizes verbose documents. It has no idea what words *mean*. "Rate limit," "throttled," and "server refusing requests" are three unrelated strings as far as it's concerned.

**Embedding search** is the newcomer that AI made fashionable. A neural network reads each document and each query and outputs a vector — a point in 384-dimensional space, in my experiment — arranged so that sentences with similar *meaning* land near each other. Search becomes geometry: embed the query, return the nearest document points. It doesn't care that the query says "bill exploded" and the document says "cardinality explosions inflate storage cost." If it learned from enough human text, those two sentences are neighbors.

Keyword search fails when **the words don't match**. Embedding search fails when **the topic matches but the specifics don't**. Neither failure shows up until you look for it.

## The experiment

To catch both failure modes on camera, I built a fake-but-realistic knowledge base: 100 short helpdesk notes for an imaginary developer cloud platform, written in one voice, organized into ten topics (auth, rate limits, errors, webhooks, SDKs, databases, deployments, observability, networking, billing). Half the notes state facts through exact identifiers — error codes like `ERR_GATEWAY_TIMEOUT`, header names like `X-RateLimit-Remaining` — and half explain concepts whose canonical vocabulary a normal human would never type ("read-your-writes consistency," "expand-contract migrations").

Then I wrote 36 queries a confused developer might actually ask, in three flavors:

- **12 lexical queries** — they contain the exact identifier their answer uses ("How do I debug a 504 `ERR_GATEWAY_TIMEOUT`?").
- **12 semantic queries** — they paraphrase their answer with essentially zero word overlap ("My script fires requests too fast and the server keeps refusing them").
- **12 ambiguous queries** — multi-hop, several plausibly correct documents ("The deploy went green but users are seeing errors. What do I check?").

Every query has hand-graded relevance judgments. I ran three retrievers: BM25, TF-IDF, and a small open-weight embedding model (`BAAI/bge-small-en-v1.5`, running locally via ONNX — no paid API). The full harness is one command in the repo; nothing here is hand-waved.

## The numbers

The headline metrics, averaged over all 36 queries:

| | P@1 | MRR@10 | nDCG@10 |
|---|---|---|---|
| BM25 | 0.528 | 0.591 | 0.516 |
| TF-IDF | 0.528 | 0.604 | 0.530 |
| Embeddings | **0.722** | **0.812** | **0.760** |

Dense retrieval wins overall, decisively. But the averages hide the interesting part. Split by query type:

| Category | BM25 P@1 | Embeddings P@1 |
|---|---|---|
| Lexical (exact terms) | 1.000 | 1.000 |
| Semantic (paraphrase) | 0.250 | 0.667 |
| Ambiguous | 0.333 | 0.500 |

Two things jumped out at me.

**First: on exact-term queries, the fancy model doesn't beat the 1970s math — it ties it.** All three retrievers scored a *perfect* 1.000 Precision@1 on lexical queries. When the query contains `ERR_GATEWAY_TIMEOUT`, everything finds the answer. I expected embeddings to at least match term matching; I did not quite expect modern encoders to be this good at exact matching on a small corpus. The residual differences show up in ordering depth (TF-IDF nudges ahead on graded nDCG, 0.933 vs 0.917), not in who wins rank 1.

**Second: on paraphrase queries, keyword search isn't slightly worse — it's broken.** BM25 put the first relevant document at rank 1 for only 3 of 12 paraphrase queries, and the misses weren't near-misses: most landed between rank 11 and rank 32. The embedding model answered 8 of 12 at rank 1. This isn't a tuning problem. When a user says "need the database back to the exact second before a bad migration" and the document says "point-in-time recovery with WAL replay," there is no word-overlap bridge to tune. Either your system understands the translation, or it doesn't.

## The three times the embeddings lost

Because averages lie, I checked every head-to-head. Per-query, comparing where each system placed the first relevant document: embeddings won 16, tied 17, and **lost exactly 3** — and all three losses are the same failure shape.

The 401-vs-403 query above is the purest case. The embedding space organizes documents by *topic*, and my corpus has ten auth documents that all live in the same neighborhood. A query that reads as "generic auth complaint" lands in that neighborhood, and within it, ranking is close to a coin flip. Meanwhile BM25 doesn't know topics — it just sees that the query's unusual phrase "valid but not allowed" appears verbatim in exactly one document. Boom, rank 1.

The lesson generalizes: **embeddings give you topical recall; exact overlap gives you surgical precision.** When a user's words echo the documentation's words, the dumb matcher is the sharp instrument. When they don't, the dumb matcher is blind. That's not a bug in either system — it's the design.

## What it costs, and what I'd actually ship

One practical wrinkle: latency. The embedding model needs a neural forward pass per query — about **8.5 ms** end-to-end on my CPU versus **0.2 ms** for BM25. At knowledge-base scale both are invisible to a human; the number only starts to matter when you're paying for it at millions of queries, and the *indexing* asymmetry matters more (adding one document to BM25's index is free; embedding it is another inference).

If I were shipping search for a small knowledge base tomorrow, the experiment suggests a boring but defensible recipe:

1. **Diagnose the query distribution first.** If your users type error codes and API names, you can ship BM25 and be done — it's literally perfect on those queries. If they describe problems in prose, keyword search will structurally fail a quarter or more of them regardless of tuning.
2. **Default to embeddings for mixed traffic.** On my corpus they win overall with no domain training whatsoever — just a 33-million-parameter open model.
3. **Keep the lexical index anyway.** It's nearly free, it anchors exact identifiers, and it's the natural ingredient for a hybrid: reciprocal rank fusion — literally just merging the two ranked lists with a small constant — is the standard way to get embeddings' paraphrase coverage *and* BM25's verbatim precision. (I left fusion out of the experiment on purpose; it's the obvious next run.)

## The bigger point

We talk about "AI search" as if it replaced the old kind. It didn't. It added a second, complementary kind, and the two fail in opposite directions. The most valuable thing a team can know is *which direction their users' questions push*. That's not a benchmark question — MS MARCO can't answer it — it's a question about your own query logs, and you can answer it with a weekend, 100 documents, and 36 queries.

All of it — corpus, queries, judgments, harness, per-query outputs, the LaTeX paper version of this analysis — is in the repo. Run `python src/evaluate.py` and watch your own two failures appear.

*Thanks for reading. If you build something with the harness, I'd love to hear what your query distribution looks like.*
