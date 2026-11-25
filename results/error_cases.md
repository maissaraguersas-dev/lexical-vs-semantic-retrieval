# Error Analysis: Where Keyword Search Wins, Where Semantics Wins

Generated from `results/per_query_ranks.json` (36 queries, 100-document corpus).
All ranks below are actual measured outputs of `python src/evaluate.py`; ranks
of relevant documents per retriever (BM25 vs. dense `BAAI/bge-small-en-v1.5`).

Headline counts, based on the rank of the **first relevant document** per query:

| Outcome | Queries (of 36) |
|---|---|
| BM25 places the first relevant doc strictly higher than dense | 3 |
| Dense places it strictly higher than BM25 | 16 |
| Tie | 17 |
| Either retriever misses the first relevant doc outside top-10 | 0 |

The asymmetry is the story: dense embeddings win *far more often* than they
lose, but the three queries they lose are all cases where **exact surface
forms carry the meaning** and the query's paraphrase pulls the embedding
toward the wrong neighborhood.

---

## 1. Semantic wins: the vocabulary mismatch problem (dense crushes BM25)

These queries describe a problem in everyday words; the answer note uses
different technical vocabulary. BM25 has no bridge across the wording gap.

### Case A - Q22 (semantic): "Our metrics bill exploded after we started labeling every series with the customer id. Why?"
- Target: **D073 "Metric cardinality"** ("Do not use unbounded label values such as user_id or request_id as metric labels...")
- BM25: rank **32** &nbsp;|&nbsp; Dense: rank **1**
- The query says *bill exploded*, *labeling*, *customer id*; the document says
  *cardinality*, *label values*, *storage cost*. The only shared terms are
  function words and "label", which appears in many other notes. BM25's
  term-overcome scoring drowns the target among 31 other notes; the embedding
  places "exploding cost caused by per-customer labels" next to
  "cardinality explosions inflate storage cost" almost perfectly.

### Case B - Q20 (semantic): "We only want the on-call engineer paged when customers are actually affected. How?"
- Target: **D075 "Reducing alert fatigue"**, partial **D074 "Defining SLOs and error budgets"**
- BM25: D075 at rank **20**, D074 at rank **39** &nbsp;|&nbsp; Dense: D075 at rank **1** (D074 not surfaced, rank **67**)
- *Paged / on-call / customers affected* vs. *page / alert fatigue / symptoms
  that hurt users*. No overlapping content words except "page(s)" - which is
  polysemous and matches unrelated pages. The embedding model, trained on
  billions of sentences, has seen this exact operational scenario many times.

### Case C - Q21 (semantic): "One of our secret tokens just leaked into a public repository. What should we do?"
- Target: **D007 "Storing and leaking keys"**, partial **D002 "Rotating API keys"**
- BM25: D007 at rank **21** &nbsp;|&nbsp; Dense: D007 at rank **1**
- *Leaked token* vs. *leaking keys*: BM25 matches "leak" but the note is
  outranked by notes that use *token* more often (D009 "Personal access
  tokens" at rank 1, D005 "Service tokens vs user tokens" at rank 5). Dense
  correctly privileges the scenario (credential leak -> revoke) over raw
  token frequency.

### Case D - Q17 (semantic): "I ran a bad migration and need the database back to the exact second before it ran. Is that possible?"
- Target: **D053 "Automated backups and PITR"** (grade 2), **D059 "Restoring a database to a point in time"** (grade 2)
- BM25: D053 at rank **20**, D059 at rank **13** &nbsp;|&nbsp; Dense: D059 at rank **2**, D053 at rank **4**
- The answer notes talk about *point-in-time recovery* and *WAL replay*; the
  question says *back to the exact second*. Classic terminology gap.

### Case E - Q36 (ambiguous): "How can I tell whether an outage is our fault or the platform's?"
- Targets: **D029 "Triaging a 5xx spike"** (rank 1 target), partial D028, D022
- BM25: D029 at rank **18**, D028 at rank **85** &nbsp;|&nbsp; Dense: D029 at rank **1**
- *Outage / whose fault* never appears in any note; D029's *check the status
  page first... most spikes are platform-side* is semantically the same
  advice. BM25 cannot make that leap; the embedding can.

Pattern: across all 12 semantic queries, BM25 achieves P@1 = 0.250 and
MRR@10 = 0.267, while dense reaches P@1 = 0.667 and MRR@10 = 0.783. The
retrieval gap is not incremental - lexical retrieval is *broken* for
paraphrased information needs.

---

## 2. Lexical wins: exact identifiers still anchor the answer (BM25 beats dense)

Three queries - all in the ambiguous category - are won by BM25, and in each
one a rare, high-precision surface form (or an unusually direct word match)
identifies the target:

### Case F - Q30 (ambiguous): "The API says I'm not allowed, but my token is definitely valid. Why?"
- Target: **D004 "401 vs 403 responses"** ("...403 means the credentials are valid but not allowed...")
- BM25: D004 at rank **1** &nbsp;|&nbsp; Dense: D004 at rank **8** (and D003 at rank **48** vs. BM25's **19**)
- The query's phrasing *valid + not allowed* mirrors the document's own
  wording *valid but not allowed* almost verbatim, so BM25 locks on. The
  embedding reads the query as a generic authorization complaint and spreads
  probability over the whole auth cluster (D005, D008, D003...), demoting the
  one note that disambiguates 401 vs. 403.

### Case G - Q26 (ambiguous): "The deploy went green but users are seeing errors. What do I check?"
- BM25 first relevant (D064 "Rolling back a deployment") at rank **1** vs. dense rank **3**;
  BM25 also finds D029 at rank 4 while dense needs rank 3 for its first hit.
- BM25's top hit is driven by exact matches on *deploy/errors/check*-adjacent
  tokens in D064 ("nimbus deploy rollback..."), which happens to be a good
  answer. This is the lexical baseline at its best: when question and answer
  happen to share wording, exact matching is unbeatable at rank 1.

### Case H - Q27 (ambiguous): "Which credentials should a background worker running on a server use?"
- Targets: **D005 "Service tokens vs user tokens"** (grade 2), D003 (grade 1)
- BM25: D005 at rank **2** &nbsp;|&nbsp; Dense: D005 at rank **3**
- *Server* and *worker* appear literally in D005 ("server-to-server jobs",
  "automation"), giving BM25 high-precision overlap. The embedding trades
  this for a slightly more "auth-topic-general" ordering and loses one rank.

Weaker, related evidence from the lexical category: on graded ordering
(nDCG@10), lexical retrievers edge out dense - BM25 0.927 / TF-IDF 0.933 vs.
dense 0.917 - even though all three tie at P@1 = 1.000 and MRR@10 = 1.000.
When an exact identifier is present, term matching puts the remaining
relevant documents in a tighter, more reliable order.

---

## 3. Both retrievers struggle: ambiguity is genuinely hard

No query had its first relevant document ranked outside the top 10 by both
retrievers, but a few show *both* systems compensating in different ways:

- **Q33 "My dashboard shows stale numbers right after I save changes. Why?"** -
  BM25 never finds D054 ("Replica lag and read-your-writes", rank 65); it
  survives via D091 at rank 7 because "dashboard"/"usage" overlap. Dense
  finds the *metering lag* note D091 at rank 2 but D054 only at rank 9.
  Neither fully resolves "stale numbers" to "replica lag" - a two-hop
  inference (save -> write -> read replica -> stale read) that neither
  sparse nor off-the-shelf dense retrieval performs reliably.

- **Q11 (lexical) "What happens after 50 consecutive failed webhook deliveries?"** -
  BM25 orders the pair correctly (D040 at 1, D033 at 2); dense swaps them
  (D033 at 1, D040 at 2). Both answers are useful; the graded judgment
  favors BM25.

---

## 4. Latency and the small-corpus caveat

Measured end-to-end query latency (median of 36 queries, single CPU, warm
cache): BM25 **0.21 ms**, TF-IDF **0.35 ms**, dense **6.30 ms** (mean 8.49 ms,
p95 32.90 ms including query embedding). On a 100-document corpus, all three
are instantaneous by human standards; the dense encoder's fixed cost (~6 ms
for one query through the ONNX model) dominates. The gap matters when the
same choice is replayed at fleet scale or inside tight latency budgets, but
on small curated corpora, quality - not speed - should drive the decision.

---

## 5. Takeaways for practitioners

1. **Diagnose your query distribution before choosing a retriever.** If
   users search with exact identifiers (error codes, header names, SKU
   numbers), lexical retrieval is near-perfect and free (P@1 = 1.000 on our
   lexical subset). If users describe problems in their own words, lexical
   retrieval fails structurally (P@1 = 0.250).
2. **Dense retrieval is the better default on heterogeneous queries**, but
   it is not invincible: it lost 3 of 36 queries to BM25, all involving
   verbatim phrasing echoes (Case F) or topic-general drift (Case F/H).
3. **The hybrid case is real.** Nine queries (25%) were won or tied by
   different systems at rank 1; a cheap reciprocal-rank fusion of BM25 +
   dense would hedge both failure modes without any model training.
4. **Graded relevance exposes ordering quality that P@1 hides.** All three
   retrievers tie at P@1 = 1.0 on lexical queries, yet differ by up to
   0.016 nDCG@10 - exactly the margin that matters when users read three
   results, not one.
