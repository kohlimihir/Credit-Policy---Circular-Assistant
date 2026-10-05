# Credit Policy & Circular Assistant: Technical Deep Dive

A complete explanation of every technical component in this project, written for someone
who knows traditional ML for credit risk (scorecards, logistic regression, GBMs, AUC/KS,
PSI) but is new to retrieval and language models.

Each section explains **what the thing is**, **why it is here**, **how this repo implements
it** (with file references), and **what an interviewer may push on**.

---

## 1. What the system actually does

**Input:** a natural-language regulatory question, e.g. *"What is the minimum CRAR
requirement for NBFC-ML?"*

**Output:** an answer grounded in RBI documents, with clause-level citations
(`[Document, Section, Page N]`), or an explicit refusal when the documents do not
support an answer.

**Corpus:** 15 RBI PDFs (NBFC directions, KYC, IRAC norms, microfinance, P2P, account
aggregator, priority sector lending, housing finance, governance, outsourcing), 19 to 108
pages each, downloaded automatically by `scripts/download_pdfs.py`.

The architecture pattern is **RAG (Retrieval-Augmented Generation)**.

### Why RAG and not a fine-tuned model

| Option | Problem |
|---|---|
| Ask a general LLM directly | It invents plausible-sounding rules. It cannot cite a page. It does not know the Nov 2025 RBI directions. |
| Fine-tune an LLM on the PDFs | Expensive, must be redone on every circular update, still cannot prove where an answer came from, still hallucinates. |
| **RAG** | Documents stay outside the model. Updating = re-ingesting a PDF. Every claim is traceable to a retrieved passage. |

In credit-risk language: RAG keeps the "policy book" as **data**, not as **model
weights**. That matters in a regulated setting because you need an audit trail, and
because the regulation changes faster than you would want to retrain.

---

## 2. Mental model: mapping to traditional credit-risk ML

This is the fastest way to get oriented. The analogies are not perfect, but they are close
enough to reason with.

| This project | Your familiar equivalent |
|---|---|
| Chunking a PDF | Defining the unit of observation (account-month vs application) |
| Embedding a chunk into 768 numbers | Feature engineering: turning raw text into a numeric vector |
| BM25 keyword search | A rules-based / WOE-binned lookup on exact tokens |
| Dense vector search | A learned representation capturing meaning, not just literal words |
| Hybrid retrieval (RRF) | Model ensembling, but combining **ranks** instead of probabilities |
| Cross-encoder reranker | Two-stage scoring: cheap model screens, expensive model decides |
| Abstention threshold | Score cut-off for auto-decline vs manual referral |
| Recall@5 / MRR | AUC / KS, but for ranking retrieved documents |
| Faithfulness score | Post-hoc model monitoring: is the output consistent with its inputs? |
| Gold evaluation set (120 questions) | Labelled out-of-time validation sample |
| Citation verification | Reason-code validation: does the stated reason actually hold? |

One genuine difference: there is **no training** in this pipeline. Every model
(embedder, reranker, LLM) is pre-trained and used as-is. The engineering is in
**retrieval design and evaluation**, not in fitting parameters.

---

## 3. End-to-end architecture

```
                        INGESTION (offline, run once per corpus update)
 PDFs ──▶ PDFParser ──▶ MetadataExtractor ──▶ Chunker (3 strategies) ──▶ Embedder
                                                                            │
                                            ┌───────────────────────────────┴───┐
                                            ▼                                   ▼
                                   Qdrant (dense vectors)              BM25 index (JSON)

                        QUERY TIME (online, ~3 seconds)
 Question
    │
    ├─▶ embed query ──▶ Qdrant top-50 (semantic)        ┐
    │                                                   ├─▶ RRF fusion ──▶ 50 candidates
    └─▶ tokenize ─────▶ BM25 top-50 (lexical)           ┘
                                                                  │
                                                                  ▼
                                                     Cross-encoder reranker ──▶ top-5
                                                                  │
                                                                  ▼
                                                        Abstention check
                                                         │              │
                                              below threshold      above threshold
                                                         │              │
                                                      refuse       Generator (LLM or extractive)
                                                                        │
                                                                        ▼
                                                             Citation verifier ──▶ response
```

Two phases is the key structural idea. Ingestion is slow and done rarely. Query time must
be fast, so all the heavy lifting (parsing, chunking, embedding) is pre-computed.

---

## 4. Stage 1: PDF parsing

**File:** `src/ingestion/pdf_parser.py`

### The problem
A PDF is a **layout** format, not a text format. It stores "draw this glyph at these
coordinates". There is no notion of a paragraph, a heading, or a reading order. RBI PDFs
add two-column layouts, bilingual headers, tables, and footers.

### Terms

- **PyMuPDF (imported as `fitz`)**: a fast C-based PDF library. `page.get_text("text")`
  extracts text with a reasonable guess at reading order. This is the primary extractor.
- **pdfplumber**: slower, pure-Python, but better at **tables** because it analyses ruling
  lines and character positions. Used here as a fallback and for table extraction.
- **Fallback trigger**: `MIN_TEXT_CHARS = 50`. If PyMuPDF returns under 50 characters for
  a page, the page is probably scanned or table-only, so pdfplumber retries it
  (`_fallback_pdfplumber`).
- **SHA-256 file hash** (`_compute_hash`): a fingerprint of the file bytes. Used for
  deduplication and provenance, so you can prove which exact file version produced a chunk.
  Analogous to hashing a data extract to prove lineage.
- **`ParsedPage` / `ParsedDocument`**: dataclasses holding per-page text, detected tables,
  `has_images`, and `char_count`. Keeping text **per page** (not one big string) is what
  makes page-level citations possible later.

### Interview angle
*"What breaks here?"* Scanned pages with no text layer. There is no OCR in this pipeline,
so an image-only page yields nothing. Adding Tesseract or a vision model would be the fix.
Also: tables are extracted but then flattened into the text stream, so table structure is
partly lost before chunking.

---

## 5. Stage 2: Metadata extraction

**File:** `src/ingestion/metadata_extractor.py`

Pulls structured fields out of unstructured text using regex, calibrated to RBI's
conventions.

| Field | How it is found |
|---|---|
| Circular number | Patterns like `RBI/2023-24/123`, `DoR.FIN.REC.21/...`, `DNBR.PD.CC.No.123` |
| Date | Three formats: `January 15, 2024`, `15.01.2024`, `2024-01-15`, normalised to ISO |
| Title | PDF metadata first, then "Master Direction - X" patterns, then first substantial line |
| Category | Keyword match: master direction, circular, notification, guidelines, etc. |
| Issuing department | `Department of Regulation`, `DoR`, `DNBS`, `DPSS`, ... |
| Applicability | Entity patterns: `NBFC-ML`, `NBFC-BL`, `All NBFCs`, `Housing Finance Companies` |
| **Sections** | Hierarchical heading patterns, 3 levels |

### Section hierarchy (the important part)

`SECTION_PATTERNS` assigns a level to each heading pattern:

- **Level 1**: `CHAPTER III`, `PART II`
- **Level 2**: `Section 5`, or numbered headings like `3. Capital Requirements`
- **Level 3**: sub-clauses `(a)`, `(ii)`

Each detected heading is stored as a `SectionInfo` with its `start_char_offset`
(character position in the concatenated document text). This offset is what lets the
structural chunker cut the document at real section boundaries, and what produces
citations like `Chapter III > Capital Requirements`.

### Known defect (worth volunteering in an interview)
`_extract_sections` advances `char_offset` for **every** page including empty ones, while
`StructuralChunker._build_full_text_with_offsets` **skips** empty pages. Each skipped page
shifts the offsets by 2 characters, so section boundaries can drift in documents with blank
pages. Also, title extraction currently picks up the bilingual header
`भारतीय रिज़र्व बैंक` as the document title on several RBI PDFs, which pollutes citations.

---

## 6. Stage 3: Chunking

**Files:** `src/ingestion/chunking/`

### Why chunk at all

Three hard constraints:

1. **Context limits.** You cannot paste 108 pages into a model prompt.
2. **Retrieval precision.** One vector per document is too coarse. If a 100-page document
   is a single point in space, "what is the CRAR" retrieves the whole document, and the
   relevant two sentences are diluted by 99 irrelevant pages.
3. **Citation granularity.** To cite "Page 7, Capital Requirements", the retrieved unit
   must be about that size.

Chunking is the **unit-of-observation decision**, and like that decision in modelling, it
dominates downstream performance.

### Tokens and tiktoken

A **token** is a sub-word unit (roughly 0.75 words in English). Models count tokens, not
characters. `BaseChunker.count_tokens` uses **tiktoken** with the `cl100k_base` encoding
(GPT-4's tokenizer) to measure chunk sizes.

**Subtlety:** `cl100k_base` is not the tokenizer the embedding model uses (BGE uses a
BERT-style WordPiece vocabulary). So the token counts are a good proxy but not exact for
the embedder. Mentioning this shows real depth.

### The three strategies

This project deliberately implements three, to **compare** them. That comparison is the
analytical deliverable.

#### Strategy A: `FixedChunker` (`fixed_512`)
- Concatenate all pages, tokenize the whole document, slice into exact 512-token blocks,
  **no overlap**.
- The naive baseline.
- **Weakness:** boundaries are arbitrary. A rule can be split mid-sentence, so neither
  half contains the complete rule. No overlap means no second chance to see the full rule.
- **Weakness:** page provenance is reconstructed by a word-overlap heuristic
  (`_estimate_page`), which is approximate, so citations are the least reliable here.

#### Strategy B: `RecursiveChunker` (`recursive_overlap`)
- Uses LangChain's `RecursiveCharacterTextSplitter`, 512 tokens with **128-token overlap**.
- **Recursive** means it tries separators in priority order:
  `["\n\n", "\n", ". ", "; ", ", ", " ", ""]`. It splits on paragraph breaks if it can,
  falls back to sentence ends, then to spaces, and only to raw characters as a last resort.
  So it respects natural boundaries when possible.
- **Overlap** means consecutive chunks share 128 tokens. A rule straddling a boundary
  appears whole in at least one chunk. Cost: roughly 25% more chunks to store and search.
- Runs **per page**, which preserves exact page numbers for citations.
- This is the industry default: good enough, structure-agnostic.

#### Strategy C: `StructuralChunker` (`structural`)
- Uses the detected section headings to cut at **semantic boundaries**. A chunk is a
  section.
- If a section is under `max_chunk_size` (1024 tokens), it becomes one chunk. If larger,
  it is sub-split at paragraph boundaries (`_sub_chunk_section`), never mid-sentence.
- Attaches the full **section hierarchy** (`["CHAPTER III", "Capital Requirements"]`) to
  each chunk, which is what produces precise citations.
- Falls back to one-chunk-per-page if no sections are detected (`_fallback_page_chunks`).
- **Hypothesis being tested:** for regulatory text, sections *are* the natural semantic
  units, so this should beat A and B on retrieval quality. This is the strategy the system
  defaults to.

#### What the three strategies actually produced on this corpus

Measured during ingestion of the same 15 PDFs:

| Strategy | Chunks | Avg tokens/chunk |
|---|---|---|
| `fixed_512` | 523 | 504 |
| `recursive_overlap` | 779 | 344 |
| `structural` | **1,994** | **132** |

This is the single most informative number pair in the project, and it was not obvious in
advance. Structural chunking produced **nearly 4x more chunks at a quarter the size**,
because RBI documents are densely subdivided: the metadata extractor detected 338 section
headings in the Housing Finance direction alone, so "one chunk per section" yields many
small chunks.

That cuts both ways. Small chunks are precise, so a retrieved chunk is likely to be
exactly the relevant clause, which helps citation accuracy. But they carry little
surrounding context, so a clause whose meaning depends on its parent paragraph may be
retrieved without the qualifying text. It also means the `max_chunk_size = 1024` cap
almost never binds, which in turn means the concern that chunks would exceed **BGE's
512-token input limit** and be silently truncated is largely moot in practice: at an
average of 132 tokens, very few chunks come close. The truncation risk is real for
`fixed_512` (avg 504, so chunks sit right at the limit) rather than for structural.

### Chunk metadata

`ChunkMetadata` (in `base.py`) travels with every chunk: `chunk_id` (UUID),
`document_title`, `file_name`, `file_hash`, `circular_number`, `date`, `category`,
`page_number`, `section_hierarchy`, `chunk_strategy`, `chunk_index`, `token_count`.

This metadata is the citation. Without it, you have a search engine but not an auditable
assistant.

---

## 7. Stage 4: Embeddings

**File:** `src/ingestion/embedder.py`

### What an embedding is

A function mapping text to a fixed-length vector of real numbers, trained so that
**semantically similar texts land near each other**. Here: text → 768 floats.

The useful property: "minimum capital adequacy ratio" and "CRAR requirement" contain no
shared keywords but produce nearby vectors. Keyword search cannot do that; this is the
entire reason dense retrieval exists.

Closest traditional analogy: a learned, dense feature representation, similar in spirit to
how an autoencoder or entity embedding compresses a high-cardinality categorical into a
few dense dimensions, except trained on text at internet scale.

### The model: `BAAI/bge-base-en-v1.5`

- **BGE** = BAAI General Embedding. A BERT-family transformer, ~110M parameters,
  12 layers, **768 output dimensions**, max input **512 tokens**.
- A **bi-encoder**: query and document are encoded **separately and independently**.
  This is what makes pre-computation possible. Document vectors are computed once at
  ingestion, stored, and reused for every future query.

### Two implementation details that matter

1. **L2 normalisation** (`normalize_embeddings=True`): every vector is scaled to unit
   length. After normalisation, cosine similarity equals the dot product, which makes
   similarity cheap to compute and bounded in `[-1, 1]`. It also removes the effect of
   text length on magnitude.

2. **Asymmetric prefix** (`BGE_QUERY_PREFIX`): BGE was trained with queries prefixed by
   `"Represent this sentence for searching relevant passages: "` and documents left bare.
   `embed_query` adds the prefix, `embed_documents` does not. Forgetting this is a classic
   silent quality bug: it still "works", just measurably worse, because the query vector
   lands slightly off-distribution.

This asymmetry reflects a real property of search: a short question and a long passage are
different kinds of text, even when they are about the same thing.

---

## 8. Stage 5: Vector store (Qdrant)

**File:** `src/retrieval/dense_retriever.py`

### What a vector database does

Given a query vector, find the `k` nearest stored vectors, fast. With 1,994 chunks you
could brute-force it. At millions of vectors you cannot, so you need an index.

### Terms

- **Cosine distance** (`Distance.COSINE`): similarity = cosine of the angle between two
  vectors. 1.0 = identical direction, 0 = orthogonal (unrelated), -1 = opposite. Direction
  carries the meaning; magnitude is discarded.
- **HNSW (Hierarchical Navigable Small World)**: Qdrant's default index. A multi-layer
  proximity graph. Search enters at a sparse top layer, greedily hops toward the query,
  then descends into denser layers to refine. Gives roughly `O(log n)` search instead of
  `O(n)`.
- **ANN (Approximate Nearest Neighbour)**: HNSW is *approximate*. It may miss a true
  nearest neighbour occasionally in exchange for a large speed gain. This accuracy/latency
  trade-off is tunable (`ef_construct`, `m`); this project uses Qdrant's defaults.
- **Point**: one stored record = `id` + `vector` + `payload`.
- **Payload**: the metadata stored alongside the vector (text, title, page, section,
  strategy). Returned with search hits, so no second database lookup is needed.
- **Collection**: the equivalent of a table. Here: `credit_policy`.
- **Filtering**: `strategy_filter` builds a Qdrant `Filter` on the `chunk_strategy`
  payload field, so one collection can hold all three chunking strategies and you can
  query a single strategy at a time. That is what makes the A/B/C comparison possible
  without maintaining three stores.

### API note (a bug we fixed)
`client.search()` was removed in qdrant-client 1.13+. The code now calls
`client.query_points()` and reads `response.points`. The installed client is 1.19.1, which
is why the original code raised `AttributeError: 'QdrantClient' object has no attribute
'search'`.

### Open defect
`upsert_chunks` assigns point IDs as `id=i`, a simple enumeration starting at 0. Calling
it twice without recreating the collection **overwrites** earlier points. IDs should be
derived from `chunk_id` (e.g. a UUID or a hash) to make ingestion idempotent.

---

## 9. Stage 6: Sparse retrieval (BM25)

**File:** `src/retrieval/sparse_retriever.py`

### Why keep keyword search at all

Dense embeddings are bad at **exact rare tokens**. Terms like `NBFC-ML`, `Section 45-IA`,
`DoR.FIN.REC.21`, or `15 per cent` may be mangled or smoothed away by an embedding, but
they are exactly what a regulatory question hinges on. BM25 nails them.

### BM25 explained

**BM25 (Best Match 25)** is the standard lexical ranking function, a refinement of TF-IDF.
For a query term `q` in document `D`:

```
score(q, D) = IDF(q) · [ f(q,D) · (k1 + 1) ] / [ f(q,D) + k1 · (1 - b + b · |D| / avgdl) ]
```

- `f(q,D)`: term frequency, how often the term appears in the chunk.
- `IDF(q)`: inverse document frequency. Rare terms score higher. `the` is worthless,
  `CRAR` is highly informative. Same intuition as information value in a scorecard: rare
  and discriminating beats common and uninformative.
- **`k1` (default 1.5): term-frequency saturation.** Ten occurrences is not ten times
  better than one. This is the diminishing-returns transform, conceptually like capping or
  log-transforming a count feature.
- **`b` (default 0.75): length normalisation strength.** Divides by document length
  relative to the average (`avgdl`), so long chunks do not win just by being long.
- Total score = sum over query terms.

`rank_bm25`'s `BM25Okapi` implements this. It is a **statistical model with no training**:
it just needs corpus term statistics.

### Custom tokenizer

`_tokenize` is tuned for regulatory text:

```python
re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
```

It **keeps hyphenated compounds intact**, so `nbfc-ml` stays one token instead of becoming
`nbfc` + `ml`. For this corpus that is a meaningful precision gain. Tokens shorter than 2
characters are dropped unless numeric.

### Persistence
The index is serialised to JSON (`bm25_index.json`, ~3.7 MB here) holding the chunks and
the tokenized corpus, then rebuilt in memory on load. BM25 is in-process, not a server,
which is why the API must be restarted after ingestion to pick up a new index.

---

## 10. Stage 7: Hybrid fusion (RRF)

**File:** `src/retrieval/hybrid_retriever.py`

Dense gives 50 results. Sparse gives 50 results. How do you merge two ranked lists whose
scores are on completely different scales (cosine in `[-1,1]` vs unbounded BM25)?

### The naive answer and why it fails
Normalise both to `[0,1]` and average. Fragile: min-max normalisation depends on the
extreme values in each individual result set, so the mapping shifts from query to query,
and one outlier distorts everything.

### Reciprocal Rank Fusion

Throw the scores away. Use only **ranks**.

```
RRF(d) = Σ over retrievers  1 / (k + rank_i(d))        with k = 60
```

- `rank` is position in that retriever's list (1-based in the formula; the code uses
  `1.0 / (self.rrf_k + rank + 1)` with a 0-based `rank`, which is equivalent).
- **`k = 60`**: a damping constant from the original RRF paper. It flattens the curve so
  the gap between rank 1 and rank 2 is not overwhelming, which stops a single retriever
  from dominating. Larger `k` = flatter, more democratic.
- A chunk found by **both** retrievers accumulates two contributions, so agreement is
  rewarded. That is the real benefit: consensus across two very different retrieval
  mechanisms is strong evidence.

This is **rank-based ensembling**. The closest familiar analogue is combining model
rankings by average rank rather than by averaging raw scores, which is exactly what you do
when two models' scores are not calibrated to each other.

### Implementation details
- Deduplication key is `chunk_id`. The same chunk from both lists merges into one entry.
- Original scores are preserved as `original_score`, fused score as `rrf_score`.
- Returns `max(top_k, 50)` candidates, so the reranker always gets a healthy pool.
- Per-stage timings (`embed_ms`, `dense_ms`, `sparse_ms`, `fuse_ms`) are attached for
  observability.

---

## 11. Stage 8: Reranking (cross-encoder)

**File:** `src/retrieval/reranker.py`

This is the single highest-leverage quality component, and a favourite interview topic.

### Bi-encoder vs cross-encoder

| | Bi-encoder (retrieval) | Cross-encoder (reranking) |
|---|---|---|
| Input | query and document **separately** | query and document **together**, as one pair |
| Output | two vectors, compared by cosine | one relevance score |
| Pre-computable? | Yes, documents embedded once | **No**, every pair needs a fresh forward pass |
| Cost | ~1 ms over millions | ~10 ms per pair |
| Accuracy | Good | **Much better** |

Why is the cross-encoder better? **Full attention between query and document tokens.** It
can see that the question says "minimum" and the passage says "shall maintain a minimum of
15 per cent" and that those refer to the same thing. A bi-encoder compresses each side to
768 numbers *before* any comparison, so that interaction is lost.

### The two-stage pattern

Running a cross-encoder over 1,994 chunks per query would take ~20 seconds. So:

1. **Retrieve** (cheap, high recall): hybrid search narrows 1,994 → 50.
2. **Rerank** (expensive, high precision): cross-encoder scores those 50 → top 5.

This is exactly the **screening-then-scoring** pattern in credit decisioning: a cheap
policy filter removes most of the population, then the expensive model scores the
survivors. The key design property is that stage 1 optimises **recall** (do not lose the
right answer) and stage 2 optimises **precision** (rank the right answer first).

### The model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- 6-layer MiniLM, a distilled (compressed) BERT.
- Trained on **MS MARCO**, a large dataset of real Bing queries with relevance labels.
- Outputs a single **logit**: unbounded, roughly `-11` to `+11`. **Not a probability.**
  This detail matters enormously for the abstention threshold (next section).

### Observed behaviour in this repo

| Question | Top rerank score |
|---|---|
| "What is the minimum CRAR requirement for NBFC-ML?" | **+6.83** |
| "What is the cooling off period for digital loans?" | **+7.33** |
| "What restrictions apply to recovery agents?" | **+3.01** |
| "What is the capital gains tax rate on equity shares in the USA?" | **-4.68** |

The separation between in-domain and out-of-domain questions is large and clean, which is
what makes threshold-based abstention viable here.

---

## 12. Stage 9: Abstention

**File:** `src/generation/abstention.py`

### Why refusing matters more than answering

In a regulated credit context, a confidently wrong answer about a capital requirement is
far more damaging than "I don't know". Abstention is the system's **safety control**.

Direct analogy: your scorecard's low-confidence band routing to manual underwriting
instead of auto-deciding. You accept lower automation for lower error cost.

### The three checks

1. **Score threshold.** `max_score < abstention_threshold (0.35)` → abstain.
2. **Minimum support.** Fewer than 2 chunks scoring above `threshold × 0.5` → abstain.
   Rationale: a real regulatory answer is usually corroborated by more than one passage.
3. **Score spread.** If the second-best score is under 30% of the best, it logs a warning
   but **does not** abstain, on the reasoning that one strong match is still useful.

### The calibration problem (be ready for this)

The threshold `0.35` sits on a **logit** scale, not a probability scale. A newcomer reads
`0.35` and assumes "35% confidence", which is wrong. On the observed range (`-4.68` to
`+7.33`), a cut at `0.35` is close to the logit midpoint and behaves sensibly, but the
value was **not** calibrated: the docstring claims it was tuned on the 120-question gold
set, and no code in the repo does that.

**The honest, high-scoring answer:** "The threshold is currently a plausible default, not a
calibrated cut-off. To calibrate it properly I would run the 120-question gold set, sweep
the threshold, and plot abstention precision against recall on the 20 unanswerable and 100
answerable questions, then pick the operating point that reflects the business cost of a
wrong answer versus an unnecessary refusal. That is the same exercise as choosing a
score cut-off from a cost matrix." You can also note that passing the logit through a
sigmoid would make the number interpretable as a probability.

### Metrics definition (as implemented in `evaluator.py`)

"Positive" = correctly abstaining on an unanswerable question.

- **TP**: abstained, truly unanswerable.
- **FP**: abstained, but it was answerable (an unnecessary refusal, hurts usefulness).
- **FN**: answered, but it was unanswerable (**the dangerous case**, a hallucination risk).
- **Abstention precision** = TP / (TP + FP): when it refuses, how often is it right?
- **Abstention recall** = TP / (TP + FN): of all the questions it should refuse, how many
  did it catch?

In a safety setting you typically want **recall** high, accepting some false refusals.

---

## 13. Stage 10: Generation

**File:** `src/generation/generator.py`

### Prompt engineering as control

The `SYSTEM_PROMPT` enforces the behaviour that makes the system auditable:

1. Answer **only** from the provided context, no outside knowledge.
2. Every factual claim must carry `[Document, Section, Page N]`.
3. If context is insufficient, say so verbatim.
4. Do not infer beyond the text.
5. Address multi-part questions part by part.

Two settings reinforce this:
- **`temperature = 0.1`**: temperature controls randomness in token sampling. High values
  give creative, varied text; near-zero gives near-deterministic, repeatable output. For
  regulatory answers you want reproducibility, so it is set low.
- **`max_tokens = 2048`**: output length cap.

### Context formatting
`_format_context` numbers each retrieved chunk and prefixes it with Document, Section and
Page. The model cannot produce a citation it was never shown, so the citation format is
established in the input, not just requested in the instructions.

### Retrieval-only mode (how this repo currently runs)

`LLM_PROVIDER=none` enables `_generate_extractive`, which needs **no API key**:

1. Extract content words from the question, dropping stopwords (`_query_terms`).
2. Split each top chunk into sentences.
3. Score each sentence by term overlap with the question.
4. Return the best 1 to 2 sentences per chunk, in original reading order, each with a
   `[Title, Section, Page N]` citation.

This is **extractive** QA (quote the source) rather than **abstractive** QA (write a
synthesis). It cannot paraphrase, combine facts across documents, or answer multi-hop
questions well. But it cannot hallucinate either, since every word is copied from a source
document. For evaluating *retrieval* quality it is actually the cleaner tool, because the
LLM is removed as a confounder.

**Pipeline position matters:** retrieval, fusion, reranking, abstention and citation
verification are all identical in this mode. Only the answer-writing step changes.

### Token usage
Tracked as `prompt` / `completion` / `total`. This is the cost meter: commercial LLM
pricing is per token, so logging it per query gives you unit economics.

---

## 14. Stage 11: Citation verification

**File:** `src/generation/citation_verifier.py`

A trust layer, and the most distinctive piece of the design. An LLM instructed to cite can
still **fabricate a citation**: a real-looking document name with a page number that does
not support the claim. So every citation is checked programmatically after generation.

### Process

1. **Parse** citations from the answer with `CITATION_PATTERN`, capturing document,
   section and page from `[...]` brackets.
2. **Verify** each against the chunks that were actually retrieved:
   - Document title: fuzzy match.
   - Section: fuzzy match against `section_hierarchy`, or against the first 500 characters
     of the chunk text.
   - Page: exact match with **±1 page tolerance**, because chunks can straddle page breaks.
3. **Mark** unverified citations with `⚠️` in the returned answer, and report
   `verified_count` / `unverified_count`.

### Fuzzy matching (`_fuzzy_match`)
Normalise whitespace and case, then: direct substring match, or at least **60% of
significant words** (longer than 3 characters) overlapping. This tolerates the LLM writing
"Master Direction - NBFC-ND-SI" when the metadata says "Master Direction – NBFC ND SI",
without accepting an entirely different document.

### Why this is a strong talking point
It converts "the model promises it cited correctly" into "the system verified the citation
against retrieved evidence". That is the difference between a demo and something defensible
in front of a risk or audit function. In all three answered test queries, 3 of 3 citations
verified and 0 were unverified.

### Known weakness
The regex mis-handles document titles containing commas, and it matches the `[Context N]`
labels from the prompt as if they were citations. A more robust approach is a delimiter
unlikely to occur in titles, or structured citation output (JSON) instead of inline text.

---

## 15. The evaluation harness

**Files:** `src/evaluation/evaluator.py`, `src/evaluation/benchmark.py`,
`src/evaluation/gold_set.json`

This is where the project becomes measurable rather than anecdotal, so expect heavy
questioning here.

### The gold set

120 questions, deliberately stratified:

| Category | Count | Purpose |
|---|---|---|
| `easy_lookup` | 50 | Single-fact retrieval ("minimum CRAR for NBFC-ML") |
| `multi_hop` | 30 | Requires combining two or more documents |
| `inference` | 20 | Requires reasoning over retrieved text, not just copying |
| `unanswerable` | 20 | **Must be refused.** Tests abstention. |

Including unanswerable questions is the mark of a serious evaluation design: it measures
what the system does when it *should not* answer, which a question set of only answerable
questions can never reveal. Comparable to deliberately including out-of-policy cases in a
validation sample.

### Retrieval metrics

**Recall@K** (`recall_at_k`): of the relevant chunks, what fraction appear in the top K?

```
Recall@5 = |retrieved_top5 ∩ relevant| / |relevant|
```

Answers: "did we find the evidence at all?" Recall is what stage 1 must maximise, because
anything not retrieved can never be reranked into the answer.

**MRR (Mean Reciprocal Rank)** (`mrr`): `1 / rank_of_first_relevant_result`, averaged over
queries.

- First result relevant → 1.0
- Second → 0.5
- Third → 0.33
- Not found → 0

Answers: "how high up is the evidence?" Recall@5 treats positions 1 and 5 as equal; MRR
does not. You want both, for the same reason you look at AUC **and** the capture rate in
the top decile.

### Faithfulness (`faithfulness_score`)

A heuristic grounding check: split the answer into sentences, and count a sentence as
"grounded" if more than 50% of its content words (longer than 3 characters) appear in the
retrieved context. Score = grounded sentences / total sentences.

Be upfront that this is **lexical overlap, not semantic entailment**. It will reward a
correctly-worded but logically wrong statement, and penalise a correct paraphrase. The
proper tool is an **LLM-as-judge** approach (e.g. Ragas, which is in `requirements.txt`
but not wired up) that asks a model whether each claim is entailed by the context.

### Latency metrics
`p50`, `p75`, `p90`, `p95`, `p99`, mean, max via numpy percentiles. **p50** is the typical
experience; **p95** is the tail that determines whether users think the system is slow.
Reporting the mean alone hides the tail.

### The comparison table (the intended deliverable)

`BenchmarkRunner._format_comparison_table` produces:

| Strategy | Recall@5 | MRR | Faithfulness | Abstention P/R | p50 | p95 |
|---|---|---|---|---|---|---|
| fixed_512 | | | | | | |
| recursive_overlap | | | | | | |
| structural | | | | | | |
| structural_reranked | | | | | | |

The design intent is to show (a) which chunking strategy wins, and (b) how much the
reranker adds on top, by isolating one variable at a time. That is an ablation study, the
same logic as adding one feature block at a time to see its marginal lift.

### Two blocking defects in the harness (know these)

1. **`relevant_chunks` is empty for all 120 questions.** Recall@5 and MRR are computed
   against chunk IDs, and `compute_aggregate_metrics` filters to entries that have
   `relevant_ids`, so **no questions qualify and both metrics come out 0**. Worse, chunk
   IDs are random UUIDs regenerated on every ingest, so they could never be labelled in
   advance. The fix is to score against `relevant_documents` (which *is* populated, with
   the 15 filenames) by mapping each retrieved chunk to its source document.

2. **`run_evaluation.py` only applies the reranker when the strategy name contains
   `reranked`.** For the other three strategies, chunks have no `rerank_score`, so
   `should_abstain` reads 0.0 and abstains on nearly everything. Their rows in the
   comparison table would be meaningless. The fix is to rerank for every strategy, and
   make reranking the explicit ablation dimension.

Walking an interviewer through these two bugs, and their fixes, demonstrates more than a
table of numbers would.

---

## 16. Serving layer

### FastAPI (`src/api/main.py`)

- **ASGI** framework: asynchronous, so the server handles concurrent requests without a
  thread per request.
- **`lifespan` context manager**: loads the embedder, retrievers, reranker, generator,
  abstention checker and verifier **once at startup** into a module-level `_pipeline`
  dict. Loading these models per request would add seconds. This is why the first startup
  is slow (it downloads ~500 MB of model weights) and subsequent queries are fast.
- **Pydantic models** (`src/api/models.py`): typed request/response schemas with
  validation at the boundary. `QueryRequest` enforces `min_length=5`, `max_length=2000`,
  `top_k` between 1 and 20, and a regex on `strategy`. Invalid input is rejected with a
  422 before any compute happens. Think of it as input schema validation on a scoring
  service.
- **Auto-generated OpenAPI docs** at `/docs`, derived from those type hints.

**Endpoints:** `POST /query`, `GET /health`, `GET /documents`, `POST /ingest`,
`GET /metrics`.

**Error-handling fix we made:** an LLM authentication failure used to surface as a 500
with a raw traceback. It now returns a **502** with an actionable message, which is the
correct semantics: the upstream dependency failed, not our server.

### Structured logging (`src/api/middleware.py`)

**structlog** emits **JSON** log lines rather than prose. Machine-parseable logs can be
queried and aggregated; free-text logs cannot.

- `RequestLoggingMiddleware` assigns a short **request ID** (UUID prefix) to every
  request, times it, and returns `X-Request-ID` and `X-Duration-Ms` headers. Request IDs
  are what let you trace one user's query through the logs.
- `log_query` emits one rich event per query: question, strategy, retrieved chunk IDs, top
  rerank score, abstention decision, token counts, per-stage latency, answer length,
  citation count.

That event schema is effectively **model monitoring** for a RAG system. With it you can
track abstention rate over time (a drift signal), latency percentiles, token spend per
query, and which chunks are retrieved most often. A known gap: `request_id` is hardcoded
to `"N/A"` in the query handler instead of reading `request.state.request_id`, so logs
cannot currently be joined end to end.

### Streamlit frontend (`frontend/app.py`)
Pure-Python UI: query box, strategy selector, top-K slider, answer card, citation badges
(green verified / amber unverified), expandable retrieved-context panels with scores, and
a benchmark comparison tab.

Two fixes we made here: answers are now HTML-escaped before rendering (closing an
injection path, since document text and model output are untrusted), and the light-coloured
cards now set explicit text colours so they are not invisible under a dark theme.

### Configuration (`config/settings.py`)
**pydantic-settings** gives typed, validated, environment-backed configuration: `.env`
values are parsed and type-checked into a `Settings` object, exposed through a cached
`get_settings()` singleton. No magic numbers scattered through the code; every knob
(top-K values, `rrf_k`, threshold, chunk sizes, model names) is declared in one place.
`Literal["openai", "google", "none"]` means an invalid provider fails at startup rather
than mid-request.

### Containerisation
`Dockerfile` (Python 3.11-slim, install requirements, expose 8000/8501) and
`docker-compose.yml` defining three services: Qdrant (with a named volume for
persistence), the API, and the frontend. **Docker** packages an app with its dependencies
so it runs identically anywhere; **compose** declares a multi-service stack and its
dependency order. Note: your machine runs Qdrant as a standalone Windows binary instead,
which is functionally equivalent for a single-node setup.

---

## 17. Measured numbers from an actual run

Useful to quote, since they are from this corpus on this machine:

| Quantity | Value |
|---|---|
| Documents ingested | 15 RBI PDFs |
| Chunks indexed (structural) | **1,994** |
| Embedding dimension | 768 |
| Candidates after hybrid fusion | 50 |
| Chunks sent to generation | 5 |
| Retrieval latency | ~2.0 s |
| Rerank latency | ~0.55 s |
| **Total query latency** | **~2.6 to 3.0 s** |
| Rerank score, in-domain questions | +3.0 to +7.3 |
| Rerank score, out-of-domain question | -4.68 |
| Citations verified in test answers | 3 of 3, zero unverified |

### Where the latency actually goes (measured, not assumed)

An earlier draft of this document assumed BM25 and query embedding dominated retrieval
latency. Instrumented per-stage logging during the benchmark run disproved that:

| Stage | Measured |
|---|---|
| Dense search (Qdrant HTTP round trip) | **~2,050 ms** |
| Cross-encoder rerank (50 pairs) | ~2,400 ms |
| Query embedding | ~50 to 1,000 ms |
| **BM25 sparse search** | **~12 to 28 ms** |
| RRF fusion | <1 ms |
| Extractive generation | 2 to 5 ms |

Two conclusions, both counter-intuitive:

1. **BM25 was never the bottleneck.** It is roughly 100x faster than the dense path,
   because `rank_bm25` is numpy-vectorised internally. The in-process design is a
   correctness and scaling concern, not a speed one.
2. **Qdrant was not slow; DNS was.** 2 seconds for a local vector search over 2,000
   vectors is implausible. The cause is the `localhost` hostname: on Windows it resolves
   to `::1` (IPv6) first, and that attempt has to fail before the client falls back to
   IPv4. Setting `QDRANT_HOST=127.0.0.1` removes the round trip entirely.

The lesson worth repeating in an interview: the per-stage timings were already being
computed in `HybridRetriever.search` (`embed_ms`, `dense_ms`, `sparse_ms`, `fuse_ms`) but
were never logged, so the bottleneck was mis-attributed by guesswork for far longer than
necessary. Instrument first, optimise second.

---

## 18. Known gaps and how you would fix them

Interviewers consistently rate honest, specific self-critique above polish. Ranked by
severity:

**Fixed since the first draft:**

| Was | Fix applied |
|---|---|
| `relevant_chunks` empty, so Recall@5 and MRR were 0 | Retrieval is now scored at **document level** against `relevant_documents`, over the 100 answerable questions. Recall@1/@3/@5, MRR and a per-category breakdown are reported |
| Reranker applied to only 1 of 4 benchmarked strategies | Reranking now runs for **every** strategy, so the chunking comparison is apples-to-apples. `structural_norerank` was added as an explicit ablation to isolate the reranker's contribution |
| Abstention metrics reported for configurations with no comparable score | Abstention is now marked **unevaluated** for the no-rerank ablation rather than reported as a misleading number |
| Retrievers rebuilt per question in the benchmark, reloading a multi-MB BM25 index 480 times | Retrievers are cached per strategy |
| Gold set referenced a filename never saved to disk (the aliased interest-rate direction) | Aliases are read from `manifest.json` and applied before scoring; any gold document still missing from the corpus is logged as a warning |
| `localhost` added ~2s to every Qdrant request via IPv6 fallback | `QDRANT_HOST=127.0.0.1` |
| LLM auth failure surfaced as a 500 with a traceback | Returns a 502 naming the likely cause |
| Answer invisible in dark theme; example questions did not run | Explicit text colours; example buttons write to the widget key and trigger the search |

**Still open:**

| Gap | Impact | Fix |
|---|---|---|
| Abstention threshold not calibrated | Safety control is a guess | Sweep threshold on the gold set, choose from a cost matrix, optionally sigmoid the logit |
| Structural chunks up to 1024 tokens vs BGE's 512 limit | Tail of long sections invisible to dense retrieval | Cap at 512, or use a long-context embedder |
| `/ingest` with `strategy="all"` raises `UnboundLocalError` | Endpoint broken on that path | Move the import to module scope |
| `scripts/ingest.py` writes per-strategy collection names the API never reads | Ingest via script then query returns nothing | Unify naming via settings |
| Document titles picked from bilingual header | Citations show `भारतीय रिज़र्व बैंक` | Use the known titles in `data/raw_pdfs/manifest.json` |
| Point IDs are `enumerate` indices | Re-ingesting overwrites points | Derive IDs from `chunk_id` |
| Section offsets drift on empty pages | Structural boundaries can slip | Make offset accounting consistent across both modules |
| Faithfulness is lexical overlap | Weak quality signal | Wire up Ragas / LLM-as-judge |
| No authentication, `CORS allow_origins=["*"]` | Unsafe beyond localhost | API keys or OAuth, restrict origins |
| `request_id` hardcoded to `"N/A"` in query logs | Cannot trace a request end to end | Pass `Request` into the handler and read `request.state.request_id` |

---

## 19. Glossary

**ANN** – Approximate Nearest Neighbour. Trades a little accuracy for a large speed gain
in vector search.

**Abstractive vs extractive** – Writing a new synthesis versus quoting source sentences.

**ASGI** – Asynchronous Server Gateway Interface, the async Python web standard FastAPI
builds on.

**Bi-encoder** – Encodes query and document separately into comparable vectors. Fast,
pre-computable.

**BM25** – Lexical ranking function based on term frequency, inverse document frequency,
and length normalisation.

**Chunk** – A passage-sized piece of a document; the unit that is embedded, retrieved and
cited.

**Cosine similarity** – Similarity as the cosine of the angle between two vectors; ignores
magnitude.

**Cross-encoder** – Scores a (query, document) pair jointly with full attention. Slow but
accurate.

**Dense retrieval** – Search over learned embedding vectors; matches meaning.

**Embedding** – A fixed-length numeric vector representing text, positioned so that similar
meanings are nearby.

**Grounding** – Ensuring every generated claim traces back to supplied evidence.

**Hallucination** – Fluent, confident model output that is not supported by any source.

**HNSW** – Hierarchical Navigable Small World, the graph index used for fast vector search.

**IDF** – Inverse Document Frequency; rare terms are more informative.

**Logit** – An unbounded raw model score, pre-sigmoid. Not a probability.

**MRR** – Mean Reciprocal Rank; rewards placing the first relevant result high.

**Payload** – Metadata stored with a vector in Qdrant and returned with search results.

**Quantisation / distillation** – Compressing a model (precision reduction / training a
small model to mimic a large one). MiniLM is distilled.

**RAG** – Retrieval-Augmented Generation; retrieve relevant text, then generate an answer
conditioned on it.

**Recall@K** – Fraction of relevant items appearing in the top K results.

**Reranking** – Second-stage, higher-precision re-scoring of first-stage candidates.

**RRF** – Reciprocal Rank Fusion; merges ranked lists using ranks, not scores.

**Semantic search** – Retrieval by meaning rather than exact keywords.

**Sparse retrieval** – Keyword-based search (BM25); matches literal tokens.

**Temperature** – Sampling randomness in generation. Low = deterministic.

**Token** – Sub-word unit; the counting unit for model inputs, outputs and cost.

**Top-K** – How many results a stage returns.

**Vector database** – A store optimised for nearest-neighbour search over embeddings.

---

## 20. Likely interview questions, with answers

**Why hybrid retrieval instead of just embeddings?**
Dense retrieval captures meaning but blurs exact rare tokens, and regulatory questions turn
on exactly those: `NBFC-ML`, `Section 45-IA`, `15 per cent`. BM25 matches them precisely.
They fail differently, so fusing them is complementary rather than redundant. RRF merges
them on rank, which avoids the scale-mismatch problem of combining cosine with BM25 scores.

**Why rerank if retrieval already ranks?**
Different objectives. Retrieval maximises recall cheaply over the whole corpus; the
cross-encoder maximises precision expensively over a shortlist. The cross-encoder sees the
query and passage together with full attention, which a bi-encoder structurally cannot,
because it compresses each side before comparison. Running it over all 1,994 chunks would
take about 20 seconds, so it is applied to 50.

**Why these three chunking strategies?**
To isolate a variable. Fixed-512 is the naive baseline. Recursive-with-overlap adds
boundary awareness and overlap so a rule straddling a split still appears whole somewhere.
Structural respects the document's own sections, which for regulatory text are the real
semantic units, and it is the only one that yields a reliable section hierarchy for
citations. Chunking is the unit-of-observation decision, so it deserved an ablation rather
than an assumption.

**How do you stop hallucination?**
Four layers. Retrieval constrains the model to a fixed evidence set. The prompt forbids
outside knowledge and mandates a citation per claim, with temperature near zero.
Abstention refuses before generation when retrieval confidence is too low. Then citation
verification programmatically checks every citation against the chunks that were actually
retrieved and flags any that do not match. The fourth layer is the important one: it does
not trust the model's own claim that it cited correctly.

**How do you know abstention works?**
20 of the 120 gold questions are deliberately unanswerable, and abstention precision and
recall are computed against them. Empirically the separation is wide: in-domain questions
scored +3.0 to +7.3 while an out-of-domain tax question scored -4.68 and was refused. But
I would not claim the threshold is calibrated, because nothing in the repo tunes it. The
proper step is a threshold sweep on the gold set with the operating point chosen from the
relative cost of a wrong answer versus an unnecessary refusal.

**What is the main weakness of the current evaluation?**
`relevant_chunks` is empty for every gold question, so Recall@5 and MRR evaluate to 0 and
are not usable. Chunk IDs are random UUIDs regenerated on each ingest, so they could never
be pre-labelled. The fix is document-level scoring against `relevant_documents`, which is
populated. Separately, the reranker is only applied to one of the four strategies in the
benchmark, which invalidates the cross-strategy comparison.

**Where does the latency go, and how would you cut it?**
About 2.0 s of the ~2.8 s total is retrieval, dominated by query embedding and by BM25
scoring 1,994 chunks in pure Python; Qdrant itself answers in milliseconds. Reranking 50
pairs on CPU is ~0.55 s. First steps: move BM25 into Qdrant's native sparse vectors to
avoid the Python loop, batch or GPU the cross-encoder, cache embeddings for repeated
queries, and trim the candidate pool from 50 if the recall cost is acceptable.

**How would you handle a regulation update?**
Re-run the scraper and re-ingest. Documents are data, not weights, so there is no
retraining. The `file_hash` on every chunk gives provenance, and `manifest.json` records
which source document and date each file came from. What is missing today is versioning:
chunks should carry an effective-date range so superseded guidance can be excluded or
clearly labelled, which matters because RBI consolidated the NBFC directions in November
2025 and several legacy documents in this corpus are superseded.

**Why retrieval-only mode, and what does it cost you?**
It removes the dependency on a paid LLM key, and it is the cleaner setup for measuring
retrieval quality because the generator is no longer a confounder. It cannot hallucinate,
since every word is copied from a source. The cost is no synthesis: it cannot paraphrase,
combine facts across documents, or handle the multi-hop and inference questions well.
Retrieval, fusion, reranking, abstention and citation verification are unchanged, so
swapping a local model such as Llama via Ollama, or Gemini, back in is a configuration
change rather than a rewrite.
