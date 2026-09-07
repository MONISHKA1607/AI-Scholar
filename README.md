# AI-Scholar — Agentic RAG System for Research Paper Q&A

AI-Scholar is a research-paper question-answering system built around a **retrieval-augmented generation (RAG)** pipeline. It converts research papers into searchable evidence, retrieves the most relevant evidence for a question, and uses Gemini to generate grounded, citation-backed answers.

> **Core idea:** Research papers → searchable evidence → retrieval → reranking → grounded Gemini answer with citations.

## End-to-End Architecture

### 1. Document ingestion

```text
Document / PDF
      ↓
PDF parsing
      ↓
Scanned-PDF detection
      ↓
Section-aware chunking
      ↓
Embedding generation
      ↓
ChromaDB
```

### 2. Query answering

```text
User question
      ↓
Query classification
      ↓
Agent / tool selection
      ↓
Internal knowledge-base retrieval
      ↓
Semantic Scholar fallback if needed
      ↓
Candidate retrieval
      ↓
Cross-encoder reranking
      ↓
Context construction
      ↓
Gemini generation
      ↓
Grounded answer + citations
```

---

# Why RAG?

A normal LLM workflow is:

```text
Question → LLM → Answer
```

AI-Scholar uses:

```text
Question → Retrieve evidence → LLM → Answer
```

RAG was chosen because the main problem is **accessing and grounding answers in research documents**, not teaching the model new behavior. New papers can be added by parsing, chunking, embedding, and indexing them without retraining the model.

RAG reduces hallucination risk by providing evidence, but it does not guarantee that every generated statement is correct.

---

# Document Ingestion

## PDF parsing

Research papers are converted into text before indexing. Text is extracted page by page so individual page failures can be isolated instead of crashing the whole ingestion process.

PDFs are difficult because they can contain:

- multi-column layouts
- tables
- figures
- equations
- unusual reading order
- scanned pages

### Scanned PDFs

The current implementation uses a lightweight heuristic based on the amount of extracted text. PDFs with insufficient extractable text are treated as likely scanned.

Current v1 behavior:

```text
Scanned / poorly extracted PDF → skipped_ocr
```

OCR and layout-aware parsing are future improvements.

---

# Section-Aware Chunking

A full research paper is too large and unfocused to use as one retrieval unit.

The pipeline uses two stages:

1. **Section-aware splitting** using research-paper structure such as Introduction, Methods, Results, Discussion, and Conclusion.
2. **Recursive splitting** inside large sections.

Current configuration is approximately:

```text
Chunk size: 1000
Chunk overlap: 200
```

Section-aware chunking preserves useful document structure. Recursive splitting keeps sections manageable for embedding and retrieval.

Overlap reduces the chance that important information is lost at chunk boundaries.

The values are practical starting points and should be tuned experimentally rather than treated as universally optimal.

### Other chunking approaches

- Fixed-size chunking — simple but can break logical structure.
- Recursive chunking — uses natural separators such as paragraphs and sentences.
- Semantic chunking — groups semantically related content but is more complex.
- Structure-aware chunking — uses headings, sections, pages, or layout.

---

# Embeddings

An embedding converts text into a numerical vector representing semantic meaning.

Example:

```text
"How does transformer attention work?"
```

and

```text
"Explain the attention mechanism in transformers"
```

use different wording but should be semantically related in embedding space.

AI-Scholar uses Gemini embeddings for:

```text
Document chunk → embedding vector → ChromaDB
```

and at query time:

```text
User question → embedding vector → similarity search
```

Semantic retrieval is useful because research questions often use wording different from the source paper.

---

# ChromaDB

ChromaDB is the **primary persistent knowledge base**.

Each stored record contains:

- chunk text
- embedding
- metadata

Important metadata includes:

- canonical paper ID
- title
- authors
- source
- chunk index
- section heading
- DOI
- arXiv ID
- Semantic Scholar ID
- venue
- year
- categories
- URL

Metadata provides provenance, helps debugging, supports filtering, and enables citation construction.

---

# Deduplication and Idempotency

The same paper may appear through multiple sources. AI-Scholar creates a deterministic canonical identity with priority:

```text
DOI
 ↓
arXiv ID
 ↓
Semantic Scholar ID
 ↓
Title + first-author hash
```

The title-author hash is only a fallback heuristic.

Chunk IDs are deterministic:

```text
canonical_id#chunk_0
canonical_id#chunk_1
...
```

Combined with upsert-style storage, this makes ingestion **idempotent**: retrying the same paper should not create duplicate chunk copies.

---

# Asynchronous Ingestion

PDF ingestion can involve parsing, chunking, embedding, and vector storage, so it should not block normal query requests.

The ingestion path uses:

- Redis
- RQ workers
- an ingestion queue

```text
Upload
  ↓
Validate
  ↓
Create / identify paper
  ↓
Enqueue job
  ↓
Background worker
  ↓
Parse → chunk → embed → store
```

The prototype also maintains an ingestion ledger with states such as:

```text
queued
processing
done
failed
skipped_ocr
```

RQ tracks the background job; the ledger tracks the paper-level business state.

SQLite is used for the prototype ledger. A production version would use a shared relational database such as PostgreSQL.

---

# Query Classification

Incoming questions are classified as:

```text
1. specific-to-paper
2. generic-research
3. non-research
```

Examples:

- Specific-to-paper: “What does the 2021 paper by Smith conclude about memory?”
- Generic-research: “How does transformer attention work?”
- Non-research: “Write me a poem.”

Classification prevents every prompt from following the same retrieval path and helps select the correct tools.

The current implementation uses Gemini for natural-language classification. A production system could combine deterministic rules with an LLM or lightweight classifier.

---

# Constrained Agentic RAG

AI-Scholar uses a **constrained agentic tool-selection layer**, not a fully autonomous agent.

For research queries, the system can choose from controlled tools such as:

```text
Internal KB retrieval
Semantic Scholar search
Specific-paper search
```

The agent's role is to select an appropriate information source while keeping the available tools and workflow bounded.

The general strategy is:

```text
Internal knowledge base first
        ↓
External academic fallback when needed
```

---

# Retrieval

## Stage 1: Vector retrieval

The query is embedded and searched against ChromaDB.

```text
Query
  ↓
Embedding
  ↓
ChromaDB similarity search
  ↓
Top ~20 candidate chunks
```

The first stage prioritizes recall: finding a broad set that hopefully contains the relevant evidence.

## Stage 2: Cross-encoder reranking

The candidates are reranked using a cross-encoder.

Conceptually:

```text
[Query + candidate chunk] → relevance score
```

The pipeline becomes:

```text
Top ~20 candidates
      ↓
Cross-encoder reranker
      ↓
Top ~5 context chunks
```

This follows a two-stage retrieval strategy:

```text
Cheap broad retrieval
        ↓
More expensive precise reranking
```

The reranker is not applied to the entire database because scoring every query-document pair would be too expensive.

If the reranker fails, the system can degrade gracefully by returning the original vector-retrieval order.

---

# Semantic Scholar Fallback

The internal knowledge base may not contain the requested paper or enough useful evidence.

Semantic Scholar is used as an academic fallback source for:

- paper metadata
- academic search
- specific-paper lookup
- open-access paper discovery

If an open-access PDF is available, the fallback path can create a temporary in-memory FAISS index from extracted content for targeted retrieval.

## ChromaDB vs FAISS

They have different roles:

```text
ChromaDB → persistent primary knowledge base
FAISS    → temporary in-memory fallback retrieval
```

They are not two competing primary databases.

---

# Context Construction and Citations

After reranking, the selected chunks are formatted as focused context for Gemini.

Each chunk carries source metadata and citation markers, conceptually:

```text
[^1] Paper Title
Author | Year | URL

Retrieved evidence...
```

The generation model receives actual retrieved evidence rather than only document identifiers.

If evidence is insufficient, the intended behavior is a controlled fallback instead of inventing an unsupported answer.

---

# Evaluation

The retrieval benchmark measures:

- **Hit@5**
- **MRR@10**
- **p50 latency**
- **p95 latency**

## Hit@5

> Did the relevant paper appear anywhere in the top 5 results?

## MRR@10

Measures how high the first relevant result appears.

```text
Rank 1 → 1
Rank 2 → 1/2
Rank 3 → 1/3
Not in top 10 → 0
```

The average reciprocal rank is calculated across benchmark queries.

## Latency

```text
p50 → typical latency
p95 → slower tail latency
```

The current benchmark primarily evaluates retrieval quality and latency. A stronger end-to-end evaluation would also measure:

- answer correctness
- groundedness
- citation correctness
- citation completeness
- insufficient-evidence behavior

---

# Failure Handling

The system has multiple possible failure points:

```text
Download
   ↓
PDF parsing
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector storage
   ↓
Retrieval
   ↓
Reranking
   ↓
Generation
```

Failures should be isolated as close as possible to their source.

Examples:

- Poor PDF extraction → controlled ingestion status.
- Scanned PDF → `skipped_ocr`.
- Queue failure → prototype synchronous fallback.
- Reranker failure → use vector retrieval order.
- External API failure → bounded retries and controlled errors.
- Missing evidence → fallback/refusal rather than fabrication.

A production system should clearly distinguish:

```text
No relevant result
```

from:

```text
Dependency unavailable
```

because these are operationally different failures.

---

# Debugging a Bad Answer

A bad RAG answer should be debugged backward through the pipeline:

```text
Bad answer
    ↓
Was the context correct?
    ↓
Was reranking correct?
    ↓
Was relevant evidence retrieved?
    ↓
Was chunking appropriate?
    ↓
Was PDF extraction correct?
```

This separates:

```text
Retrieval failure
```

from:

```text
Generation failure
```

For example, if the correct evidence was never retrieved, the issue is retrieval. If the correct evidence was given to Gemini but the answer is still wrong, the issue is generation or prompt behavior.

---

# Technology Stack

## Core

- Python
- FastAPI

## RAG and orchestration

- LangChain
- Gemini / Google GenAI SDK

## Vector retrieval

- ChromaDB
- FAISS

## Reranking

- Cross-encoder
- `BAAI/bge-reranker-base`

## Background processing

- Redis
- RQ

## Research sources

- Semantic Scholar
- arXiv

## Prototype ingestion state

- SQLite

---

# Why These Technologies?

### Python

Strong ecosystem for AI, NLP, vector retrieval, and API development.

### FastAPI

Lightweight Python API framework with validation and async support.

### LangChain

Used mainly for tool/agent abstraction and retrieval integration. RAG itself could also be implemented directly in Python.

### ChromaDB

Persistent vector storage with metadata support and straightforward semantic retrieval.

### FAISS

Efficient temporary in-memory vector search for fallback scenarios.

### Redis + RQ

Moves expensive ingestion work off the normal request path using Python background workers.

### Cross-encoder reranker

Improves precision when vector similarity returns chunks that are related to the topic but do not directly answer the question.

---

# Production Scaling Direction

A larger deployment would separate query serving and ingestion workers so they can scale independently:

```text
             Load Balancer
                  ↓
           FastAPI Instances
                  ↓
       ┌──────────┴──────────┐
       ↓                     ↓
Query / Retrieval      Ingestion Queue
       ↓                     ↓
Vector Store          Worker Pool
                            ↓
                    Parse / Embed / Store
```

Possible improvements:

- horizontally scaled API instances
- worker scaling based on queue depth
- ANN/vector indexing for larger corpora
- caching
- rate limiting
- bounded retries with backoff
- object storage for PDFs
- PostgreSQL for shared ingestion state
- monitoring and tracing
- hybrid BM25 + vector retrieval

---

# Security Consideration: Prompt Injection

Retrieved document content should be treated as **untrusted data**, not as instructions.

The generation layer should clearly separate:

```text
System instructions
```

from:

```text
Retrieved paper content
```

A malicious document should not be able to redefine system behavior or tool permissions.

---

# Current Limitations and Future Improvements

- Scanned PDFs are currently skipped; OCR is future work.
- Hybrid retrieval (BM25 + vector search) is not part of the current primary pipeline.
- The benchmark focuses on retrieval quality and latency rather than full end-to-end answer correctness.
- Prototype ingestion state uses SQLite.
- Production storage, observability, retries, and distributed scaling would need further work.

---

# Repository Structure

```text
AI-Scholar/
│
├── service/
│   ├── src/
│   │   ├── ingestion/       # Paper ingestion pipeline
│   │   ├── retrieval/       # Retrieval and fallback logic
│   │   ├── agents/          # Classification and tool selection
│   │   ├── evaluation/      # Retrieval benchmark
│   │   └── app.py           # FastAPI application
│   │
│   └── data/
│       └── ingestion_ledger.sqlite
│
├── README.md
└── project.md
```
