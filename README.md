# NEXUS TwinLoop

**NEXUS TwinLoop** is a demonstration framework for *almost‑continuous* learning with Large Language Models. It uses a dual‑loop design—**Active** serves users while **Shadow** learns from fresh data—plus domain adapters (PEFT‑style), external memory (RAG), canary releases, atomic swaps, and instant rollbacks. The goal: safe, incremental self‑updates without downtime.

> Status: research/demo skeleton. Standard Python library only (no external deps).

---

## Why this exists

Traditional LLMs are batch‑trained and static. NEXUS TwinLoop shows how to keep a system “alive” via:
- **Blue/Green (Active/Shadow)** deployment with atomic swaps & rollbacks
- **Domain modularity** through adapters (PEFT‑like) instead of constantly rewriting the foundation
- **RAG external memory** for fast factual updates (no weight churn)
- **Safety & quality gates** (dry‑run QA, canary routing)
- **Reproducibility** via snapshots, artifact registry, and determinism

---

## High‑level architecture

```
[Users] → [API (demo main)] → [Router (thresholds, priorities)]
                                ├─► [Foundation (stable)]
                                ├─► [Domain Adapters: Law | Med | Fin | General]
                                └─► [RAG: per‑domain indexes]

             feedback/data ───────────────────▲
                                             [Shadow Trainer]
                                  (ingest → clean → replay → PEFT‑like updates
                                   → RAG refresh → QA dry‑run → Canary → Swap)

[Artifact Registry]  versions: adapters / router / RAG payloads
[Snapshots]          complete state for rollback (adapters + RAG + router rules)
```

**Key loops**
- **Serving loop (Active):** route → encode → adapters → RAG retrieve → generate → metrics
- **Learning loop (Shadow):** ingest feedback → clean → replay sampling → train adapters with EWC‑like regularization → update RAG → evaluate (QA dry‑run) → canary → **atomic swap** if healthy → **rollback** if not

---

## Features (demo level)

- **Active/Shadow** services with **atomic swap** & **instant rollback**
- **Domain adapters** (toy PEFT) with dynamic “importance” (EWC‑like)
- **RAG** per domain (toy vector search) with simple ingestion
- **Router** with thresholding & priorities
- **Replay buffers** per domain
- **QA dry‑run**: no side‑effects on real services during evaluation
- **Canary deployer** with traffic split
- **Artifact registry** (in‑memory) + deterministic runs (fixed seed)
- **PR‑pipeline** skeleton for controlled self‑modifications (e.g., router rules)

> The demo intentionally simplifies generation, search, safety checks and persistence to keep the idea clear.

---

## Repository layout

```
.
├── nexus_twinloop_demo_2.py   # Modernized demo script (Active/Shadow, RAG, QA, Canary, Swap, Rollback)
├── nexus_twinloop_demo.py     # Earlier modernized demo (optional)
└── README.md                  # This file
```

---

## Quickstart

1) **Python 3.10+**, no external dependencies required.

2) Run the demo:
```bash
python nexus_twinloop_demo_2.py
# or: python nexus_twinloop_demo.py
```

You’ll see:
- Active answers to a small query set
- Shadow ingests feedback and fine‑tunes domain adapters
- RAG index refresh
- QA dry‑run over a holdout
- Canary traffic routing
- Decision to **swap** (promote Shadow) or **abort**
- Optional **PR** example (router rules tweak)
- (Commented) **rollback** example

---

## How it works (components)

- **FoundationModel** — immutable “base” capable of encoding & toy generation
- **DomainAdapter** — tiny “PEFT‑like” adapter with EWC‑style regularization
- **RAGIndex** — per‑domain store with toy token‑match retrieval
- **Router** — thresholded, priority‑driven keyword router (pluggable)
- **ReplayBuffer** — domain‑scoped memory for continual learning
- **ModelService** — composes Foundation + Adapters + RAG + Router; produces answers and metrics
- **ShadowTrainer** — data cleaning → replay sampling → adapter updates → RAG refresh
- **QA** — dry‑run evaluation (no mutation of active services)
- **CanaryDeployer** — traffic split, **atomic swap**, **rollback**
- **ArtifactRegistry** — in‑memory record of adapter versions & metadata
- **PRPipeline** — accepts “proposals” (e.g., router rule updates) behind basic tests

---

## What’s intentionally simplified

- **Generation** uses toy heuristics (plug an actual LLM for realism)
- **RAG** is a minimal search; replace with FAISS/Chroma/Weaviate
- **Safety** checks are placeholder heuristics (swap to proper classifiers/rules)
- **Persistence** is in‑memory (use S3/MinIO/SQLite/Postgres for artifacts & snapshots)
- **Concurrency** is single‑threaded (add locks/actors for multithread/multiprocess)

---

## Roadmap (suggested)

- Swap toy components for real ones (HF PEFT, vector DB, LLM backend)
- Persist artifacts/snapshots with checksums & lineage
- Stronger safety layer (toxicity, PII, prompt‑injection) + configs
- Robust A/B & canary decisioning (statistical tests, SLO gates)
- Admin API (FastAPI): `/answer`, `/ingest`, `/train`, `/qa`, `/canary`, `/swap`, `/rollback`, `/pr`
- Observability: Prometheus/Grafana dashboards

---

## License

MIT (suggested). Add your preferred license file for production use.

---

## Citation

If you build on this, please cite as:

> Avin & John (En‑Do), **NEXUS TwinLoop: A Dual‑Loop Framework for Almost‑Continuous LLM Updates**, 2025. GitHub repository.

---

*Made with ❤️ by Avin & John (En‑Do).*

