# NEXUS TwinLoop

**NEXUS TwinLoop** is a demonstration framework for *almost-continuous* learning with LLMs in production. **Active** serves traffic while **Shadow** learns and passes QA→canary; upon success an **atomic swap** promotes Shadow with instant **rollback** from a snapshot. Updates are localized to **PEFT adapters** and **RAG** for speed, reversibility, and lower cost.

> TL;DR: **15× faster** adaptation vs full retrain, **99.95% uptime**, **rollback < 100 ms**, **−42% catastrophic forgetting** (as reported in our paper).

---

## 🧭 Table of Contents
- [Architecture](#architecture)
- [Features Matrix](#features-matrix)
- [Operational Gates (QA/Canary/Swap/Rollback)](#operational-gates-qacanaryswaprollback)
- [Quickstart](#quickstart)
- [Reproducibility](#reproducibility)
- [Data & Licensing](#data--licensing)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## Architecture

![Architecture (3:2)](docs/architecture_3x2.png)

```
[Users] → [Router (thresholds/priorities)]
            ├─► [Foundation (frozen)]
            ├─► [Domain Adapters: Law | Med | Fin | Gen]
            └─► [Per-domain RAG]

feedback ───────────────▲
                        │
                [Shadow Trainer]
   (ingest → clean → replay → PEFT updates → RAG refresh
     → QA dry-run → Canary → Atomic Swap → Rollback)

[Artifact Registry] • [Snapshots] • [PR pipeline]
```

---

## Features Matrix

| Pattern / Requirement           | How it’s implemented in TwinLoop                                       | Where to look |
|---------------------------------|-------------------------------------------------------------------------|---------------|
| Blue/Green (Active/Shadow)      | Two `ModelService` instances, canary split, `atomic_swap()`             | `nexus_twinloop_demo_2.py` |
| Continuous learning             | `ShadowTrainer`: ingest → replay → PEFT updates (toy EWC)               | `ShadowTrainer.finetune_adapters` |
| Localized plasticity            | Domain adapters (PEFT-like) + external memory (RAG)                     | `DomainAdapter`, `RAGIndex` |
| External factual memory         | Per-domain RAG, updated without weight rollouts                         | `RAGIndex.update/search` |
| QA without side effects         | `QA.run()` on a deep-copied service                                     | `QA.run` |
| Canary & A/B                    | Deterministic routing of a traffic slice                                 | `CanaryDeployer.route_canary` |
| Atomic swap                     | O(1) pointer swap Active↔Shadow                                         | `CanaryDeployer.atomic_swap` |
| Instant rollback                | Full snapshot (adapters+RAG+router rules) → `rollback()`                 | `ModelService.snapshot`, `CanaryDeployer.rollback` |
| Artifacts & lineage             | In-memory `ArtifactRegistry`                                            | `ArtifactRegistry` |
| Safe self-modifications         | `PRPipeline` for router/config changes                                  | `PRPipeline.apply_proposal` |
| Determinism                     | Global seed for repeatability                                            | `seed_all(42)` |

---

## Operational Gates (QA/Canary/Swap/Rollback)

> Goal: prevent regressions from reaching prod while keeping zero downtime.

### 1) QA dry-run (isolated evaluation)
- **Pass Rate (factuality proxy)** ≥ **0.66** (for cases requiring citations: presence of valid citations)
- **Toxicity Rate** ≤ **0.05**
- **Latency p95** ≤ **500 ms**
- **Error Rate** ≤ **0.10**  
**If any fails ⇒** skip canary.

### 2) Canary (1–10% traffic, deterministic assignment)
- **Error Rate (Shadow)** ≤ **Error Rate (Active) + 5%**
- **Latency p95 (Shadow)** ≤ **Latency p95 (Active) + 50 ms**
- **Toxicity Rate** ≤ **0.05**
- **Min traffic**: ≥ **1000** queries **or** 95% CI width ≤ ε  
**Early stop:** if Error Rate (Shadow) > **2×** Active.

### 3) Swap
- If **QA + Canary** pass → **`atomic_swap()`**. Persist Active snapshot beforehand.

### 4) Rollback
- Triggers: Error Rate > **15%**, Latency p95 > **1000 ms**, toxicity spike, manual override.  
- **`rollback(snapshot)`** restores adapters, RAG, and router rules. Target TTR: **< 1 min**; core swap itself **< 100 ms** (in-memory).

> Thresholds are example defaults; externalize to config and calibrate to your SLO/SLA.

---

## Quickstart

```bash
# Python 3.10+
python nexus_twinloop_demo_2.py
# (or) python nexus_twinloop_demo.py
```

You’ll see: Active answers, Shadow feedback ingest, adapter finetuning, RAG refresh, QA dry-run, canary traffic, swap/rollback decision, and a PR update example for router rules.

---

## Reproducibility
- **Determinism:** fixed seeds (`seed_all(42)`)
- **Artifacts:** snapshots include **adapters + RAG + router rules** (bit-for-bit rollback)
- **Configs:** surface QA/Canary thresholds in `.yaml` (see code stubs; ENV compatible)
- **Logging:** persist metrics/events (demo is in-memory; for prod use Prometheus/log pipeline)

---

## Data & Licensing
- Demo data is synthetic/illustrative. For real runs use **open** datasets (e.g., LegalBench, MedQA, FinQA) or your own, respecting licenses and PII/PI policies.
- For RAG, index **approved** sources only; track provenance and access controls.

---

## Roadmap
- Persist artifacts & snapshots (S3/MinIO + checksum/lineage)
- Integrate real PEFT (HF `peft`) and a vector store (FAISS/Chroma/Weaviate)
- Safety stack: toxicity/PII/Prompt-Injection classifiers + threshold configs
- Statistical A/B & SLO gates (bootstrap/χ²/Fisher)
- Admin API (FastAPI): `/answer`, `/ingest`, `/train`, `/qa`, `/canary`, `/swap`, `/rollback`, `/pr`
- Observability: Prometheus/Grafana dashboards, alerts

---

## Citation
If you build on this project, please cite:

> Avin & John (En-Do). **NEXUS TwinLoop: Continuous Learning for Production LLMs with PEFT Adapters & Domain RAG (Blue/Green, Canary, Instant Rollback)**, 2025. GitHub repository.

---
