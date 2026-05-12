# HW Starter — Agentic RAG

> 📍 **Lost? Start here:** [`/STUDY_GUIDE.pdf`](../../STUDY_GUIDE.pdf) ([MD](../../STUDY_GUIDE.md)) — explains how to use the materials. HW workflow detail is in [`HW_INSTRUCTIONS.pdf`](../HW_INSTRUCTIONS.pdf).
>
> 📄 **PDFs vs Markdown:** read the `.pdf` versions for better diagrams + syntax highlighting. The `.md` files are the editable source with identical content.

This is your code skeleton.

**Read in this order:**
1. [`../HW_INSTRUCTIONS.md`](../HW_INSTRUCTIONS.md) — assignment writeup, prerequisites, setup, submission, rubric
2. [`MILESTONES.md`](MILESTONES.md) — the build checklist (M1 → M7) with hint-level guidance and the inline pre-fill walkthrough

**Run the agent (after `uv sync`):**

```bash
uv run python main.py
```

The starter ships with `TEST_TO_RUN = "ingestion"` set in `main.py`. Your first run will hit a `NotImplementedError` pointing you at M3 — that's expected.

## File roles

| File | Status |
|---|---|
| `prompts.py` | Provided. Read in M1. |
| `config.py` | Provided. The `CONFIG` dict has all hyperparameters. Read in M1. |
| `pyproject.toml` | Provided. `uv sync` generates `uv.lock`. |
| `env.template` | Provided. Copy to `.env` and add your four API keys. |
| `ingestion.py` | TODO body in `load_and_process_filings`. Built in M3. |
| `retriever.py` | TODO bodies in `create_vector_store` and `create_retriever_with_reranking`. Built in M4. |
| `tools.py` | TODO bodies in three `_run` methods plus two helpers on `CompanyDirectorsTool`. Built in M5. |
| `main.py` | TODO bodies in `build_pipeline`, `build_tools`, `build_agent`, and the M7 invocation block. Built in M6 + M7. |

## Test runner switch

In `main.py`, change `TEST_TO_RUN` to verify each milestone independently:

```python
TEST_TO_RUN = "ingestion"   # M3 done?
TEST_TO_RUN = "retriever"   # M4 done?
TEST_TO_RUN = "tools"       # M5 done?
TEST_TO_RUN = None          # M7 — full agent end-to-end
```

The directors tool burns ~8 SerpAPI calls per run (one per Tesla director). See the API quota strategy in `HW_INSTRUCTIONS.md`.
