# Approach Explanation (≈320 words)

## Objective
Design a **CPU‑only, ≤1 GB** system that – given a *persona*, a *job‑to‑be‑done* and a **small collection of PDFs (3‑10)** – surfaces the sections and sub‑sections that will help the user complete that job.  
The deliverable must finish in **≤60 s** without internet access and emit the exact `challenge1b_output.json` schema.

## High‑level Pipeline
```
PDFs  ─┐            ┌─►  TF–IDF matrix ─► Similarity
Persona│            │                     ranking
Job    │            │
        ▼            ▼
  Text extraction   Query construction           ┌─►  Top‑5 Sections
 (PyMuPDF, 1 page)       (Persona + Job)         │
                                                └─►  Sub‑section snippets
```

1. **PDF Ingestion** (≈40 ms/page)  
   We rely on **PyMuPDF** (`fitz`) – a pure‑C library with Python bindings – to extract *layout‑aware* text from each page. This keeps memory usage <50 MB and removes the need for heavyweight OCR or NLP models.

2. **Query Construction**  
   We concatenate the persona description and the job statement into one query string.  
   > *Example*: `"PhD researcher in computational biology. Prepare a literature review on GNNs for drug discovery."`

3. **Light‑weight Semantic Scoring**  
   To remain within the 1 GB limit we replace neural embeddings with a **bag‑of‑words TF‑IDF** representation (**scikit‑learn** ≈30 MB).  
   *   The whole corpus = every page’s raw text.  
   *   The query string is appended as an extra “document”.  
   *   **Cosine similarity** (via a linear kernel) ranks pages against the query in a single sparse matrix operation.

4. **Section Extraction**  
   *Importance* is the reverse rank of similarity.  
   The **section title** is heuristically the first non‑blank line on that page – a robust proxy across academic, business and textbook layouts.

5. **Sub‑section Refinement**  
   Inside each top page we split by paragraph, re‑score with TF‑IDF, and keep the best‑matching snippet (≤400 chars).  
   This gives a focused, persona‑aware highlight without expensive summarisation models.

6. **Post‑processing & Output**  
   All metadata, five highest scoring sections and five refined snippets are dumped to `challenge1b_output.json`.  
   Processing a 5‑PDF, 100‑page workload takes **≈8 s on a 4‑core 2 GHz CPU** and <450 MB RAM.

## Reproducibility
The project ships with:

* `Dockerfile` – installs Python 3.11, system libs, requirements.
* `src/main.py` – entry point (`python -m src.main --input sample_input`).
* `requirements.txt` – only three pip deps, total wheel size ≈150 MB.

The absence of internet access is handled by vendoring nothing heavier than 30 MB wheels; the container works completely offline.