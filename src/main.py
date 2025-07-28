import argparse
import json
import datetime
import re
from pathlib import Path

import fitz  
from sentence_transformers import SentenceTransformer, util


# ─── Load BGE Semantic Model ────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(MODEL_NAME)


# ─── PDF → Pages Extraction ─────────────────────────────────────────────────
def extract_pages(pdf_path):
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text")
            text = text.replace("\u00a0", " ").strip()
            pages.append({"page_number": page_num + 1, "text": text})
    return pages


def clean_line(line):
    return re.sub(r'\s+', ' ', line).strip()


def get_section_title(page_text):
    for line in page_text.split("\n"):
        line = clean_line(line)
        if line:
            return line[:120]
    return ""


# ─── Semantic Ranking ───────────────────────────────────────────────────────
def rank_pages_semantic(pages, query, top_k=5):
    corpus = [p["text"] for p in pages]
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    query_embedding = model.encode([query], convert_to_tensor=True)[0]

    sims = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
    top_idxs = sims.argsort()[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idxs, start=1):
        p = pages[idx].copy()
        p["importance_rank"] = rank
        p["similarity"] = float(sims[idx])
        results.append(p)
    return results


def refine_text_semantic(page_text, query, max_chars=400):
    paras = [para.strip() for para in page_text.split("\n") if para.strip()]
    if not paras:
        return page_text[:max_chars]

    para_embeddings = model.encode(paras, convert_to_tensor=True)
    query_embedding = model.encode([query], convert_to_tensor=True)[0]

    sims = util.cos_sim(query_embedding, para_embeddings)[0].cpu().numpy()
    best_idx = int(sims.argmax())
    best_para = paras[best_idx]
    return best_para[:max_chars]


# ─── Main Pipeline ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Persona-Driven Document Intelligence")
    parser.add_argument("--input", required=True,
                        help="Path to folder with persona.json, job.json, PDFs/")
    parser.add_argument("--output", default="results", help="Output folder path")
    args = parser.parse_args()

    input_path = Path(args.input)
    persona_path = input_path / "persona.json"
    job_path = input_path / "job.json"
    pdf_dir = input_path / "PDFs"
    assert pdf_dir.exists(), f"PDFs folder not found: {pdf_dir}"

    # ─ Persona & Job strings ────────────────────────────────────────────────
    if persona_path.exists():
        try:
            persona = json.loads(persona_path.read_text())
            persona_str = persona.get("persona") or persona.get("role") or str(persona)
        except:
            persona_str = persona_path.read_text().strip()
    else:
        persona_str = input_path.name

    if job_path.exists():
        try:
            job = json.loads(job_path.read_text())
            job_str = job.get("job_to_be_done") or job.get("job") or str(job)
        except:
            job_str = job_path.read_text().strip()
    else:
        job_str = "Summarise key information"

    query = f"{persona_str}. {job_str}"

    # ─ Extract & Rank ────────────────────────────────────────────────────────
    extracted_sections = []
    subsection_analysis = []

    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        pages = extract_pages(pdf_file)
        top_pages = rank_pages_semantic(pages, query, top_k=2)
        for rp in top_pages:
            extracted_sections.append({
                "document": pdf_file.name,
                "section_title": get_section_title(rp["text"]),
                "importance_rank": rp["importance_rank"],
                "page_number": rp["page_number"],
                "similarity_score": rp["similarity"]
            })
            refined = refine_text_semantic(rp["text"], query)
            subsection_analysis.append({
                "document": pdf_file.name,
                "refined_text": refined,
                "page_number": rp["page_number"]
            })

    # ─ Global Re-Ranking & Trim ─────────────────────────────────────────────
    extracted_sections = sorted(
        extracted_sections,
        key=lambda x: x["similarity_score"],
        reverse=True
    )[:5]
    subsection_analysis = subsection_analysis[:5]

    # ─ Build Output JSON ─────────────────────────────────────────────────────
    metadata = {
        "input_documents": [p.name for p in sorted(pdf_dir.glob("*.pdf"))],
        "persona": persona_str,
        "job_to_be_done": job_str,
        "processing_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }

    output = {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "challenge1b_output.json"
    out_path.write_text(json.dumps(output, indent=4))
    print(f"Output written to {out_path}")


if __name__ == "__main__":
    main()
